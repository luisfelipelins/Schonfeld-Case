import pandas as pd
import numpy as np
from config import DATA_FACTORS
from config import DATA_FINAL

def build_strategy(factor, rule, start, freq='ME', side='long_short', cap=3.0, verbose=False):
    """
    Allocation the factor implies on every business day, for one factor and one filer universe.

    The position taken in a security is proportional to how far its factor stands from the cross-sectional average
    of the date, which is the portfolio whose payoff is the coefficient of a cross-sectional regression of returns
    on the factor. Weighting this way means testing the signal and running the strategy are the same computation:
    the regression does not decide the weights, it reports what the weights earned.

    On each rebalance date the prevailing quarter T is the one the factor file already carries, so the 45-day
    convention is inherited rather than recomputed. Everything the date reads is public by then: the factor is
    built from filings already delivered, and the return the coefficient is measured against closed 45 days
    earlier. Securities with no price are dropped, since a position cannot be held in what cannot be valued.

    Parameters
    ----------
    factor : str
        One of the factor files in DATA_FACTORS: 'net_buying', 'breadth_change', 'conviction', 'centrality'.
    rule : str
        Filer universe the factor was built on: 'aum', 'cv_high', 'cv_low', 'conc'.
    start : str
        First date the strategy is allowed to allocate on.
    freq : str, optional
        Rebalance frequency as a pandas offset alias, taking the last decision date of each period. 'ME' is month
        end, 'QE' quarter end, 'W' weekly, by default 'ME'
    side : str, optional
        'long_short' holds the whole cross-section, long above the average factor and short below, netting to zero
        with gross exposure of 1. 'long_only' keeps the long leg alone, normalised to 1, by default 'long_short'
    cap : float, optional
        Standard deviations the standardised factor is capped at before it becomes a position, so that a name at
        the ceiling of the cross-section is held as an extreme name rather than as the whole book. Set to 0 to
        weight on the raw distance, by default 3.0
    verbose : bool, optional
        Prints the shape of the resulting book. Off by default, since the grid builds 96 of these, by default False

    Returns
    -------
    w : pandas.DataFrame
        One row per business day and one column per security, holding the fraction of the portfolio allocated to
        it. The allocation is held unchanged between rebalance dates.
    """

    fac   = pd.read_parquet(DATA_FACTORS / f'{factor}_{rule}.parquet').dropna(subset=[factor])
    close = pd.read_parquet(DATA_FINAL / 'spx_close.parquet')
    cmap  = pd.read_parquet(DATA_FINAL / 'spx_cusip.parquet')

    # Prices are indexed by ticker and the factor by cusip, so the quarter's return is built on the ticker side and
    # carries the cusip the name had at the close of that quarter. Empty rows are dropped before the reindex: the
    # price file runs on business days, so a quarter ending on a holiday would otherwise be read off a blank row,
    # and a blank end kills its own quarter and the next one
    quarters = pd.DatetimeIndex(np.sort(fac['ref_date'].unique()))

    r = close.dropna(how='all').reindex(quarters, method='ffill').pct_change(fill_method=None).stack().rename('ret')
    c = cmap.reindex(quarters, method='ffill').stack().rename('cusip')

    ret = pd.concat([r, c], axis=1).dropna().rename_axis(['ref_date','ticker']).reset_index()
    ret = ret.groupby(['ref_date','cusip'], as_index=False)['ret'].mean()

    panel = fac.merge(ret, on=['ref_date','cusip'], how='left')

    dates = pd.DatetimeIndex(np.sort(fac['date'].unique()))
    dates = dates[dates >= pd.Timestamp(start)]
    rebal = pd.DatetimeIndex(dates.to_series().resample(freq).last().dropna())

    rows, held, coef = [], [], []
    for d in rebal:
        g = panel.loc[panel['date'].eq(d)].dropna(subset=['ret'])
        if len(g) < 3: continue

        # Standardised inside the date, so the position is not decided by the few names whose raw factor is orders
        # of magnitude above the rest, and so the four factors are held on a comparable scale. The cap then keeps
        # the ordering while refusing the cardinal distance behind it: a singular vector has no units, so how many
        # times larger the leading name is than the second carries no economic reading and should not size a bet
        z = (g[factor] - g[factor].mean()) / g[factor].std()
        if not np.isfinite(z).all(): continue

        if cap: z = z.clip(-cap, cap)

        x = z / (z ** 2).sum()

        # The payoff of x over the quarter just closed is the regression coefficient itself, kept as a diagnostic
        coef.append(float((x * g['ret']).sum()))

        w = x / x.abs().sum() if side == 'long_short' else x.clip(lower=0) / x.clip(lower=0).sum()

        held.append(int((w != 0).sum()))
        rows.append(pd.Series(w.to_numpy(), index=g['cusip'].to_numpy(), name=d))

    w = pd.DataFrame(rows).fillna(0.0).sort_index()

    grid = pd.bdate_range(w.index.min(), dates.max())
    w    = w.reindex(grid, method='ffill').rename_axis('date')

    coef = pd.Series(coef)

    if verbose:
        print(f"\nStrategy built | {factor} on {rule} | {side} | {len(rows):,} rebalances at {freq} | {len(w):,} business days")
        print(f"  from {w.index.min().date()} to {w.index.max().date()} | {w.shape[1]:,} securities ever held")
        print(f"  names held per rebalance: min {min(held):,} | median {int(np.median(held)):,} | max {max(held):,} of {panel.groupby('date').size().median():,.0f} in the index")
        print(f"  net exposure {w.sum(axis=1).median():.3f} | gross {w.abs().sum(axis=1).median():.3f} | largest position {w.abs().max().max()*100:.2f}%")
        print(f"  regression coefficient on the quarter just closed: median {coef.median()*100:>6.3f}% | positive on {coef.gt(0).mean()*100:.0f}% of rebalances")

    return w

def contributions(w, cost=0.0, ret=None, freq='YE'):
    """
    Breaks the result of a strategy down into what each security contributed, period by period.

    A security contributes what its own return earned on the money actually sitting in it, which is
    NAV * weight * return on each day, less what its own trading was charged. Measured this way the parts add up:
    summing every security over every period gives the final NAV minus the unit of capital the strategy started
    with. Attributing percentage returns instead would not add up, since a percentage earned early compounds into
    everything that follows.

    Parameters
    ----------
    w : pandas.DataFrame
        Allocation as build_strategy returns it.
    cost : float, optional
        Basis points charged on each unit of notional traded, matching the backtest being explained, by default 0.0
    ret : pandas.DataFrame, optional
        Daily security returns as daily_returns builds them, rebuilt here when left out, by default None
    freq : str, optional
        Period the contributions are summed over, as a pandas offset alias. 'YE' is a year, '3YE' three, 'QE' a
        quarter, by default 'YE'

    Returns
    -------
    con : pandas.DataFrame
        One row per period and one column per security, in units of the capital the strategy started with.
    """

    if ret is None: ret = daily_returns()
    ret = ret.reindex(index=w.index, columns=w.columns).fillna(0.0)

    moved = w.diff().abs().sum(axis=1).gt(0)
    moved.iloc[0] = True

    hold, nav, rows = None, 1.0, []
    for t in w.index:
        c = pd.Series(0.0, index=w.columns)

        if hold is not None:
            r  = ret.loc[t]
            rp = float((hold * r).sum())

            # The money in a name is nav * weight, so what it earned for the book is that times its return
            c    = nav * hold * r
            hold = hold * (1 + r) / (1 + rp)
            nav  = nav * (1 + rp)

        # Trading is charged to the name being traded, so the cost lands where it was incurred
        if moved.at[t]:
            traded = (w.loc[t] - (hold if hold is not None else 0.0)).abs()
            c      = c - nav * cost / 1e4 * traded
            nav    = nav - nav * cost / 1e4 * float(traded.sum())
            hold   = w.loc[t]

        rows.append(c.rename(t))

    con = pd.DataFrame(rows).resample(freq).sum()
    con = con.loc[:, con.abs().sum().gt(0)]

    print(f"\nContributions | {con.shape[1]:,} securities | {len(con)} periods | they add up to {con.to_numpy().sum():+.3f} of the initial capital")

    return con

def compare_benchmark(bt, mixes=(0.25, 0.5, 0.75, 1.0), rf=0.0):
    """
    Reads a backtested strategy against the index, on its own, levered to the same volatility, and added on top of
    the index in several proportions.

    A market neutral book and the index are paid for different risks, so their returns are not alternatives to be
    ranked against each other. What decides whether the strategy is worth holding is whether it improves a book
    that already owns the index, and that depends on its correlation more than on its return.

    Parameters
    ----------
    bt : pandas.DataFrame
        Backtest as run_backtest returns it.
    mixes : tuple, optional
        Fractions of the strategy added on top of one unit of the index, by default (0.25, 0.5, 0.75, 1.0)
    rf : float, optional
        Annual risk free rate used in the Sharpe ratio, by default 0.0

    Returns
    -------
    cmp : pandas.DataFrame
        One row per book, carrying its return, volatility, Sharpe, drawdown, correlation to the index and beta.
    """

    b = pd.read_parquet(DATA_FINAL / 'spx_index.parquet')['spx'].pct_change(fill_method=None)
    b = b.reindex(bt.index).fillna(0.0)
    s = bt['ret']

    # Levering to the index's volatility is the only comparison on which the two stand at the same risk, and it is
    # read off the sample rather than chosen
    lever = float(b.std() / s.std())

    books = {'benchmark': b, 'strategy': s, f'strategy at {lever:.2f}x (vol matched)': lever * s}
    for m in mixes: books[f'benchmark + {m:.2f} strategy'] = b + m * s

    rows = []
    for name, r in books.items():
        nav = (1 + r).cumprod()
        ann = float(nav.iloc[-1] ** (252 / len(r)) - 1)
        vol = float(r.std() * np.sqrt(252))

        rows.append({'book'          : name,
                     'ann_return'    : ann,
                     'ann_vol'       : vol,
                     'sharpe'        : (ann - rf) / vol,
                     'max_drawdown'  : float((nav / nav.cummax() - 1).min()),
                     'corr_benchmark': float(r.corr(b)),
                     'beta'          : float(r.cov(b) / b.var())})

    cmp = pd.DataFrame(rows)

    print(f"\nAgainst the benchmark | {len(bt):,} business days | {bt.index.min().date()} to {bt.index.max().date()}")
    print(f"  {'book':<36}{'return':>9}{'vol':>8}{'sharpe':>8}{'max dd':>9}{'corr':>7}{'beta':>7}")
    for r in cmp.itertuples():
        print(f"  {r.book:<36}{r.ann_return*100:>8.2f}%{r.ann_vol*100:>7.2f}%{r.sharpe:>8.2f}{r.max_drawdown*100:>8.1f}%{r.corr_benchmark:>7.2f}{r.beta:>7.2f}")

    return cmp

def daily_returns():
    """
    Daily return of every security, with the cusip it carried on each day on the columns.

    Built once and passed into run_backtest when many strategies are evaluated, since reading the price file and
    rebuilding this frame costs more than the backtest it feeds.

    Returns
    -------
    pandas.DataFrame
        One row per trading day and one column per cusip.
    """

    close = pd.read_parquet(DATA_FINAL / 'spx_close.parquet')
    cmap  = pd.read_parquet(DATA_FINAL / 'spx_cusip.parquet')

    px = close.dropna(how='all')
    rr = px.pct_change(fill_method=None).stack().rename('ret')
    cc = cmap.reindex(px.index, method='ffill').stack().rename('cusip')

    ret = pd.concat([rr, cc], axis=1).dropna().rename_axis(['date','ticker']).reset_index()

    return ret.groupby(['date','cusip'])['ret'].mean().unstack()

def run_backtest(w, cost=0.0, rf=0.0, ret=None, verbose = False):
    """
    Daily return of the allocation, letting the positions drift between rebalance dates.

    The book is bought at the close of a rebalance date and left alone until the next one, so a name that rises
    becomes a larger share of the portfolio on its own and the only turnover is the one the rebalance asks for.
    Weights are fractions of current wealth, which is what makes the level of the portfolio drop out of the
    arithmetic: the whole book is reallocated at each rebalance, gains included and losses deducted, and the
    series is simply compounded.

    Parameters
    ----------
    w : pandas.DataFrame
        Allocation as build_strategy returns it, one row per business day and one column per security. Rows that
        differ from the previous one are read as rebalance dates.
    cost : float, optional
        Basis points charged on each unit of notional traded: the half spread paid for crossing, plus commission
        and impact. Since a round trip sells one name to buy another, it pays this twice, and the charge lands on
        the return of the date the trade is executed. Large US caps sit around 5 bps, by default 0.0
    rf : float, optional
        Annual risk free rate used in the Sharpe ratio, by default 0.0
    ret : pandas.DataFrame, optional
        Daily security returns as daily_returns builds them. Passed in when many strategies are evaluated against
        the same prices, rebuilt here when left out, by default None

    Returns
    -------
    bt : pandas.DataFrame
        One row per business day, carrying the portfolio return net of costs, the compounded value of one unit of
        capital, the drawdown from its running peak, the exposure the drift left the book with, the one-way
        turnover of the date and what it was charged for it.
    """

    if ret is None: ret = daily_returns()

    # A security with no price on a day is held flat rather than dropped, so the book keeps its shape instead of
    # silently reallocating to whatever else happens to be quoted
    ret = ret.reindex(index=w.index, columns=w.columns).fillna(0.0)

    # The allocation is forward filled, so a row that differs from the one before it is a rebalance. The first row
    # is one by definition
    moved = w.diff().abs().sum(axis=1).gt(0)
    moved.iloc[0] = True

    hold, out = None, []
    for t in w.index:
        if hold is None:
            rp, net, gross = 0.0, 0.0, 0.0
        else:
            r     = ret.loc[t]
            rp    = float((hold * r).sum())
            net   = float(hold.sum())
            gross = float(hold.abs().sum())

            # The position of each name grows by its own return while the book grows by the portfolio's, so the
            # share it takes moves by the difference between the two
            hold = hold * (1 + r) / (1 + rp)

        # Set at the close, so the weights decided on a date earn from the next session onwards. The trade is
        # measured against the drifted book, which is what the day actually has to be traded away from, and the
        # opening date pays for the whole book since it is bought from nothing
        if moved.at[t]:
            traded = float((w.loc[t] - (hold if hold is not None else 0.0)).abs().sum())
            hold   = w.loc[t]
        else:
            traded = 0.0

        out.append((rp - cost / 1e4 * traded, net, gross, traded / 2, cost / 1e4 * traded))

    bt = pd.DataFrame(out, index=w.index, columns=['ret','net','gross','turnover','cost'])

    bt['nav']      = (1 + bt['ret']).cumprod()
    bt['drawdown'] = bt['nav'] / bt['nav'].cummax() - 1

    ann   = (1 + bt['ret']).prod() ** (252 / len(bt)) - 1
    vol   = bt['ret'].std() * np.sqrt(252)
    turns = bt.loc[moved, 'turnover']

    if verbose:
        print(f"\nBacktest | {len(bt):,} business days | {int(moved.sum()):,} rebalances | {bt.index.min().date()} to {bt.index.max().date()}")
        print(f"  return {ann*100:>7.2f}% a year | vol {vol*100:>6.2f}% | sharpe {(ann - rf)/vol:>5.2f} | final nav {bt['nav'].iloc[-1]:,.2f}")
        print(f"  max drawdown {bt['drawdown'].min()*100:>6.1f}% | best day {bt['ret'].max()*100:>5.2f}% | worst day {bt['ret'].min()*100:>6.2f}% | positive days {bt['ret'].gt(0).mean()*100:.1f}%")
        print(f"  turnover per rebalance: median {turns.median()*100:>5.1f}% | max {turns.max()*100:.1f}% | {turns.sum()*100/len(bt)*252:,.0f}% a year")
        print(f"  costs at {cost:.0f} bps: {bt['cost'].sum()*100/len(bt)*252:>5.2f}% a year | {bt['cost'].sum()*100:.2f}% over the whole sample")
        # The opening row carries no position, since the first allocation is only set at its close
        e = bt.iloc[1:]
        print(f"  drift leaves net exposure between {e['net'].min():>6.3f} and {e['net'].max():.3f} | gross between {e['gross'].min():.3f} and {e['gross'].max():.3f}")

    return bt
