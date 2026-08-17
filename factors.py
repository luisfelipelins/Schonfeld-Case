import pandas as pd
import numpy as np
import sys
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
from scipy.stats import spearmanr

# A terminal redraws the line on a carriage return, a notebook only shows it once a newline arrives
_END = '\r' if sys.stdout.isatty() else '\n'

def net_buying(snap, layers):
    """
    Net buying of each security by the filers in the universe: the change in shares held between the prevailing
    quarter and the one before it, aggregated across filers and scaled by the two quarters taken together.

    Shares are used instead of value because value also carries the price move of the quarter, which is not a
    decision of the manager. The scaling leaves the factor bounded in [-1, 1], where +1 is a position the group
    only opened in the prevailing quarter and -1 one it fully left, so entries and exits stay in the sample.

    Parameters
    ----------
    snap : pandas.DataFrame
        Holdings prevailing at the decision date, one row per (cik, ref_date, cusip).
    layers : pandas.DatetimeIndex
        Reference dates carried by snap, most recent first.

    Returns
    -------
    pandas.Series
        Net buying of each security, indexed by cusip.
    """

    if len(layers) < 2: return pd.Series(dtype='float64')

    curr = snap.loc[snap['ref_date'].eq(layers[0])]
    prev = snap.loc[snap['ref_date'].eq(layers[1])]

    # A filer visible in only one of the two quarters would read as a full entry or a full exit; flow measured only over the filers who have reported both sides of it
    both = np.intersect1d(curr['cik'].unique(), prev['cik'].unique())

    s = curr.loc[curr['cik'].isin(both)].groupby('cusip')['sshprnamt'].sum()
    p = prev.loc[prev['cik'].isin(both)].groupby('cusip')['sshprnamt'].sum()

    s, p = s.align(p, fill_value=0.0)
    tot  = s + p

    norm_net_buy = ((s - p) / tot.where(tot.gt(0))).dropna()

    return norm_net_buy

def breadth_change(snap, layers):
    """
    Change in the number of filers holding each security between the prevailing quarter and the one before it,
    scaled by how many filers reported both quarters.

    Every filer counts once, whatever its size, which is what separates this from net_buying: a name that forty
    small managers entered and one large manager left moves the two factors in opposite directions. The scaling is
    what keeps the factor comparable over time, since the universe grows from around 350 to 870 filers.

    Parameters
    ----------
    snap : pandas.DataFrame
        Holdings prevailing at the decision date, one row per (cik, ref_date, cusip).
    layers : pandas.DatetimeIndex
        Reference dates carried by snap, most recent first.

    Returns
    -------
    pandas.Series
        Change in the number of holders of each security, indexed by cusip.
    """

    if len(layers) < 2: return pd.Series(dtype='float64')

    # Holding is having shares of it, so a position reported flat is not a holder
    held = snap.loc[snap['sshprnamt'].gt(0)]

    curr = held.loc[held['ref_date'].eq(layers[0])]
    prev = held.loc[held['ref_date'].eq(layers[1])]

    # Same restriction net_buying makes: a filer seen in one quarter only would read as every one of its names
    # having been opened or dropped at once
    both = np.intersect1d(curr['cik'].unique(), prev['cik'].unique())
    if len(both) == 0: return pd.Series(dtype='float64')

    n = curr.loc[curr['cik'].isin(both)].groupby('cusip')['cik'].nunique()
    o = prev.loc[prev['cik'].isin(both)].groupby('cusip')['cik'].nunique()

    n, o = n.align(o, fill_value=0)

    delta_breadth = (n - o) / len(both)

    return delta_breadth

def conviction(snap, layers):
    """
    Average weight each security has inside the books of the filers holding it, in the prevailing quarter.

    Separates a name that is a real bet from one that is a leftover: at equal number of holders, a security taking
    4% of the books that own it is not the same as one taking 0.05%. Being a level and not a change, it reads the
    prevailing quarter alone and does not restrict filers to those who also reported the quarter before.

    Parameters
    ----------
    snap : pandas.DataFrame
        Holdings prevailing at the decision date, one row per (cik, ref_date, cusip), carrying the weight w each
        position has inside its own filing.
    layers : pandas.DatetimeIndex
        Reference dates carried by snap, most recent first.

    Returns
    -------
    pandas.Series
        Average weight of each security among its holders, indexed by cusip.
    """

    # A position filed at zero value weighs zero, which would read as the lowest conviction there is rather than
    # as the reporting gap it actually is
    curr = snap.loc[snap['ref_date'].eq(layers[0]) & snap['w'].gt(0)]

    mean_weight = curr.groupby('cusip')['w'].mean()

    return mean_weight

def build_factors(df, factors, universe, spx, depth=2, lag=45):
    """
    Walks the holdings universe forward in time and evaluates every factor on the information available at each
    decision date, producing one value per (cusip, date) for the securities in the index that day.

    Parameters
    ----------
    df : pandas.DataFrame
        Holdings and filers universe, one row per (cik, ref_date, vintage_date, cusip).
    factors : dict
        Maps a factor name to a function f(snap, layers) returning a pandas.Series indexed by cusip.
    universe : pandas.DataFrame
        Filer universe as selected on each date, one row per (vintage_date, ref_date, cik).
    spx : pandas.DataFrame
        Index composition by business day, tickers on the columns and the cusip of each member in the cells.
    depth : int, optional
        Number of quarters kept in the snapshot, counting back from the prevailing one. Sets how far a factor is
        allowed to look, by default 2
    lag : int, optional
        Days after a reference date at which its filing deadline passes, which is when the quarter becomes the
        prevailing one. 45 is the statutory 13F deadline, by default 45

    Returns
    -------
    fac : pandas.DataFrame
        One row per (cusip, date), carrying the prevailing quarter and one column per factor.
    """

    keys = ['cik','ref_date']

    quarters  = pd.DatetimeIndex(np.sort(df['ref_date'].unique()))
    deadlines = quarters + pd.Timedelta(days=lag)

    # The factor moves when a new filing lands or the current quarter rolls. Vintages older than the first deadline
    # are walked as well, so that the first quarter is already fully assembled by the time it becomes the prevailing one
    grid = pd.DatetimeIndex(np.sort(df['vintage_date'].unique())).union(deadlines)

    arrivals = dict(list(df.groupby('vintage_date', sort=False)))

    # The holdings file only carries the union of every filer the rule ever selected, so the selection of the day has
    # to be reapplied here. Without it a filer that only reaches the top of the ranking years from now is already
    # being read today, which is the ranking of the future deciding the cross-section of the present
    picks = universe.groupby('vintage_date')['cik'].apply(frozenset)

    # Index membership as it stood on each business day. stack drops the cells of the tickers outside the index, so
    # what is left for a date is the cusip of everything it held that day
    members = spx.stack().groupby(level=0).apply(frozenset)

    state, out = {}, []
    for i, d in enumerate(grid, 1):

        # The state is the pile of filings prevailing at date d: for each (cik, ref_date), the last vintage seen up to here, so a later vintage replaces the earlier one in place
        if d in arrivals:
            for k, g in arrivals[d].groupby(keys, sort=False): state[k] = g

        # j denotes the index of the latest previous deadline, absorving the information for all quarters up to 'depth' quarters ago
        j      = deadlines.searchsorted(d, side='right') - 1
        if j < 0: continue
        layers = quarters[max(0, j - depth + 1): j + 1][::-1]

        m = picks.index.searchsorted(d, side='right') - 1
        if m < 0: continue

        n = members.index.searchsorted(d, side='right') - 1
        if n < 0: continue

        for k in [k for k in state if k[1] < layers[-1]]: del state[k]

        # Snap contains all the fillings available up to d for each vintage
        snap = pd.concat([g for k, g in state.items() if k[1] in layers and k[0] in picks.iat[m]], ignore_index=True)
        if snap.empty: continue

        # Weight of each position inside its own filing, computed once for every factor that needs it. The state
        # holds whole filings, so the total below is the filer's entire book and not just its part in any cut
        snap['w'] = snap['value'] / snap.groupby(keys)['value'].transform('sum')

        # Every factor reads the whole book, and only the output is cut to what the strategy can trade. Cutting the
        # snapshot instead would normalise weights against an index-only book and shrink the filer sets the change
        # factors are built on, so the cut belongs here and nowhere earlier
        vals = pd.DataFrame({name: f(snap = snap, layers = layers) for name, f in factors.items()})
        vals = vals.loc[vals.index.isin(members.iat[n])]
        if vals.empty: continue

        out.append(vals.rename_axis('cusip').reset_index().assign(date=d, ref_date=layers[0]))

        print(f"    {i}/{len(grid)}: {d.date()} | {len(picks.iat[m]):,} filers selected | {len(vals):,} securities".ljust(130), end=_END, flush=True)
    print()

    fac = pd.concat(out, ignore_index=True)[['cusip','date','ref_date'] + list(factors)]

    print(f"\nFactors built | {len(fac):,} rows | {fac['cusip'].nunique():,} CUSIPs | {fac['date'].nunique():,} decision dates | {fac['ref_date'].nunique()} quarters")
    print(f"  from {fac['date'].min().date()} to {fac['date'].max().date()} | {fac.groupby('date').size().median():,.0f} securities per date (median)")
    for name in factors:
        print(f"  {name:<14} {fac[name].notna().mean()*100:>5.1f}% non-null | p10 {fac[name].quantile(.1):>7.3f} | median {fac[name].median():>7.3f} | p90 {fac[name].quantile(.9):>7.3f}")

    return fac

FACTORS = {'net_buying'    : net_buying,
           'breadth_change': breadth_change,
           'conviction'    : conviction}

def build_centrality(df, universe, spx, min_positions=10, min_aum=100e6, k=1, lag=45):
    """
    Centrality of each security in the co-holding network of the filers in the universe, walked forward in time on
    the same decision dates the other factors use.

    On each date the prevailing quarter is pivoted into a matrix of filers by securities carrying the weight each
    position has inside its filer's whole book. The columns are centred before the decomposition: on the raw matrix
    the leading singular vector converges to the average portfolio, which is roughly the market, and the factor
    would be a market cap proxy. Centred, the cell is how far a filer sits from the average filer on that security,
    and the leading component is the axis along which the universe splits.

    The factor is the magnitude of the loading on that axis, not the loading itself. A singular vector and its
    negative describe the same axis, so which of its two ends comes out positive is not decided by the data, and
    reading the sign would need it anchored to something. Anchored to the previous date it breaks at the roll,
    where the new quarter carries around 70% of the reports the old one had accumulated and the two dates have
    least in common. Taking the magnitude drops the question: high means the security is where the dominant
    pattern departs most from the average filer, in either direction, and low means everyone looks alike on it.

    Parameters
    ----------
    df : pandas.DataFrame
        Holdings and filers universe, one row per (cik, ref_date, vintage_date, cusip).
    universe : pandas.DataFrame
        Filer universe as selected on each date, one row per (vintage_date, ref_date, cik).
    spx : pandas.DataFrame
        Index composition by business day, tickers on the columns and the cusip of each member in the cells.
    min_positions : int, optional
        Least positions inside the index a filer must hold to enter the matrix, by default 10
    min_aum : float, optional
        Least total book value in dollars a filer must report to enter the matrix, by default 100e6
    k : int, optional
        Components taken from the decomposition. Only the leading one becomes the factor, by default 1
    lag : int, optional
        Days after a reference date at which its filing deadline passes, which is when the quarter becomes the
        prevailing one. 45 is the statutory 13F deadline, by default 45

    Returns
    -------
    fac : pandas.DataFrame
        One row per (cusip, date), carrying the prevailing quarter, the raw centrality and its z score.
    hold : pandas.DataFrame
        One row per (cik, date), carrying the same for the filer side of the decomposition.
    diag : pandas.DataFrame
        One row per date with the shape of the matrix, what each filter removed, how much of the variance the
        leading component carries, and the correlations that say whether the factor is adding anything.
    """

    quarters  = pd.DatetimeIndex(np.sort(df['ref_date'].unique()))
    deadlines = quarters + pd.Timedelta(days=lag)

    grid = pd.DatetimeIndex(np.sort(df['vintage_date'].unique())).union(deadlines)

    arrivals = dict(list(df.groupby('vintage_date', sort=False)))
    picks    = universe.groupby('vintage_date')['cik'].apply(frozenset)
    members  = spx.stack().groupby(level=0).apply(frozenset)

    state, out, side, diag = {}, [], [], []
    prev_v, last_v, last_q = None, None, None

    for i, d in enumerate(grid, 1):
        # Keyed by quarter as well as filer, so a book filed early for the next quarter does not overwrite the one
        # being read now. The prune then drops whatever fell behind the prevailing quarter
        if d in arrivals:
            for key, g in arrivals[d].groupby(['cik','ref_date'], sort=False): state[key] = g

        j = deadlines.searchsorted(d, side='right') - 1
        if j < 0: continue
        q = quarters[j]

        m = picks.index.searchsorted(d, side='right') - 1
        n = members.index.searchsorted(d, side='right') - 1
        if m < 0 or n < 0: continue

        for key in [key for key in state if key[1] < q]: del state[key]

        sel  = picks.iat[m]
        snap = pd.concat([g for key, g in state.items() if key[1] == q and key[0] in sel], ignore_index=True)
        if snap.empty: continue

        assert snap['vintage_date'].le(d).all(), f'{d.date()}: filing from the future in the snapshot'

        # Book of the filer, whole, before anything is cut to the index. A filer holding 30% of its book inside the
        # index contributes less to the decomposition than one holding 90%, and that is the intent
        fund = snap.groupby('cik')['value'].sum()
        snap['w'] = snap['value'] / snap['cik'].map(fund)

        held = snap.loc[snap['cusip'].isin(members.iat[n])]
        if held.empty: continue

        # Absence of a position is a weight of zero, which is information, not a missing value
        b = held.pivot_table(index='cik', columns='cusip', values='w', aggfunc='sum', fill_value=0.0)
        assert not b.isna().to_numpy().any(), f'{d.date()}: null in the matrix'

        r0 = len(b)
        b  = b.loc[b.sum(axis=1).gt(0)]
        r1 = len(b)
        b  = b.loc[b.gt(0).sum(axis=1).ge(min_positions)]
        r2 = len(b)
        b  = b.loc[fund.reindex(b.index).ge(min_aum)]
        r3 = len(b)

        c0 = b.shape[1]
        b  = b.loc[:, b.sum(axis=0).gt(0)]

        if b.shape[0] <= k + 1 or b.shape[1] <= k + 1: continue

        x  = b.to_numpy()
        bc = x - x.mean(axis=0)

        uu, ss, vt = svds(bc, k=k)
        order      = np.argsort(ss)[::-1]
        uu, ss, vt = uu[:, order], ss[order], vt[order]

        # v and -v are both singular vectors, so which end of the axis comes out positive is not decided by the data.
        # The factor is how strongly a security loads on the axis, so it takes the magnitude and the question never
        # arises. Reading the sign instead would need it anchored to something, and anchoring it to the previous date
        # fails exactly at the roll, where the matrix is at its thinnest and the two dates have least in common
        v = pd.Series(vt[0], index=b.columns).abs()
        u = pd.Series(uu[:, 0], index=b.index).abs()

        vz = (v - v.mean()) / v.std()
        uz = (u - u.mean()) / u.std()

        holders = held.loc[held['sshprnamt'].gt(0)].groupby('cusip')['cik'].nunique().reindex(v.index).fillna(0)

        rec = {'date'          : d,
               'ref_date'      : q,
               'n_filers'      : b.shape[0],
               'n_cusips'      : b.shape[1],
               'cut_zero_w'    : r0 - r1,
               'cut_positions' : r1 - r2,
               'cut_aum'       : r2 - r3,
               'cut_cols'      : c0 - b.shape[1],
               'density'       : float((x > 0).mean()),
               'var_explained' : float(ss[0] ** 2 / (bc ** 2).sum()),
               'spearman_breadth': float(spearmanr(vz.to_numpy(), holders.to_numpy()).statistic)}

        # Consecutive dates differ by whatever landed that day, so the day to day correlation reads as stability and
        # the one against the previous quarter as how much of the factor survives a whole reporting cycle
        for name, ref in [('corr_prev_date', prev_v), ('corr_prev_quarter', last_v)]:
            if ref is None:
                rec[name] = np.nan
            else:
                ix        = v.index.intersection(ref.index)
                rec[name] = float(spearmanr(v.loc[ix].to_numpy(), ref.loc[ix].to_numpy()).statistic) if len(ix) > 2 else np.nan

        # The three heavy checks answer questions about the construction, not about the day, and their answer barely
        # moves inside a quarter. Running them once per roll keeps 2,300 dates from paying for three extra decompositions
        if q != last_q:
            sv            = svdvals(bc)
            rec['eff_rank'] = float(sv.sum() ** 2 / (sv ** 2).sum())

            for name, mat in [('corr_uncentered', x), ('corr_rownorm', (x / np.linalg.norm(x, axis=1, keepdims=True)))]:
                y  = mat if name == 'corr_uncentered' else mat - mat.mean(axis=0)
                _, s2, w2 = svds(y, k=1)
                alt = pd.Series(w2[0], index=b.columns).abs()
                rec[name] = float(spearmanr(alt.to_numpy(), v.to_numpy()).statistic)

            last_v = v.copy()
            last_q = q

        out.append(pd.DataFrame({'cusip': v.index, 'date': d, 'ref_date': q, 'centrality': v.to_numpy(), 'centrality_z': vz.to_numpy()}))
        side.append(pd.DataFrame({'cik': u.index, 'date': d, 'ref_date': q, 'u_raw': u.to_numpy(), 'u_z': uz.to_numpy()}))
        diag.append(rec)

        prev_v = v

        print(f"    {i}/{len(grid)}: {d.date()} | B {b.shape[0]}x{b.shape[1]} | var {rec['var_explained']*100:.1f}%".ljust(130), end=_END, flush=True)
    print()

    fac  = pd.concat(out, ignore_index=True)
    hold = pd.concat(side, ignore_index=True)
    diag = pd.DataFrame(diag).set_index('date')

    print(f"\nCentrality built | {len(fac):,} rows | {fac['cusip'].nunique():,} CUSIPs | {fac['date'].nunique():,} decision dates | {fac['ref_date'].nunique()} quarters")
    print(f"  B: {diag['n_filers'].median():,.0f} filers x {diag['n_cusips'].median():,.0f} cusips (median) | density {diag['density'].median()*100:.1f}%")
    print(f"  removed per date (median): zero weight {diag['cut_zero_w'].median():.0f} | under {min_positions} positions {diag['cut_positions'].median():.0f} | under {min_aum/1e6:.0f}mn {diag['cut_aum'].median():.0f} | empty columns {diag['cut_cols'].median():.0f}")
    print(f"  leading component carries {diag['var_explained'].median()*100:.1f}% of the variance (median) | effective rank {diag['eff_rank'].median():.1f}")
    print(f"\n  spearman vs holder count : median {diag['spearman_breadth'].median():.3f} | p90 {diag['spearman_breadth'].quantile(.9):.3f}")
    print(f"  spearman vs uncentered   : median {diag['corr_uncentered'].median():.3f}")
    print(f"  spearman vs row normalised: median {diag['corr_rownorm'].median():.3f}")
    print(f"  spearman vs previous date : median {diag['corr_prev_date'].median():.3f} | vs previous quarter {diag['corr_prev_quarter'].median():.3f}")

    return fac, hold, diag
