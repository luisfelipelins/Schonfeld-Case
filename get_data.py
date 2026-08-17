import pandas as pd
import numpy as np
import requests
import io
import sys
import zipfile
import time
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from config import DATA_ZIP
from config import DATA_RAW
from config import DATA_INT
from config import DATA_FINAL
from config import START_REF
import yfinance as yf
import yfinance.data

# A terminal redraws the line on a carriage return, a notebook only shows it once a newline arrives
_END = '\r' if sys.stdout.isatty() else '\n'

_UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
yfinance.data.YfData.user_agent_headers = _UA
yf.utils.user_agent_headers = _UA

def sec_get_links(url):
    """
    For any SEC website page with downloadable data in .zip format, gets the urls for these downloadable files.
    Used to get 13F bulk data and Fails-to-Deliver data used in the S&P500 ticker to CUSIP mapping.

    Parameters
    ----------
    url : str
        Main webpage url, where the rest of the.zip links are

    Returns
    -------
    links : list
        List of .zip urls to be fetched
    """
    
    print(f'Starting to fetch {url} for the .zip files...')

    h   = {"User-Agent": "Luis Felipe Lins lf.valadares@hotmail.com"}
    r   = requests.get(url, headers=h)
    
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".zip"):
            links.append(urljoin(url, href))
    print('Fetching done!')
    print('\n')

    return links

def download_13f():
    """
    Downloads the 13F bulk data .zip folders from the SEC EDGAR website. Saves them to DATA_ZIP.
    Each .zip represents one quarter.

    Returns
    -------
    sorted_zip_list : list
        List with the sorted values of the .zip files just downloaded.

    Raises
    ------
    RuntimeError
        If a given .zip file couldn't be downloaded. It may be related to connection issues or trying too many times downloading something from SEC EDGAR's website.
    """

    url   = 'https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets'
    links = sec_get_links(url=url)
    h     = {"User-Agent": "Luis Felipe Lins lf.valadares@hotmail.com"}

    for i, link in enumerate(links, 1):
        path = DATA_ZIP / link.rsplit('/',1)[1]

        # SEC drops connections mid-stream, so if an error, try again. Don't download .zips already downloaded
        for attempt in range(5):
            if path.exists(): break
            try:
                z = requests.get(link, headers=h, timeout=180)
                z.raise_for_status()

                temp = path.with_suffix('.part')
                temp.write_bytes(z.content)
                with zipfile.ZipFile(temp): pass
                temp.rename(path)
            except Exception as err:
                print(f"    {path.name}: {type(err).__name__}, retry {attempt+1}/5".ljust(130))
                time.sleep(2 ** attempt)

        if not path.exists(): raise RuntimeError(f"could not download {link}")

        print(f"    {i}/{len(links)}: {path.name} on disk".ljust(130), end=_END, flush=True)
    print()

    sorted_zip_list = sorted(DATA_ZIP.glob('*.zip'))

    return sorted_zip_list

def importer_13f(path):
    """
    From the .zip files, reads the folders and take only the important datasets we'll need in our analysis.

        1. raw   : one row per (cusip, ref_date, vintage_date, cik, amendment), value and sshprnamt exactly as filed
        2. filing: one row per filing, carrying the manager name the coverpage states
        3. issued: the issuer name last seen for each cusip


    Parameters
    ----------
    path : pathlib.WindowsPath
        Windowns Path of the .zip folder to be opened.

    Returns
    -------
    filing : pandas.DataFrame
        Dataframe with every filling mapped by firm (CIK), reference date, vintage date, and type (ammendment or not)
    
    issuer : pandas.DataFrame
        Dataframe with the CUISP list of different assets
    
    len(df) : int
        Number of lines in the raw data
    """

    stem   = path.stem.replace('_form13f','')

    raw    = DATA_RAW / f"13f_raw_{stem}.parquet"
    filed  = DATA_RAW / f"13f_filing_{stem}.parquet"
    issued = DATA_RAW / f"13f_issuer_{stem}.parquet"

    # If the raw files already exist, we won't run the function again
    if raw.exists() and filed.exists() and issued.exists():
        return pd.read_parquet(filed), pd.read_parquet(issued), pq.ParquetFile(raw).metadata.num_rows

    # Selecting only the interesting dataframes: "SUBMISSION.TSV", "COVERPAGE.TSV", "INFOTABLE.TSV"
    df_dict = {}
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            name = n.rsplit("/", 1)[-1].upper()
            if name in ("SUBMISSION.TSV", "COVERPAGE.TSV", "INFOTABLE.TSV"):

                cols  = ['ACCESSION_NUMBER','NAMEOFISSUER','TITLEOFCLASS','CUSIP','VALUE','SSHPRNAMT','SSHPRNAMTTYPE','PUTCALL'] if name == "INFOTABLE.TSV" else None
                types = {'ACCESSION_NUMBER':str,'NAMEOFISSUER':str,'CUSIP':str,'VALUE':str,'SSHPRNAMT':str,'TITLEOFCLASS':'category','SSHPRNAMTTYPE':'category','PUTCALL':'category'} if name == "INFOTABLE.TSV" else str

                with zf.open(n) as f: df = pd.read_csv(f, sep="\t", dtype=types, encoding="latin-1", usecols=cols)

                match name:
                    case "INFOTABLE.TSV":
                        deriv = [c for c in df['TITLEOFCLASS'].cat.categories if c.strip().upper() in ('CALL','PUT')]
                        keep  = df['PUTCALL'].isna() & df['SSHPRNAMTTYPE'].eq('SH') & ~df['TITLEOFCLASS'].isin(deriv)
                        df    = df.loc[keep].drop('PUTCALL',axis=1)
                    case "COVERPAGE.TSV":
                        df = df[['ACCESSION_NUMBER','FILINGMANAGER_NAME','ISAMENDMENT','AMENDMENTTYPE']]
                    case "SUBMISSION.TSV":
                        df = df.loc[df['SUBMISSIONTYPE'].str[:6]=='13F-HR'].drop('SUBMISSIONTYPE',axis=1)
                        
                df_dict[name] = df

    # Merging the dfs to the infotable one. Each row is an information of a given asset (CUSIP), filer (CIK), reference date and a vintage date depending on when it was filled and if it's an ammendment or not
    df = (df_dict["INFOTABLE.TSV"].join(df_dict["SUBMISSION.TSV"].set_index("ACCESSION_NUMBER"), 
                                        on="ACCESSION_NUMBER", 
                                        how="inner", 
                                        validate="m:1").join(df_dict["COVERPAGE.TSV"].set_index("ACCESSION_NUMBER"),  
                                                             on="ACCESSION_NUMBER", 
                                                             how="left",  
                                                             validate="m:1"))

    df['ref_date']     = pd.to_datetime(df['PERIODOFREPORT'], format='%d-%b-%Y')
    df['vintage_date'] = pd.to_datetime(df['FILING_DATE'], format='%d-%b-%Y')

    df = df.drop(['FILING_DATE','PERIODOFREPORT'],axis=1)

    df['CUSIP']        = df['CUSIP'].str.strip().str.upper()
    df['VALUE']        = pd.to_numeric(df['VALUE'], errors='coerce')
    df['SSHPRNAMT']    = pd.to_numeric(df['SSHPRNAMT'], errors='coerce')
    df['AMENDMENT']    = df['AMENDMENTTYPE'].fillna('UNSPECIFIED').where(df['ISAMENDMENT'].eq('Y'), 'ORIGINAL')

    # Our index will be (CUSIP, ref_date, vintage_date, CIK), so adding up managers inside a same CIK (investment firm)
    index = ['CUSIP','ref_date','vintage_date','CIK']
    df = df.groupby(index + ['AMENDMENT'], as_index=False).agg(VALUE               = ('VALUE','sum'),
                                                                SSHPRNAMT          = ('SSHPRNAMT','sum'),
                                                                NAMEOFISSUER       = ('NAMEOFISSUER','first'),
                                                                FILINGMANAGER_NAME = ('FILINGMANAGER_NAME','first'))
    df.columns = [i.lower() for i in df.columns]

    keys   = ['cik','ref_date','vintage_date','amendment']
    filing = df.groupby(keys, as_index=False).agg(filingmanager_name = ('filingmanager_name','first'))
    issuer = df.sort_values('ref_date').drop_duplicates('cusip', keep='last')[['cusip','ref_date','nameofissuer']]

    df = df.drop(['nameofissuer','filingmanager_name'], axis=1)

    df.to_parquet(raw, index=False)
    filing.to_parquet(filed, index=False)
    issuer.to_parquet(issued, index=False)

    return filing, issuer, len(df)

def clean_units(files, names, ceiling=1e7, lag=45):
    """
    Function used to clean the units of wrongly reported 13F data among filers. 

    It uses the expanding (only up to the available vintage) median implicit price of the CUSIP among filers (CIK) to check if a filer has reported its holdings wrong (e.g. in thousands).

    Adjusts the data based on how off they are. Heavy and inneficient function.  

    Parameters
    ----------
    files : list
        List of WindownsPaths of the raw holdings files.
    names : pandas.DataFrame
        Dataframe with the information on CIK filing, filing date, refernece date, and whether it's amendemnt or not
    ceiling : float, optional
        Maximum value allowed for a stock, since some bonds are wrongly parsed as stocks in the dataset, by default 1e7
    lag : int, optional
        Days after a reference date at which its filing deadline passes. The quarter's unit is read from everything public by then, by default 45
    """

    keys  = ['cik','ref_date','vintage_date','amendment']
    peers = ['cusip','ref_date']

    # A peer group never crosses ref_date, so a quarter is the largest thing that ever needs to be in memory. The links
    # are cut by filing window instead, so each quarter is gathered from the few of them that happen to carry it
    floor = pd.Timestamp(START_REF) - pd.offsets.QuarterEnd(4)
    where = {}
    for p in files:
        for q in pd.read_parquet(p, columns=['ref_date'])['ref_date'].unique():
            if q >= floor: 
                where.setdefault(q, []).append(p)

    rate, modal, unexp, wild = [], [], 0, 0
    for i, q in enumerate(sorted(where), 1):
        df = pd.concat([pd.read_parquet(p, filters=[('ref_date','==',q)]) for p in where[q]], ignore_index=True)
        df = df.sort_values(peers + ['vintage_date'], ignore_index=True)

        df['px'] = df['value'].where(df['value'].gt(0)) / df['sshprnamt'].where(df['sshprnamt'].gt(0))

        # Compare the filers who have the same security as the respective row, but taking into consideration data available up to the vintage
        g = df.groupby(peers, sort=False, observed=True)
        df['_p'] = g['px'].expanding().median().to_numpy()
        df['_n'] = g.cumcount().to_numpy() + 1

        v    = df.groupby(peers + ['vintage_date'], sort=False, observed=True)
        peer = v['_p'].transform('last').where(v['_n'].transform('max').ge(9))

        # Test against peers and if off by a factor of 3, correct unit; otherwise unexplained and drop
        lg     = np.log10(df['px'] / peer).replace([np.inf, -np.inf], np.nan)
        mag    = lg.abs()
        peered = lg.notna()
        unit   = peered & mag.round().isin([3,6]) & mag.sub(mag.round()).abs().lt(0.05)
        normal = peered & mag.lt(0.5)

        df['expo'] = lg.round().where(unit, 0).astype('int8')
        df['zone'] = np.select([~peered, unit, normal], ['unpeered','unit','normal'], 'unexplained')

        # For each row of df, we start to add the checkers: if it's peered, has correct unit, and if it's non-explained
        z = df[keys].assign(peered=peered.to_numpy(), unit=unit.to_numpy(), unexp=df['zone']=='unexplained')
        rate.append(z.groupby('cik', as_index=False).agg(peered=('peered','sum'), unit=('unit','sum'), unexp=('unexp','sum')))

        # Getting the correct exponent only for peered rows (can't compare unpeered rows)
        m = z.loc[peered.to_numpy(), keys].assign(expo=df.loc[peered.to_numpy(),'expo']).groupby(keys + ['expo'], as_index=False).size()
        m = m.sort_values('size').drop_duplicates(keys, keep='last').rename(columns={'expo':'modal'})
        modal.append(m)

        # Now we remove unexplained rows, since we can't correct their units. They are probably a filing mistake
        unexp += int(z['unexp'].sum())
        df     = df.loc[df['zone'] != 'unexplained']

        # Corrects the units. If unpeered
        e = np.where(df['zone'] == 'unpeered',
                     df[keys].merge(m[keys + ['modal']], on=keys, how='left')['modal'].fillna(0).to_numpy(),
                     df['expo'].to_numpy())
        df['value'] = df['value'] / 10.0**e

        # The peer step only made the quarter agree with itself; the level it agreed on is read from everything public by the
        # filing deadline, which is also the first date any analysis reads the quarter, so nothing here precedes its own use.
        # A quarter reached only by later amendments has an empty deadline window and falls back to whatever it does carry
        px  = df['value'].where(df['value'].gt(0)) / df['sshprnamt'].where(df['sshprnamt'].gt(0))
        due = px[df['vintage_date'].le(pd.Timestamp(q) + pd.Timedelta(days=lag))]

        if np.nanmedian(due if due.notna().any() else px) < 1:
            solo = df['zone'].eq('unpeered') & df.assign(_x=px).groupby(keys)['_x'].transform('median').ge(1)
            df.loc[~solo, 'value'] *= 1000

        # Some bonds are reported as shares, so we remove 'shares' worth more than 10 million USD, as they are likely bonds
        df['px'] = df['value'].where(df['value'].gt(0)) / df['sshprnamt'].where(df['sshprnamt'].gt(0))
        w        = df['px'].gt(ceiling)
        wild    += int(w.sum())

        df.loc[~w].drop(['px','_p','_n','expo','zone'], axis=1).to_parquet(DATA_INT / f"13f_units_{pd.Timestamp(q).date()}.parquet", index=False)
        print(f"    {i}/{len(where)}: {pd.Timestamp(q).date()} | {len(df):,} rows".ljust(130), end=_END, flush=True)
    print()

    rate  = pd.concat(rate, ignore_index=True).groupby('cik', as_index=False).sum()
    modal = pd.concat(modal, ignore_index=True)
    judged = rate.loc[rate['peered'].gt(0)]
    up     = modal.loc[modal['modal'].eq(-3)]
    down   = modal.loc[modal['modal'].eq(3)]

    print(f"\n  Filers: {len(rate):,} | judged on units: {len(judged):,} | never judgeable (no security with 9+ filers): {len(rate) - len(judged):,}")
    print(f"  Rows repaired by a clean power of 1000: {judged['unit'].sum():,}")
    print(f"  Rows dropped as unexplained           : {judged['unexp'].sum():,} ({judged['unexp'].sum()/judged['peered'].sum()*100:.2f}% of peered rows)")
    print(f"  Filings dominated by thousands (x1000): {len(up):,} / {up['cik'].nunique():,} filers | {up['vintage_date'].ge('2023-01-03').mean()*100:.1f}% from 2023 on")
    print(f"  Filings dominated by units (/1000)    : {len(down):,} / {down['cik'].nunique():,} filers | {down['vintage_date'].lt('2023-01-03').mean()*100:.1f}% before 2023")

    # Since we remove unexplained fillings, we make a diagnostic of who's jumping out
    noisy = judged.assign(share=judged['unexp'] / judged['peered']).nlargest(5, 'share')
    print(f"\n  Noisiest filers by unexplained share (kept, but they lose those rows):")
    for r in noisy.merge(names.drop_duplicates('cik')[['cik','filingmanager_name']], on='cik', how='left').itertuples():
        print(f"      {r.cik}  {str(r.filingmanager_name)[:44]:<44} {r.share*100:>5.1f}% of {int(r.peered):>7,} peered rows")

    print(f"\n  {unexp:,} unexplained rows removed from the sample")
    print(f"  {int(wild):,} rows dropped above ${ceiling:,.0f} per share (debt reported as equity, invisible to the peer check)")

def get_13f_data():
    """
    Creates the treated 13F data, still with the full universe of firms (CIKs) and assets (CUSIPs).

    It uses function clean_units to solve unit issues among different filers, using the median implicit price as a measure o
    """

    # Only the zips missing from disk are fetched
    paths = download_13f()

    # Parsing each .zip and keeping only the three tables we need
    filing_list = []
    issuer_list = []
    for i, path in enumerate(paths, 1):
        filing, issuer, rows = importer_13f(path)
        filing_list.append(filing)
        issuer_list.append(issuer)

        print(f"    {i}/{len(paths)}: {path.name} | {rows:,} rows".ljust(130), end=_END, flush=True)
    print()

    names = pd.concat(filing_list, ignore_index=True)

    issuer = pd.concat(issuer_list, ignore_index=True).sort_values('ref_date').drop_duplicates('cusip', keep='last').drop('ref_date', axis=1)
    issuer.to_parquet(DATA_INT / 'cusip_issuer.parquet', index=False)
    print(f"Total of {len(issuer.index)} CUSIP/assets found.")

    key = ['cik','ref_date']

    # Cleared first: clean_units repartitions by ref_date, so a quarter left by an earlier run would double the universe
    for p in DATA_INT.glob('13f_units_*.parquet'): p.unlink()
    for p in DATA_INT.glob('13f_holdings_*.parquet'): p.unlink()

    # Repairing positions which are in the wrong unit using the median implicit price
    clean_units(files = sorted(DATA_RAW.glob('13f_raw_*.parquet')), names = names)

    # NEW HOLDINGS are summed to the base filing, and filings sharing a vintage are assigned to the amendment. Every
    # vintage of a (cik, ref_date) now sits in the same file, so a group is resolved without ever leaving it. It reads
    # the units files and writes the holdings ones, so no step ever rewrites its own input and a rerun is harmless
    amended = 0
    for p in sorted(DATA_INT.glob('13f_units_*.parquet')):
        out = []
        for _, g in pd.read_parquet(p).groupby(key, sort=False):
            state = None
            for v in sorted(g['vintage_date'].unique()):
                f = g.loc[g['vintage_date'].eq(v)]
                a = set(f['amendment'])

                # A complete restatement of the table beats anything else filed the same day; NEW HOLDINGS only adds to it
                if 'RESTATEMENT' in a:
                    f = f.loc[f['amendment'].eq('RESTATEMENT')]
                elif 'UNSPECIFIED' in a:
                    f = f.loc[f['amendment'].eq('UNSPECIFIED')]
                elif 'NEW HOLDINGS' in a:
                    base = f.loc[f['amendment'].eq('ORIGINAL')] if 'ORIGINAL' in a else state
                    f    = f.loc[f['amendment'].eq('NEW HOLDINGS')]
                    if base is not None:
                        f = pd.concat([base, f]).groupby(key + ['cusip'], as_index=False).agg(value     = ('value','sum'),
                                                                                              sshprnamt = ('sshprnamt','sum'))
                        f['vintage_date'] = v
                        f['amendment']    = 'NEW HOLDINGS'
                        amended += 1
                state = f
                out.append(f)

        pd.concat(out, ignore_index=True).to_parquet(DATA_INT / p.name.replace('13f_units_', '13f_holdings_'), index=False)

    print(f"Resolved {amended:,} amended filings")

    files = sorted(DATA_INT.glob('13f_holdings_*.parquet'))

    # Total is the full position of each CIK is each ref and vintage
    total = pd.concat([pd.read_parquet(p, 
                                       columns=key+['vintage_date','amendment','cusip','value']).groupby(key+['vintage_date','amendment'], 
                                                                                                         as_index=False).agg(total_value = ('value','sum'),
                                                                                                                             n_positions = ('cusip','nunique')) for p in files], ignore_index=True)
    total = total.merge(names, on=key+['vintage_date','amendment'], how='left')
    total.to_parquet(DATA_INT / '13f_total.parquet', index=False)

    # Row counts come from the parquet footers, so nothing is read back into memory
    lines = sum(pq.ParquetFile(p).metadata.num_rows for p in files)

    print(f"13F dataset done!")
    print(f"Total universe: {lines:,} rows | {total['cik'].nunique():,} CIKs | {total['ref_date'].nunique():,} reference dates | {len(total):,} filings")

def get_spx_composition():
    """
    Creates a historic of S&P500 composition over time and use Fails-to-Deliver data to match it with CUSIP.

    Returns
    -------
    cusip : pd.DataFrame
        Dataframe with, for each vintage date (business day), the composition of the S&P 500 and the CUSIPs of the stocks in the index. Tickers are column names
    """

    # Getting historical S&P composition from a GitHub repository
    print('\nStarting to fetch historical S&P composition...\n')
    url = ("https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
 
    snp = pd.read_csv(url, parse_dates=["date"])
    snp["tickers"] = snp["tickers"].str.split(",")

    snp = snp.explode("tickers").rename(columns={"date": "Date", "tickers": "Ticker"})

    snp["Ticker"] = snp["Ticker"].str.strip().str.replace('.','',regex=False)

    snp = snp.loc[snp['Date'] >= START_REF].reset_index(drop=True)
    snp = pd.crosstab(snp['Date'], snp['Ticker']).clip(upper=1)
    snp = snp.resample('B').ffill()

    print(f"\nAverage monthly S&P500 turnover: {round(snp.diff().abs().sum(axis=1).resample('ME').sum().mean()/2,2)}")
    print(f"Average S&P500 count:{round(snp.sum(axis=1).mean(),2)} | Min S&P500 count:{snp.sum(axis=1).min()} | Max S&P500 count:{snp.sum(axis=1).max()}\n")

    print('Finished fetching S&P data!')

    # Getting fail-to-deliver to make ticket to CUSIP mapping per 15-day period (only since 2013)
    print('Starting to create CUSIP/ticker mapping...')

    url   = 'https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data'
    links = sec_get_links(url=url)

    # Downloading each .zip and keeping only the three tables we need
    df_list = []
    for link in links:
        folder = DATA_INT

        h   = {"User-Agent": "Luis Felipe Lins lf.valadares@hotmail.com"}
        z = requests.get(link, headers=h)
        z.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            filename = zf.namelist()[0]
            with zf.open(filename) as f: df = pd.read_csv(f, sep="|", dtype=str, encoding="latin-1", on_bad_lines='skip')[['SETTLEMENT DATE','CUSIP','SYMBOL','DESCRIPTION']]
        df["SETTLEMENT DATE"] = pd.to_datetime(df["SETTLEMENT DATE"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns={'SETTLEMENT DATE':'ref_date'})
        df = df.dropna(subset='ref_date')
        df_list.append(df)

        print(f"    {link}: done | {len(df):,} rows".ljust(130), end=_END, flush=True)

        if df['ref_date'].min() < pd.Timestamp(START_REF): break

    print()
    cusip_ticker_dict = pd.concat(df_list)
    cusip_ticker_dict = cusip_ticker_dict.loc[cusip_ticker_dict['ref_date'] >= START_REF].reset_index(drop=True)
    cusip_ticker_dict = cusip_ticker_dict.loc[cusip_ticker_dict['SYMBOL'].isin(snp.columns)]

    print(f"Total S&P500 tickers: {len(snp.columns)} | Respective CUSIP codes: {len(cusip_ticker_dict['SYMBOL'].unique())}")

    cusip = cusip_ticker_dict.groupby(['ref_date','SYMBOL'])['CUSIP'].last().unstack().bfill(limit=16)
    cusip = cusip.reindex(cusip.index.union(snp.index)).ffill().reindex(index=snp.index, columns=snp.columns).where(snp.eq(1))

    snp_cusip = (cusip.notna()*1).sum(axis=1)

    print(f"\nAverage CUSIP coverage of S&P500 over time: {round((snp_cusip/snp.sum(axis=1)).mean()*100,1)}% | Min. coverage: {round((snp_cusip/snp.sum(axis=1)).min()*100,1)}%")

    tickers = cusip.columns.tolist()

    print("Downloading prices from Yahoo Finance")
    prices = yf.download(tickers=tickers, start=START_REF, end=cusip.index.max(), auto_adjust=False, progress=True, threads=True)

    close = prices["Close"]
    close = close.reindex(index=cusip.index, columns=cusip.columns).where(cusip.notna())

    index = get_spx_index(calendar=cusip.index)

    return cusip, close, index

def get_spx_index(calendar):
    """
    Level of the S&P 500 itself, to serve as the benchmark the strategies are read against.

    The price index is taken rather than the total return one, since the security prices this project works with
    carry no dividends either and only the two together compare like with like.

    Parameters
    ----------
    calendar : pandas.DatetimeIndex
        Business days the composition runs on, so the benchmark lands on the same calendar as everything else.

    Returns
    -------
    index : pandas.DataFrame
        One row per business day, carrying the close of the index.
    """

    print("\nDownloading the S&P 500 index level")
    px = yf.download(tickers='^GSPC', start=START_REF, end=calendar.max(), auto_adjust=False, progress=False)

    # A single ticker comes back as a Series on some yfinance versions and as a one column frame on others
    close = px['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]

    index = close.reindex(calendar).rename('spx').to_frame()

    r = index['spx'].pct_change(fill_method=None).dropna()
    print(f"  {index['spx'].notna().sum():,} days | {index.index.min().date()} to {index.index.max().date()}")
    print(f"  {((1 + r).prod() ** (252/len(r)) - 1)*100:.2f}% a year | vol {r.std()*np.sqrt(252)*100:.2f}%")

    return index

def filer_stats(aum_top, top, lag = 45):
    """
    Calculates some statistics for the pool of CIKs (filers), only among top AUM filers. 
    Used to find the universe of CIKs we'll consider in the analysis.

    Parameters
    ----------
    aum_top : float
        Share of AUM to be considered when calculating the statistics, to avoid taking small firms with extreme statistics values.
    top : int
        Number of largest positions of a filing whose weights are summed into top_weight. Does not affect hhi or eff_n, which are computed over the whole book.
    lag : int, optional
        Days after a reference date at which its filing deadline is taken to have passed. Sets the window of the AUM ranking, which is the previous quarter as reported by the current quarter's deadline. 45 is the statutory 13F deadline. By default, 45

    Returns
    -------
    stat : pd.DataFrame
        Dataframe with the statistics for all CIKs in each point of time (vintage)
    """

    keys = ['cik','ref_date','vintage_date']
    stat = pd.read_parquet(DATA_INT / '13f_total.parquet', columns=keys+['amendment','n_positions','total_value']).sort_values(['cik','vintage_date'])

    stat['filing_lag'] = (stat['vintage_date'] - stat['ref_date']).dt.days

    # Size ranking is made against the prev. Q, which should be fully reported by the time this one is due. 
    # Ranking against curr. Q instead would rank an early filer against a near-empty cross-section
    rank = []
    for q in sorted(stat['ref_date'].unique()):
        x = stat.loc[stat['ref_date'].eq(pd.Timestamp(q) - pd.offsets.QuarterEnd(1)) & stat['vintage_date'].le(pd.Timestamp(q) + pd.Timedelta(days=lag))]
        if x.empty: continue

        x = x.sort_values('vintage_date').drop_duplicates('cik', keep='last')
        rank.append(pd.DataFrame({'cik': x['cik'].to_numpy(), 'ref_date': q, 'aum_pct': x['total_value'].rank(pct=True, ascending=False).to_numpy()}))

    # The rank is a property of (cik, ref_date), so every vintage of the same filing carries the same one
    stat = stat.merge(pd.concat(rank, ignore_index=True), on=['cik','ref_date'], how='left').sort_values(['cik','vintage_date'])

    # Creating a number of positions variance metric
    gb = stat.groupby('cik')
    n  = gb.cumcount() + 1

    stat['n_filings']  = n
    stat['amend_rate'] = gb['amendment'].transform(lambda s: s.ne('ORIGINAL').cumsum()) / n

    mean = gb['n_positions'].cumsum() / n
    var  = (stat.assign(sq=stat['n_positions']**2).groupby('cik')['sq'].cumsum() / n - mean**2).clip(lower=0)

    stat['n_positions_cv'] = np.sqrt(var) / mean.where(mean.gt(0))

    # Filtering since our REF_DATE
    before = len(stat)
    stat   = stat.loc[stat['ref_date'].ge(START_REF)]
    if aum_top is not None: stat = stat.loc[stat['aum_pct'].le(aum_top)]
    wanted = pd.MultiIndex.from_frame(stat[keys])

    # Concentration is a function of a single filing, so each link is finished before the next one is opened
    parts = []
    for p in sorted(DATA_INT.glob('13f_holdings_*.parquet')):
        g = pd.read_parquet(p, columns=keys+['cusip','value'])
        g = g.loc[g['value'].gt(0) & pd.MultiIndex.from_frame(g[keys]).isin(wanted)].sort_values(keys+['value'], ascending=[True]*len(keys)+[False])
        if g.empty: continue

        g['w'] = g['value'] / g.groupby(keys)['value'].transform('sum')
        g['r'] = g.groupby(keys).cumcount()

        parts.append(g.assign(sq = g['w']**2, hd = g['w'].where(g['r'].lt(top), 0)).groupby(keys, as_index=False).agg(top_weight = ('hd','sum'), hhi = ('sq','sum'), n_priced = ('cusip','size')))

    conc = pd.concat(parts, ignore_index=True)
    conc['eff_n'] = 1 / conc['hhi'].where(conc['hhi'].gt(0))

    stat = stat.merge(conc.drop('hhi', axis=1), on=keys, how='left').sort_values(['cik','vintage_date'])
    stat = stat.drop('amendment', axis=1)
    stat.to_parquet(DATA_INT / '13f_filer_stats.parquet', index=False)

    m = stat['n_filings'].ge(8)
    print(f"Filer stats written | {len(stat):,} of {before:,} filings kept (top {aum_top:.0%} by AUM) | {stat['cik'].nunique():,} CIKs | {stat['top_weight'].isna().sum():,} without holdings")
    print(f"\n  Median across filings with 8+ filings of history ({m.sum():,}):")
    print(f"    top {top} weight   {stat.loc[m,'top_weight'].median()*100:>6.1f}%   | effective N {stat.loc[m,'eff_n'].median():>7.1f} of {stat.loc[m,'n_positions'].median():,.0f} positions")
    print(f"    filing lag     {stat.loc[m,'filing_lag'].median():>6.0f} d   | amend rate  {stat.loc[m,'amend_rate'].median()*100:>6.1f}%")
    print(f"    n_positions cv {stat.loc[m,'n_positions_cv'].median():>6.2f}")
    print(f"\n  Filing lag deciles: {[int(x) for x in stat['filing_lag'].quantile(np.arange(.1,1,.1)).tolist()]}")
    print(f"  Filings arriving after the 45-day deadline: {stat['filing_lag'].gt(45).mean()*100:.1f}%")

    return stat

def selecting_firms(rule, pool, target, top, lag):
    """
    Given each of the four rules considered, uses filer_stats to get the universe of CIKs under consideration.

    Parameters
    ----------
    rule : str
        One of 'aum', 'cv_high', 'cv_low', 'conc'. Chooses which statistic orders the filers to be selected.
    pool : float
        Among all filers, only the top (pool)% of the firms by AUM are considered for the pool of statistics, to avoid selecting small firms with extreme stats
    target : float
        Share of all filers the universe should end up with, e.g. 5% means we want 5% of the filers based on the stat under consideration. Cannot exceed pool.
    top : int
        Passed straight to filer_stats.
    lag : int
        Days after a reference date at which its filing deadline is taken to have passed. Sets the decision dates each quarter is responsible for.

    Returns
    -------
    univ : pd.DataFrame
        One row per (vintage_date, ref_date, cik) in the universe, carrying the rule that selected it.

    Raises
    ------
    ValueError
        If target exceeds pool, since the universe cannot be wider than the pool the statistics were cut to.
    """

    # The AUM rule reaches it directly, the other three by taking that share of the pool the stats were already cut to
    rules = {'aum'     : None,
             'cv_high' : ('n_positions_cv', False),
             'cv_low'  : ('n_positions_cv', True),
             'conc'    : ('eff_n',          False)}

    if target > pool: raise ValueError(f"target {target} cannot exceed the pool {pool} the stats are cut to")

    cut  = target / pool
    stat = filer_stats(aum_top=pool, top=top, lag=lag)[['cik','ref_date','vintage_date','aum_pct','n_positions_cv','eff_n']]

    # Creating a ref date, deadline for 13F filing, and vintage date calendar
    quarters = stat['ref_date'].drop_duplicates().sort_values()
    calendar = pd.DataFrame({'deadline': quarters + pd.Timedelta(days=lag), 'ref_date': quarters}).sort_values('deadline')

    grid = pd.DataFrame({'vintage_date': stat['vintage_date'].drop_duplicates().sort_values()})
    grid = pd.merge_asof(grid, calendar, left_on='vintage_date', right_on='deadline', direction='backward').dropna(subset=['ref_date'])

    # For each vintage date in grid, we now create the dataframe with the set of CIKs which comply with the rule in that vintage
    out = []
    for q, w in grid.groupby('ref_date', sort=True):
        f = stat.loc[stat['ref_date'].eq(q)].sort_values('vintage_date')
        for d in w['vintage_date']:
            # The split is recomputed on whoever is visible on d, so the universe fills in as the quarter is reported
            v = f.loc[f['vintage_date'].le(d)].drop_duplicates('cik', keep='last')
            if v.empty: continue

            if rule == 'aum': v = v.loc[v['aum_pct'].le(target)]
            else:
                col, asc = rules[rule]
                v = v.loc[v[col].rank(pct=True, ascending=asc).le(cut)]

            out.append(pd.DataFrame({'vintage_date': d, 'ref_date': q, 'cik': v['cik'].to_numpy(), 'rule': rule}))

    univ = pd.concat(out, ignore_index=True)
    univ.to_parquet(DATA_INT / f'13f_cik_universe_{rule}.parquet', index=False)

    n = univ.groupby('vintage_date')['cik'].size()
    print(f"\nUniverse '{rule}' written | {len(univ):,} rows | {univ['cik'].nunique():,} distinct CIKs")
    print(f"  targeting {target:.0%} of filers: {'aum_pct <= '+format(target,'.3f') if rule == 'aum' else f'top {cut:.0%} of the {pool:.0%} pool'}")
    print(f"  CIKs per filing date: median {n.median():,.0f} | min {n.min():,} | max {n.max():,}")

    return univ

def final_data_creator(univ):
    """
    Helper to create the fulldataframe given that we have a universe of CIK and CUSIPs

    Parameters
    ----------
    univ : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # The universe is a decision-date mask and the holdings are the fact table, so only the CIK list crosses over here.
    # Materialising the mask itself would replicate every filing once per date on which it was still the latest one
    ciks  = set(univ['cik'].unique())
    parts = []
    for p in sorted(DATA_INT.glob('13f_holdings_*.parquet')):
        g = pd.read_parquet(p)
        g = g.loc[g['cik'].isin(ciks) & g['ref_date'].ge(START_REF)]
        if not g.empty: parts.append(g)

    hold = pd.concat(parts, ignore_index=True).sort_values(['cik','ref_date','vintage_date','cusip'])
    hold.to_parquet(DATA_FINAL / f"13f_holdings_universe_{univ['rule'].iloc[0]}.parquet", index=False)

    print(f"\nHoldings written | {len(hold):,} rows | {hold['cik'].nunique():,} CIKs | {hold['cusip'].nunique():,} CUSIPs | {hold['ref_date'].nunique()} quarters")
    print(f"  from {hold['ref_date'].min().date()} to {hold['ref_date'].max().date()} | {hold.memory_usage(deep=True).sum()/1e9:.2f} GB in memory")

    return hold

def create_final_dataset(rule, pool, target, top, lag):
    """
    Selects the filer universe a rule asks for and cuts the treated holdings down to it.

    The treated 13F base is expected to be on disk already. Every rule reads the same files and only the selection
    step changes between them, so building the base is the caller's call to make once and not this function's to
    repeat four times.

    Parameters
    ----------
    rule : str
        One of 'aum', 'cv_high', 'cv_low', 'conc'. Passed straight to selecting_firms.
    pool : float
        Share of filers by AUM the statistics are computed on.
    target : float
        Share of all filers the universe should end up with.
    top : int
        Number of largest positions of a filing summed into top_weight.
    lag : int
        Days after a reference date at which its filing deadline is taken to have passed.

    Returns
    -------
    fin_df : pd.DataFrame
        Holdings of the filers the rule selected, one row per (cik, ref_date, vintage_date, cusip).
    """

    fin_df = final_data_creator(selecting_firms(rule=rule, pool=pool, target=target, top=top, lag=lag))

    return fin_df

def analyzing_total(floor=START_REF, lag=45, pcts=(1,5,10)):
    """
    Creates and saves a chart with the total AUM amount and share of total holdings for the top 1, 5 and 10% of the holders in each time period.

    not used in any of the dataset creator steps, only for analysis.

    Parameters
    ----------
    floor : str, optional
        String with the date to start considering the chart, by default START_REF
    lag : int, optional
        The function is discontinuous around quarter changes, i.e. it only shows the latest quarter (remember previous quarters can be amended).
        Hence, lag identify how many days after the quarter's over that the quarter becomes the current quarter. By default 45
    pcts : tuple, optional
        The groups of percentiles to be considered in the chart, by default (1,5,10)

    """

    total = pd.read_parquet(DATA_INT / '13f_total.parquet')
    total = total.loc[total['ref_date'].ge(floor), ['cik','ref_date','vintage_date','total_value']]

    # A quarter becomes the prevailing one once its filing deadline has passed, and holds until the next deadline
    quarters = total['ref_date'].drop_duplicates().sort_values()
    calendar = pd.DataFrame({'deadline': quarters + pd.Timedelta(days=lag), 'ref_date': quarters}).sort_values('deadline')

    grid = pd.DataFrame({'vintage_date': total['vintage_date'].drop_duplicates().sort_values()})
    grid = pd.merge_asof(grid, calendar, left_on='vintage_date', right_on='deadline', direction='backward').dropna(subset=['ref_date'])

    rows = []
    for q, w in grid.groupby('ref_date', sort=True):
        f = total.loc[total['ref_date'].eq(q)].sort_values('vintage_date')
        for d in w['vintage_date']:
            # Latest vintage of each filer as known on d, so late arrivals and amendments both move the cross-section
            v = f.loc[f['vintage_date'].le(d)].drop_duplicates('cik', keep='last')
            if v.empty: continue

            c   = np.sort(v['total_value'].to_numpy())[::-1].cumsum()
            n   = len(c)
            rec = {'vintage_date': d, 'ref_date': q, 'n_ciks': n, 'total_value': c[-1]}
            for p in pcts: rec[f'top{p}'] = c[max(1, int(np.ceil(p/100*n))) - 1]
            rows.append(rec)

    conc = pd.DataFrame(rows)
    conc.to_parquet(DATA_FINAL / 'concentration.parquet', index=False)

    shade = dict(zip(sorted(pcts, reverse=True), ['#86b6ef','#2a78d6','#104281']))
    ink   = {'primary':'#0b0b0b', 'secondary':'#52514e', 'muted':'#898781', 'grid':'#e1e0d9', 'axis':'#c3c2b7'}

    fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True, facecolor='#fcfcfb')
    fig.suptitle('Concentration of 13F assets among the largest filers', x=0.125, y=0.98, ha='left', fontsize=14, color=ink['primary'])

    # Title, subtitle and legend each get their own band above the axes, so none of the three lands on another
    fig.text(0.125, 0.94, 'Point-in-time: at each filing date, the prevailing quarter is the last one past its 45-day deadline',
             ha='left', fontsize=10, color=ink['secondary'])

    x = conc['vintage_date']
    for p in sorted(pcts, reverse=True):
        ax[0].plot(x, conc[f'top{p}']/1e12, lw=2, color=shade[p], label=f'Top {p}%')
        ax[1].plot(x, conc[f'top{p}']/conc['total_value']*100, lw=2, color=shade[p], label=f'Top {p}%')

    ax[0].set_ylabel('USD trillions', fontsize=10, color=ink['secondary'])
    ax[1].set_ylabel('% of all 13F assets', fontsize=10, color=ink['secondary'])
    ax[1].set_ylim(0, 100)

    for a in ax:
        a.set_facecolor('#fcfcfb')
        a.grid(axis='y', color=ink['grid'], lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=ink['muted'], labelsize=9)
        for s in ('top','right'): a.spines[s].set_visible(False)
        for s in ('bottom','left'): a.spines[s].set_color(ink['axis'])

        # Direct labels at the right edge carry identity without a colour-only lookup
        ends = sorted(((conc[f'top{p}']/1e12 if a is ax[0] else conc[f'top{p}']/conc['total_value']*100).iloc[-1], p) for p in pcts)
        gap  = 0.05 * (a.get_ylim()[1] - a.get_ylim()[0])

        # Walking bottom-up, a label closer than 5% of the axis to the one below it is pushed up to that distance,
        # so the top bands stay readable where the three lines converge instead of printing on top of each other
        for i, (y, p) in enumerate(ends):
            if i and y - ends[i-1][0] < gap: y = ends[i-1][0] + gap
            ends[i] = (y, p)
            a.annotate(f'Top {p}%', (x.iloc[-1], y), xytext=(6,0), textcoords='offset points',
                       va='center', fontsize=9, color=ink['secondary'])

    # Legend sits above the axes so it never lands on the 2013 spike
    ax[0].legend(frameon=False, fontsize=9, labelcolor=ink['secondary'], ncol=len(pcts),
                 loc='lower left', bbox_to_anchor=(0, 1.02))
    fig.subplots_adjust(top=0.88, right=0.9)
    fig.savefig(DATA_FINAL / 'concentration.png', dpi=150, bbox_inches='tight', facecolor='#fcfcfb')

    last = conc.iloc[-1]
    print(f"Concentration written | {len(conc):,} filing dates | {conc['ref_date'].nunique()} prevailing quarters")
    print(f"  As of {last['vintage_date'].date()} ({last['n_ciks']:,} filers, {last['total_value']/1e12:,.1f}tn):")
    for p in sorted(pcts, reverse=True):
        print(f"    top {p:>2}% ({max(1,int(np.ceil(p/100*last['n_ciks']))):>4,} filers) holds {last[f'top{p}']/1e12:>6,.1f}tn | {last[f'top{p}']/last['total_value']*100:>5.1f}%")
