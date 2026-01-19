import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MONTH_MAP = {
    'januari': 1, 'februari': 2, 'maret': 3, 'april': 4, 'mei': 5, 'juni': 6,
    'juli': 7, 'agustus': 8, 'september': 9, 'oktober': 10, 'november': 11, 'desember': 12,
}


def parse_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        pass
    parts = s.lower().split()
    if len(parts) == 2 and parts[0] in MONTH_MAP:
        m = MONTH_MAP[parts[0]]
        y = int(parts[1])
        return date(y, m, 1)
    return None


def load_prices(path, sheet_name):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header = raw.loc[1].tolist()

    cols = []
    for idx, val in enumerate(header):
        if idx == 0:
            cols.append('Komoditas')
        elif idx == 1:
            cols.append('Deskripsi')
        else:
            cols.append(parse_date(val))

    rows = raw.loc[2:14].copy()
    rows.columns = cols
    rows = rows.dropna(axis=1, how='all')

    date_cols = [c for c in rows.columns if isinstance(c, date)]
    for c in date_cols:
        rows[c] = pd.to_numeric(rows[c], errors='coerce')

    rows = rows.set_index('Komoditas')
    return rows[date_cols]


def compute_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna(how='any')


def annualize_mean_cov(returns_df, periods_per_year=12):
    mu = returns_df.mean() * periods_per_year
    cov = returns_df.cov() * periods_per_year
    return mu.values, cov.values, mu


def optimize_target_return(mu, cov, target_return, bounds):
    n = len(mu)

    def port_var(w):
        return w.T @ cov @ w

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w @ mu - target_return},
    )
    w0 = np.ones(n) / n
    res = minimize(port_var, w0, bounds=bounds, constraints=cons)
    return res


def optimize_target_risk(mu, cov, target_risk, bounds):
    n = len(mu)

    def port_risk(w):
        return np.sqrt(w.T @ cov @ w)

    def objective(w):
        return (port_risk(w) - target_risk) ** 2

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )
    w0 = np.ones(n) / n
    res = minimize(objective, w0, bounds=bounds, constraints=cons)
    return res


def main():
    parser = argparse.ArgumentParser(description='Mean-variance portfolio demo.')
    parser.add_argument('--path', default='Simulasi Portofolio Tambang.xlsx')
    parser.add_argument('--sheet', default='HBA HMA')
    parser.add_argument('--assets', default='Emas,Perak,Besi,Tembaga')
    parser.add_argument('--target-return', type=float, default=None)
    parser.add_argument('--target-risk', type=float, default=0.12)
    parser.add_argument('--min-weight', type=float, default=0.05)
    parser.add_argument('--max-weight', type=float, default=0.50)
    args = parser.parse_args()

    prices = load_prices(args.path, args.sheet)
    assets = [a.strip() for a in args.assets.split(',') if a.strip()]
    prices = prices.loc[assets].T

    returns = compute_log_returns(prices)
    mu, cov, mu_series = annualize_mean_cov(returns)

    print('Date range:', returns.index.min(), 'to', returns.index.max())
    print('Annualized mean returns:')
    for name, val in mu_series.items():
        print(f'  {name}: {val:.2%}')

    if args.target_return is None:
        target_return = float(mu.mean())
    else:
        target_return = args.target_return

    bounds = [(args.min_weight, args.max_weight)] * len(assets)

    res_ret = optimize_target_return(mu, cov, target_return, bounds)
    if not res_ret.success:
        print('Target return optimization failed:', res_ret.message)
    w_ret = res_ret.x

    exp_ret = w_ret @ mu
    risk_ret = np.sqrt(w_ret.T @ cov @ w_ret)

    print('\nTarget return optimization')
    print('Target return:', f'{target_return:.2%}')
    for name, w in zip(assets, w_ret):
        print(f'  {name}: {w:.2%}')
    print(f'  Expected return: {exp_ret:.2%}')
    print(f'  Risk: {risk_ret:.2%}')

    res_risk = optimize_target_risk(mu, cov, args.target_risk, bounds)
    if not res_risk.success:
        print('Target risk optimization failed:', res_risk.message)
    w_risk = res_risk.x

    exp_ret2 = w_risk @ mu
    risk2 = np.sqrt(w_risk.T @ cov @ w_risk)

    print('\nTarget risk optimization')
    print('Target risk:', f'{args.target_risk:.2%}')
    for name, w in zip(assets, w_risk):
        print(f'  {name}: {w:.2%}')
    print(f'  Expected return: {exp_ret2:.2%}')
    print(f'  Risk: {risk2:.2%}')


if __name__ == '__main__':
    main()
