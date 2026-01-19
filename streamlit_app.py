import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date
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


@st.cache_data
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
    return minimize(port_var, w0, bounds=bounds, constraints=cons)


def optimize_target_vol(mu, cov, target_vol, bounds):
    n = len(mu)

    def port_vol(w):
        return np.sqrt(w.T @ cov @ w)

    def objective(w):
        return (port_vol(w) - target_vol) ** 2

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    w0 = np.ones(n) / n
    return minimize(objective, w0, bounds=bounds, constraints=cons)


def fmt_pct(value):
    pct = abs(value) * 100
    if pct < 10:
        return f'{value:.2%}'
    if pct < 100:
        return f'{value:.1%}'
    return f'{value:.0%}'




st.set_page_config(page_title='Portfolio Optimizer', layout='wide')

st.title('Portfolio Optimizer')

with st.sidebar:
    st.header('Inputs')
    path = st.text_input('Excel path', 'Simulasi Portofolio Tambang.xlsx')
    sheet = st.text_input('Sheet name', 'HBA HMA')

try:
    prices = load_prices(path, sheet)
except Exception as exc:
    st.error(f'Failed to load data: {exc}')
    st.stop()

all_assets = list(prices.index)

with st.sidebar:
    default_assets = [a for a in ['Emas', 'Perak', 'Besi', 'Tembaga'] if a in all_assets]
    if not default_assets:
        default_assets = all_assets[:4]
    assets = st.multiselect('Assets', all_assets, default=default_assets)

if len(assets) < 2:
    st.warning('Select at least 2 assets.')
    st.stop()

price_df = prices.loc[assets].T
returns = compute_log_returns(price_df)

mu, cov, mu_series = annualize_mean_cov(returns)

with st.sidebar:
    min_weight_pct = st.number_input('Min weight (%)', value=0.0, step=0.50, format='%.2f')
    max_weight_pct = st.number_input('Max weight (%)', value=100.0, step=0.50, format='%.2f')
    if min_weight_pct >= max_weight_pct:
        st.error('Min weight must be less than max weight.')
        st.stop()

    mode = st.selectbox('Optimization mode', ['Target Return', 'Target Volatility'])
    default_target_pct = float(mu.mean()) * 100
    target_return_pct = st.number_input(
        'Target return (%)',
        value=default_target_pct,
        step=0.10,
        format='%.2f',
        disabled=mode != 'Target Return',
    )
    target_vol_pct = st.number_input(
        'Target volatility (%)',
        value=10.0,
        step=0.10,
        format='%.2f',
        disabled=mode != 'Target Volatility',
    )

min_weight = min_weight_pct / 100
max_weight = max_weight_pct / 100
bounds = [(min_weight, max_weight)] * len(assets)

if mode == 'Target Return':
    target_return = target_return_pct / 100
    res = optimize_target_return(mu, cov, target_return, bounds)
    if not res.success:
        st.error(f'Optimization failed: {res.message}')
        st.stop()
    w = res.x
else:
    target_vol = target_vol_pct / 100
    res = optimize_target_vol(mu, cov, target_vol, bounds)
    if not res.success:
        st.error(f'Optimization failed: {res.message}')
        st.stop()
    w = res.x

exp_return = float(w @ mu)
vol = float(np.sqrt(w.T @ cov @ w))

st.subheader('Commodity Metrics')

asset_vols = np.sqrt(np.diag(cov))
asset_rets = mu
asset_sharpe = np.divide(asset_rets, asset_vols, out=np.zeros_like(asset_rets), where=asset_vols > 0)

metrics_df = pd.DataFrame({
    'Commodity': assets,
    'Return': [fmt_pct(v) for v in asset_rets],
    'Risk': [fmt_pct(v) for v in asset_vols],
    'Sharpe Ratio': np.round(asset_sharpe, 3),
})
st.dataframe(metrics_df, width='stretch')

st.subheader('Results')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Expected Return', fmt_pct(exp_return))
with col2:
    st.metric('Volatility', fmt_pct(vol))
with col3:
    st.empty()

weights_df = pd.DataFrame({'Asset': assets, 'Weight': w})
weights_df['Weight'] = weights_df['Weight'].apply(fmt_pct)

st.dataframe(weights_df, width='stretch')

st.subheader('Return vs Risk Profile')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(asset_vols, asset_rets, color='#2a7ea0', label='Assets')
for name, x, y in zip(assets, asset_vols, asset_rets):
    ax.annotate(name, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.scatter([vol], [exp_return], color='red', label='Portfolio', zorder=5)
ax.annotate('Portofolio', (vol, exp_return), textcoords='offset points', xytext=(6, -10), fontsize=9, color='red')
ax.set_xlabel('Volatility (annual)')
ax.set_ylabel('Expected Return (annual)')
ax.legend()
st.pyplot(fig)
