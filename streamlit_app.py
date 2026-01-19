import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import re
import requests
from io import StringIO


MINERBA_URL = 'https://www.minerba.esdm.go.id/harga_acuan'

MONTH_MAP = {
    'Januari': 1,
    'Februari': 2,
    'Maret': 3,
    'April': 4,
    'Mei': 5,
    'Juni': 6,
    'Juli': 7,
    'Agustus': 8,
    'September': 9,
    'Oktober': 10,
    'November': 11,
    'Desember': 12,
}
MONTH_NAMES = {v: k for k, v in MONTH_MAP.items()}

DEFAULT_ASSETS = [
    'Batubara (USD/ton)',
    'Nikel (USD/dmt)',
    'Aluminium (USD/dmt)',
    'Tembaga (USD/dmt)',
    'Bijih Besi Laterit/Hematit/Magnetit (USD/dmt)',
    'Emas sebagai mineral ikutan (USD/Troy Ounce)',
]


def parse_minerba_column(col_name):
    match = re.search(r'([A-Za-z]+)\s+(\d{4})', col_name)
    if not match:
        return None
    month_name = match.group(1)
    year = int(match.group(2))
    month = MONTH_MAP.get(month_name)
    if not month:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


@st.cache_data
def load_minerba_prices(start_mm_yyyy, end_mm_yyyy):
    session = requests.Session()
    html = session.get(MINERBA_URL, timeout=20).text
    match = re.search(r'name=\"csrf_test_name\" value=\"([^\"]+)\"', html)
    csrf_token = match.group(1) if match else ''
    payload = {
        'csrf_test_name': csrf_token,
        'bulan_awal': start_mm_yyyy,
        'bulan_akhir': end_mm_yyyy,
    }
    response = session.post(MINERBA_URL, data=payload, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    if 'Komoditas' not in df.columns:
        return pd.DataFrame()
    df = df.set_index('Komoditas')
    # Convert values to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Melt into long format
    long_rows = []
    for col in df.columns:
        date = parse_minerba_column(col)
        if date is None:
            continue
        series = df[col].dropna()
        for commodity, value in series.items():
            long_rows.append({'Komoditas': commodity, 'Date': date, 'Value': value})
    if not long_rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(long_rows)
    # Average across periods within the same month
    grouped = long_df.groupby(['Komoditas', 'Date'], as_index=False)['Value'].mean()
    pivot = grouped.pivot(index='Date', columns='Komoditas', values='Value').sort_index()
    return pivot


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


def optimize_target_risk(mu, cov, target_risk, bounds):
    n = len(mu)

    def port_risk(w):
        return np.sqrt(w.T @ cov @ w)

    def objective(w):
        return (port_risk(w) - target_risk) ** 2

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
    st.caption('Data source: Minerba Harga Acuan')

end_month = pd.Timestamp.today()
end_mm_yyyy = f"{end_month.month:02d}/{end_month.year}"
start_mm_yyyy = "01/2017"

try:
    raw_prices = load_minerba_prices(start_mm_yyyy, end_mm_yyyy)
except Exception as exc:
    st.error(f'Failed to load Minerba data: {exc}')
    st.stop()

if raw_prices.empty:
    st.error('No data returned from Minerba.')
    st.stop()

all_assets = list(raw_prices.columns)

with st.sidebar:
    defaults = [a for a in DEFAULT_ASSETS if a in all_assets]
    if not defaults:
        defaults = all_assets[:4]
    assets = st.multiselect('Assets', all_assets, default=defaults)
    date_options = list(raw_prices.index)
    def fmt_period(dt_value):
        return f"{MONTH_NAMES.get(dt_value.month, dt_value.month)} {dt_value.year}"
    from_date = st.selectbox('From period', date_options, index=0, format_func=fmt_period)
    to_date = st.selectbox('To period', date_options, index=len(date_options) - 1, format_func=fmt_period)

if len(assets) < 2:
    st.warning('Select at least 2 assets.')
    st.stop()

price_df = raw_prices[assets]
if from_date > to_date:
    st.error('From period must be earlier than or equal to To period.')
    st.stop()
price_df = price_df.loc[(price_df.index >= from_date) & (price_df.index <= to_date)]

coverage = price_df.notna().mean()
min_coverage = 0.8
valid_assets = coverage[coverage >= min_coverage].index.tolist()
removed_assets = [a for a in assets if a not in valid_assets]
if removed_assets:
    st.warning(
        'Excluded assets with insufficient history (< 80% coverage): '
        + ', '.join(removed_assets)
    )
    price_df = price_df[valid_assets]

if len(price_df.index) < 2:
    st.error('Not enough data for the selected time period.')
    st.stop()

returns = compute_log_returns(price_df)

mu, cov, mu_series = annualize_mean_cov(returns)

with st.sidebar:
    min_weight_pct = st.number_input('Min weight (%)', value=0.0, step=0.50, format='%.2f')
    max_weight_pct = st.number_input('Max weight (%)', value=100.0, step=0.50, format='%.2f')
    if min_weight_pct >= max_weight_pct:
        st.error('Min weight must be less than max weight.')
        st.stop()

    mode = st.selectbox('Optimization mode', ['Target Return', 'Target Risk'])
    default_target_pct = float(mu.mean()) * 100
    target_return_pct = st.number_input(
        'Target return (%)',
        value=default_target_pct,
        step=0.10,
        format='%.2f',
        disabled=mode != 'Target Return',
    )
    target_risk_pct = st.number_input(
        'Target risk (%)',
        value=10.0,
        step=0.10,
        format='%.2f',
        disabled=mode != 'Target Risk',
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
    target_risk = target_risk_pct / 100
    res = optimize_target_risk(mu, cov, target_risk, bounds)
    if not res.success:
        st.error(f'Optimization failed: {res.message}')
        st.stop()
    w = res.x

exp_return = float(w @ mu)
risk = float(np.sqrt(w.T @ cov @ w))

st.subheader('Commodity Metrics')

asset_risks = np.sqrt(np.diag(cov))
asset_rets = mu
asset_sharpe = np.divide(asset_rets, asset_risks, out=np.zeros_like(asset_rets), where=asset_risks > 0)

metrics_df = pd.DataFrame({
    'Commodity': assets,
    'Return': [fmt_pct(v) for v in asset_rets],
    'Risk': [fmt_pct(v) for v in asset_risks],
    'Sharpe Ratio': np.round(asset_sharpe, 3),
})
st.dataframe(metrics_df, width='stretch')

st.subheader('Results')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Expected Return', fmt_pct(exp_return))
with col2:
    st.metric('Risk', fmt_pct(risk))
with col3:
    st.empty()

weights_df = pd.DataFrame({'Asset': assets, 'Weight': w})
weights_df['Weight'] = weights_df['Weight'].apply(fmt_pct)

st.dataframe(weights_df, width='stretch')

st.subheader('Return vs Risk Profile')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(asset_risks, asset_rets, color='#2a7ea0', label='Assets')
for name, x, y in zip(assets, asset_risks, asset_rets):
    ax.annotate(name, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.scatter([risk], [exp_return], color='red', label='Portfolio', zorder=5)
ax.annotate('Portofolio', (risk, exp_return), textcoords='offset points', xytext=(6, -10), fontsize=9, color='red')
ax.set_xlabel('Risk (annual)')
ax.set_ylabel('Expected Return (annual)')
ax.legend()
st.pyplot(fig)
