import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import re
import requests
from io import StringIO
import matplotlib.pyplot as plt


MINERBA_URL = 'https://www.minerba.esdm.go.id/harga_acuan'
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'price_cache.csv')

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
    match = re.search(r'([A-Za-z]+)\s+(?:(I{1,3}|[12])\s+)?(\d{4})', col_name)
    if not match:
        return None
    month_name = match.group(1)
    half_marker = match.group(2)
    year = int(match.group(3))
    month = MONTH_MAP.get(month_name)
    if not month:
        return None
    day = 1
    if half_marker:
        half_marker = half_marker.upper()
        if half_marker in ('II', '2'):
            day = 16
    return pd.Timestamp(year=year, month=month, day=day)


def fetch_minerba_prices(start_mm_yyyy, end_mm_yyyy):
    """Fetch prices from Minerba API for the given date range."""
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
    df = df.apply(pd.to_numeric, errors='coerce')
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
    grouped = long_df.groupby(['Komoditas', 'Date'], as_index=False)['Value'].mean()
    pivot = grouped.pivot(index='Date', columns='Komoditas', values='Value').sort_index()
    return pivot


def load_minerba_prices():
    """Load prices with file-based caching. Only fetches new data since last cache."""
    current_month = pd.Timestamp.today().replace(day=1)
    cached_df = None

    # Load existing cache
    if os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        except Exception:
            cached_df = None

    if cached_df is not None and not cached_df.empty:
        last_cached = cached_df.index.max()
        # If cache is current, return it
        if last_cached >= current_month:
            return cached_df
        # Fetch only missing data (from last cached month to now)
        start_mm_yyyy = f"{last_cached.month:02d}/{last_cached.year}"
        end_mm_yyyy = f"{current_month.month:02d}/{current_month.year}"
        new_data = fetch_minerba_prices(start_mm_yyyy, end_mm_yyyy)
        if not new_data.empty:
            # Merge: new data overwrites overlapping dates
            combined = pd.concat([cached_df, new_data])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            combined.to_csv(CACHE_FILE)
            return combined
        return cached_df
    else:
        # No cache, fetch all historical data
        start_mm_yyyy = "01/2017"
        end_mm_yyyy = f"{current_month.month:02d}/{current_month.year}"
        full_data = fetch_minerba_prices(start_mm_yyyy, end_mm_yyyy)
        if not full_data.empty:
            full_data.to_csv(CACHE_FILE)
        return full_data


def compute_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna(how='any')


def infer_periods_per_year(index):
    if len(index) < 2:
        return 12
    deltas = np.diff(index.values).astype('timedelta64[D]').astype(int)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 12
    median_days = np.median(deltas)
    if median_days <= 0:
        return 12
    return max(1, int(round(365.25 / median_days)))


def annualize_mean_cov(returns_df, periods_per_year=None):
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(returns_df.index)
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




st.set_page_config(
    page_title='Commodity Portofolio Optimizer',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        button[kind="headerNoPadding"] {
            display: none !important;
        }
        .st-emotion-cache-1gwvy71 {
            display: none !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 320px;
            max-width: 320px;
        }
        section[data-testid="stSidebar"] button[aria-label="Close sidebar"] {
            display: none !important;
        }
        /* Keep select dropdowns aligned within sidebar width */
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stMultiSelect {
            width: 100%;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] {
            max-width: 100%;
        }
        [data-testid="stSidebar"] div[data-baseweb="popover"] {
            max-width: 320px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Commodity Portofolio Optimizer')

with st.sidebar:
    st.header('Inputs')
    st.caption('Data source: Harga Minerba Acuan')

try:
    raw_prices = load_minerba_prices()
except Exception as exc:
    st.error(f'Failed to load Minerba data: {exc}')
    st.stop()

if raw_prices.empty:
    st.error('No data returned from Minerba.')
    st.stop()

all_assets = list(raw_prices.columns)

with st.sidebar:
    assets = st.multiselect('Assets', all_assets, default=[])
    date_options = list(raw_prices.index)
    month_counts = {}
    for dt_value in date_options:
        key = (dt_value.year, dt_value.month)
        month_counts[key] = month_counts.get(key, 0) + 1
    month_seen = {}
    date_labels = []
    for dt_value in date_options:
        key = (dt_value.year, dt_value.month)
        month_seen[key] = month_seen.get(key, 0) + 1
        base = f"{MONTH_NAMES.get(dt_value.month, dt_value.month)} {dt_value.year}"
        if month_counts[key] > 1:
            label = f"{base} ({month_seen[key]})"
        else:
            label = base
        date_labels.append(label)
    def fmt_period(dt_value):
        return f"{MONTH_NAMES.get(dt_value.month, dt_value.month)} {dt_value.year}"
    from_idx = st.selectbox(
        'From period',
        list(range(len(date_options))),
        index=0,
        format_func=lambda i: date_labels[i],
    )
    to_idx = st.selectbox(
        'To period',
        list(range(len(date_options))),
        index=len(date_options) - 1,
        format_func=lambda i: date_labels[i],
    )
    from_date = date_options[from_idx]
    to_date = date_options[to_idx]

if len(assets) < 2:
    st.header('Welcome')
    st.markdown(
        """
This app builds a commodity portofolio using Harga Minerba Acuan data and Modern Portofolio Theory.

**Core logic (simple):**
- **Prices → Returns**: convert monthly prices into log returns.
- **Returns → Risk**: compute covariance matrix to capture correlation effects.
- **Optimization**: pick weights to hit a **target return** or **target risk**.

**Usage guide:**
1. Select at least 2 commodities from the sidebar.
2. Choose a **From period** and **To period**.
3. Set min/max weight constraints.
4. Pick **Target Return** or **Target Risk**.
5. Adjust the target value and review results.
        """
    )

    with st.expander('Why can portofolio risk be lower than any individual asset?', expanded=True):
        st.markdown(
            r"""
**The Short Answer:** Diversification. When assets don't move perfectly together,
their price movements partially cancel out, reducing overall portofolio volatility.

---

**The Math Behind It**

For a portofolio of $n$ assets with weights $w_i$, the portofolio variance is:

$$\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij}$$

In matrix form:

$$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$$

Where $\Sigma$ is the **covariance matrix**:
- Diagonal entries $\Sigma_{ii}$ are individual asset variances ($\sigma_i^2$).
- Off-diagonal entries $\Sigma_{ij}$ are covariances (how two assets move together).
- Lower covariance (or negative covariance) reduces total risk.

**3-asset covariance matrix with real commodities (example, in % units):**

Assume Batubara (Coal), Nikel (Nickel), and Emas (Gold) with weights:
            """
        )
        st.latex(
            r"""
\mathbf{w} =
\begin{bmatrix}
40\% \\\\
35\% \\\\
25\%
\end{bmatrix}
            """
        )
        st.markdown("Covariance matrix (entries in $\\%^{2}$):")
        st.latex(
            r"""
\Sigma =
\begin{bmatrix}
256 & 60 & 30 \\\\
60 & 625 & 40 \\\\
30 & 40 & 144
\end{bmatrix}
            """
        )
        st.markdown("Then:")
        st.latex(r"\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w} = 156.3225\ \%^{2}")
        st.markdown("So the portfolio risk is:")
        st.latex(r"\sigma_p = \sqrt{156.3225}\% \approx 12.50\%")
        st.markdown(
            r"""
This can be rewritten as:

$$\sigma_p^2 = \sum_{i=1}^{n} w_i^2 \sigma_i^2 + 2\sum_{i<j} w_i w_j \sigma_{ij}$$

Where:
- $\sigma_i^2$ = variance of asset $i$
- $\sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$ = covariance between assets $i$ and $j$
- $\rho_{ij}$ = correlation coefficient between assets $i$ and $j$ (ranges from -1 to +1)

**The key insight:** When $\rho_{ij} < 1$, the cross-terms $\sigma_{ij}$ are smaller than $\sigma_i \sigma_j$,
which pulls down the total portofolio variance below what you'd expect from a simple weighted average.

---

**Simple Two-Asset Example**

For two assets with equal weights ($w_1 = w_2 = 0.5$):

$$\sigma_p^2 = 0.25\sigma_1^2 + 0.25\sigma_2^2 + 0.5\rho_{12}\sigma_1\sigma_2$$

If both assets have 20% risk ($\sigma_1 = \sigma_2 = 0.2$):

| Correlation ($\rho$) | Portofolio Risk ($\sigma_p$) |
|:---:|:---:|
| +1.0 (perfect positive) | 20.0% |
| +0.5 | 17.3% |
| 0.0 (uncorrelated) | 14.1% |
| -0.5 | 10.0% |
| -1.0 (perfect negative) | 0.0% |

---

**Intuitive Understanding**

Think of it like this:
- **Correlation = +1**: Both assets always move together. No benefit from diversification.
- **Correlation = 0**: Assets move independently. When one zigs, the other might zag, smoothing out the bumps.
- **Correlation = -1**: Assets move in opposite directions. They perfectly offset each other.

In the real world, most commodities have correlations between 0 and +1.
This is why a well-constructed portofolio can achieve lower risk than holding any single commodity alone,
while still capturing the average returns of its components.
            """
        )

    st.info('Select at least 2 commodities in the sidebar to start.')
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
    max_return = float(np.max(mu))
    min_return = float(np.min(mu))
    if target_return > max_return:
        st.error('Maximum Return exceeded.')
        st.stop()
    if target_return < min_return:
        st.error('Minimum Return exceeded.')
        st.stop()
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
metrics_df_display = metrics_df.copy()
metrics_df_display.index = range(1, len(metrics_df_display) + 1)
st.dataframe(metrics_df_display, width='stretch')

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

weights_df_display = weights_df.copy()
weights_df_display.index = range(1, len(weights_df_display) + 1)
st.dataframe(weights_df_display, width='stretch')

st.subheader('Price Chart')
fig, ax = plt.subplots()
price_df.plot(ax=ax, linewidth=1.5)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend(loc='best', fontsize=8)
st.pyplot(fig)

st.subheader('Covariance / Correlation Matrix')
matrix_mode = st.radio('Matrix type', ['Correlation', 'Covariance'], index=0, horizontal=True)
if matrix_mode == 'Correlation':
    matrix_df = returns.corr()
    cmap = 'coolwarm'
    vmin, vmax = -1, 1
    fmt = '{:.2f}'
else:
    matrix_df = pd.DataFrame(cov, index=assets, columns=assets)
    cmap = 'viridis'
    vmin = float(np.nanmin(matrix_df.values))
    vmax = float(np.nanmax(matrix_df.values))
    fmt = '{:.2f}'
fig, ax = plt.subplots()
im = ax.imshow(matrix_df.values, cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(assets)))
ax.set_yticks(range(len(assets)))
ax.set_xticklabels(assets, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(assets, fontsize=8)
ax.set_xlabel('Assets')
ax.set_ylabel('Assets')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Show values in each cell for readability
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap_obj = plt.get_cmap(cmap)
for i in range(matrix_df.shape[0]):
    for j in range(matrix_df.shape[1]):
        val = matrix_df.iat[i, j]
        if pd.isna(val):
            label = ''
            color = 'black'
        else:
            label = fmt.format(val)
            rgba = cmap_obj(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = 'black' if luminance > 0.6 else 'white'
        ax.text(j, i, label, ha='center', va='center', fontsize=7, color=color)

st.pyplot(fig)

st.subheader('Return vs Risk Profile')

fig, ax = plt.subplots()
ax.scatter(asset_risks, asset_rets, color='#2a7ea0', label='Assets')
for name, x, y in zip(assets, asset_risks, asset_rets):
    ax.annotate(name, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.scatter([risk], [exp_return], color='red', label='Portofolio', zorder=5)
ax.annotate('Portofolio', (risk, exp_return), textcoords='offset points', xytext=(6, -10), fontsize=9, color='red')
ax.set_xlabel('Risk (annual)')
ax.set_ylabel('Expected Return (annual)')
ax.legend()
st.pyplot(fig)
