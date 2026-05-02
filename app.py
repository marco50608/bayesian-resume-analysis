import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime, timezone
import time
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
# Anonymous Logging (Google Sheets, fire-and-forget)
# -----------------------------------------------------------------------------
def logging_enabled() -> bool:
    """Cheap pre-check so callers can skip the whole debounce / autorefresh
    machinery when no logging credentials are configured. Without this, the
    app would still spin through the rerun loop trying to log on every
    debounce tick even when nothing could ever be written."""
    try:
        return ("gcp_service_account" in st.secrets) and ("sheet_id" in st.secrets)
    except Exception:
        # st.secrets raises if no secrets.toml exists at all
        return False


def log_event(prior_label: str, prior_alpha: float, prior_beta: float, strategies: list) -> bool:
    """
    Log one row per analysis run to a Google Sheet. Numeric inputs only — no
    names, emails, IPs, or identifying information. Silently no-ops if
    credentials aren't configured. Never blocks or breaks the app.

    Session-level rate limit: max 1 write per 10 seconds per session.

    `prior_alpha` and `prior_beta` are logged separately from `prior_label`
    because under "Slider (Custom)" mode the label alone hides the actual
    (α, β) — and the posterior is sensitive to the prior, so aggregating
    rows by label only would lump together very different analyses.

    Returns True iff a row was successfully appended (so the caller can decide
    whether to advance `last_logged_fp`). Returns False when secrets are
    missing, the rate limit blocks the write, or any exception is caught.
    """
    # --- session-level rate limit ---
    RATE_LIMIT_SECONDS = 10.0
    now = time.time()
    last = st.session_state.get("_log_event_last_ts", 0.0)
    if now - last < RATE_LIMIT_SECONDS:
        return False  # silently drop — too soon since last write
    st.session_state["_log_event_last_ts"] = now

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # Defensive re-check (callers should already have gated on
        # logging_enabled(), but keep this so the function is safe to call
        # standalone).
        if "gcp_service_account" not in st.secrets or "sheet_id" not in st.secrets:
            st.session_state["_log_event_last_ts"] = last  # don't burn the rate-limit slot
            return False

        # Only the spreadsheets scope is needed for `open_by_key` + `append_row`.
        # The wider `drive` scope is unnecessary and increases blast radius if
        # the service-account key ever leaks.
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=scopes,
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(st.secrets["sheet_id"]).sheet1

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        num_strat = len(strategies)
        row = [ts, prior_label, float(prior_alpha), float(prior_beta), num_strat]
        for i in range(5):
            if i < num_strat:
                s = strategies[i]
                row += [s["n"], s["k"], s["invalid"]]
            else:
                row += ["", "", ""]
        sheet.append_row(row, value_input_option="RAW")
        return True
    except Exception:
        # Roll back the timestamp so a failed write doesn't block the next attempt
        st.session_state["_log_event_last_ts"] = last
        return False


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def hex_to_rgba_str(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'

@st.cache_data(max_entries=30, show_spinner=False)
def render_png(fig_json: str, width: int = 1200, height: int = 600, scale: int = 2) -> bytes:
    """
    Render a Plotly figure (passed as JSON string so it's hashable for cache)
    to PNG via kaleido. Cached by (fig_json, width, height, scale), so the
    same inputs only ever spawn headless Chromium once.
    """
    fig = go.Figure(json.loads(fig_json))
    return fig.to_image(format="png", width=width, height=height, scale=scale)


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Resume Bayesian Analyzer", page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# URL state (Permalink) — read query params as widget defaults so a shared
# link reproduces the inputs. Widget `value=` only takes effect on first
# render, so once the user interacts, session_state takes over (which is what
# we want — links seed initial state, they don't override live input).
# -----------------------------------------------------------------------------
_qp = st.query_params

def _qp_int(key, default, min_value=None, max_value=None):
    """Read an int query param, fall back to default on error, then clamp.
    Hostile or stale URLs (?pa=999999) won't blow past widget min/max bounds."""
    try:
        v = _qp.get(key)
        v = int(v) if v is not None else default
    except (ValueError, TypeError):
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v

def _qp_str(key, default, max_len=None):
    v = _qp.get(key)
    if v is None:
        v = default
    v = str(v)
    if max_len is not None:
        v = v[:max_len]
    return v

def _clean_label(label, max_len: int = 40) -> str:
    """Strip newlines, trim, cap length so label can't break markdown layout
    or smuggle very long URLs / image references into rendered output."""
    label = str(label).replace("\n", " ").replace("\r", " ").strip()
    return label[:max_len] if label else "Unnamed strategy"

st.title("📊 Resume Conversion Rate Analyzer (Bayesian Approach)")
st.markdown("""
This tool uses **Bayesian Inference** to distinguish between *skill* and *luck*.
It calculates the **True Conversion Rate** distribution and visualizes the uncertainty for each resume version.
""")

if 'privacy_notice_shown' not in st.session_state:
    st.session_state.privacy_notice_shown = True
    # Only surface the privacy toast when logging is actually configured.
    # On a self-hosted deploy without secrets, nothing is logged, so a
    # logging notice would be misleading.
    if logging_enabled():
        st.toast(
            "ℹ️ This app may log anonymous numeric inputs only. See footer for details.",
            icon="🔒",
        )

with st.expander("ℹ️ How this works (and what the numbers mean)"):
    st.markdown("""
**The model.** Each strategy's true conversion rate is treated as a Beta-distributed random variable.
We start from a prior Beta(α, β) and update it with your data using the conjugate rule:
Posterior = Beta(α + interviews, β + (valid_apps − interviews))
**What you see.**
- **Posterior (PDF tab)** — your current belief about the true rate after seeing data. Narrower = more certain.
- **Forest plot** — 95% equal-tailed credible intervals. Overlapping intervals ≈ indistinguishable strategies.
- **Effort survival** — probability of getting ≥1 interview as you send more applications.
- **Reverse goal calculator** — apps needed to reach a target offer count with the chosen probability of success.
- **P(Row > Column) matrix** — Monte Carlo probability from posteriors. 50% = no signal; 95%+ = strong evidence.

**What this tool doesn't do.** It doesn't tell you *why* a resume works. With n=15, k=0 you'll see a wide posterior — that's the honest answer, not a bug.

**Priors.** The default slider (1, 1) is equivalent to Beta(1,1) = uniform. Jeffreys Beta(0.5, 0.5) is the objective reference prior; Flat is the same as slider at (1,1). If you have strong reason to think conversion rates are typically low, increase Beta to e.g. (1, 20).

Full write-up: [Medium post](#) · Code: [GitHub](#)
    """)

# -----------------------------------------------------------------------------
# 2. Sidebar: User Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# Prior Selection — defaults from URL query params if present
PRIOR_MODES = ["Slider (Custom)", "Jeffreys (0.5, 0.5)", "Flat (1, 1)"]
_url_prior = _qp_str("prior", "Slider (Custom)")
_prior_idx = PRIOR_MODES.index(_url_prior) if _url_prior in PRIOR_MODES else 0

prior_mode = st.sidebar.radio(
    "1. Prior Belief Mode",
    PRIOR_MODES,
    index=_prior_idx,
    help="Choose how to set your initial assumptions."
)

if prior_mode == "Slider (Custom)":
    prior_alpha = st.sidebar.slider("Prior Successes (Alpha)", 1, 50, _qp_int("pa", 1, 1, 50))
    prior_beta = st.sidebar.slider("Prior Failures (Beta)", 1, 50, _qp_int("pb", 1, 1, 50))
elif prior_mode == "Jeffreys (0.5, 0.5)":
    prior_alpha, prior_beta = 0.5, 0.5
else:
    prior_alpha, prior_beta = 1.0, 1.0

st.sidebar.markdown("---")
st.sidebar.header("📝 Strategy Data")

# Author's real job-search data (V1/V2/V3 from the Medium post)
EXAMPLE = [
    {"label": "V1 (English)",        "n": 23, "k": 0, "invalid": 2},
    {"label": "V2 (German CV)",      "n": 20, "k": 5, "invalid": 6},
    {"label": "V3 (English, tuned)", "n": 15, "k": 0, "invalid": 1},
]

# Load Example button — explicitly writes session_state and reruns. Streamlit
# widget `value=` only applies on first render, so a checkbox-driven approach
# silently fails to refill fields after the user has interacted with them.
if st.sidebar.button("📥 Load example data",
                     help="Overwrite all strategy fields with the author's real job-search data (see Medium post)."):
    st.session_state["ns"] = len(EXAMPLE)
    for i, ex in enumerate(EXAMPLE):
        st.session_state[f"name_{i}"] = ex["label"]
        st.session_state[f"n_{i}"] = ex["n"]
        st.session_state[f"k_{i}"] = ex["k"]
        st.session_state[f"inv_{i}"] = ex["invalid"]
    st.rerun()

num_strategies = st.sidebar.number_input(
    "How many versions to compare?",
    min_value=1,
    max_value=5,
    value=_qp_int("ns", 2, 1, 5),
    key="ns",
)

strategies_data = []
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']  # Blue, Red, Green, Purple, Yellow

for i in range(num_strategies):
    # URL params seed initial widget values; clamp to widget bounds so a hostile
    # or stale link can't blow past them and crash Streamlit.
    _label_default = _qp_str(f"l{i+1}", f"Version {i + 1}", max_len=40)
    _n_default = _qp_int(f"n{i+1}", 30, 1, 10000)
    _k_default = _qp_int(f"k{i+1}", 0, 0, 10000)
    _inv_default = _qp_int(f"iv{i+1}", 0, 0, 10000)

    st.sidebar.markdown(f"#### Strategy {i + 1}")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        label = st.text_input(
            "Name",
            value=_label_default,
            key=f"name_{i}",
            max_chars=40,
        )
        n_apps = st.number_input(
            "Total Apps",
            min_value=1,
            max_value=10000,
            value=_n_default,
            key=f"n_{i}",
        )

    with col2:
        k_interviews = st.number_input(
            "Interviews",
            min_value=0,
            max_value=10000,
            value=_k_default,
            key=f"k_{i}",
            help="Number of interviews/first-stage responses you actually received.",
        )
        n_invalid = st.number_input(
            "Noise (Invalid)",
            min_value=0,
            max_value=10000,
            value=_inv_default,
            key=f"inv_{i}",
            help="External rejections unrelated to resume quality (Visa, Language, etc.).",
        )

    strategies_data.append({
        "label": _clean_label(label),
        "n": n_apps,
        "k": k_interviews,
        "invalid": n_invalid,
        "color": colors[i % len(colors)]
    })

# -----------------------------------------------------------------------------
# Permalink: write current inputs to URL so the link can be shared.
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔗 Generate shareable link"):
    new_params = {
        "prior": prior_mode,
        "ns": str(int(num_strategies)),
    }
    if prior_mode == "Slider (Custom)":
        new_params["pa"] = str(int(prior_alpha))
        new_params["pb"] = str(int(prior_beta))
    for i, s in enumerate(strategies_data, start=1):
        new_params[f"l{i}"] = s["label"]
        new_params[f"n{i}"] = str(int(s["n"]))
        new_params[f"k{i}"] = str(int(s["k"]))
        new_params[f"iv{i}"] = str(int(s["invalid"]))
    st.query_params.clear()
    st.query_params.update(new_params)
    st.sidebar.success("URL updated — copy it from your browser's address bar to share.")

# -----------------------------------------------------------------------------
# 3. Analysis Engine
# -----------------------------------------------------------------------------
# --- session state init ---
DEBOUNCE_SECONDS = 2.5

for key, default in [
    ('run_analysis',     False),
    ('last_logged_fp',   None),   # fingerprint of last row written to sheet
    ('pending_fp',       None),   # fingerprint currently being observed
    ('pending_since',    0.0),    # when did pending_fp first appear?
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.button("🚀 Run Bayesian Analysis", type="primary"):
    st.session_state.run_analysis = True

if st.session_state.run_analysis:
    st.divider()

    # --- Pre-flight validation (BEFORE logging) ---
    # Hard-stop on inputs that would silently produce garbage. Collect ALL
    # errors first so the user can fix them in one pass instead of one at a
    # time. Earlier versions silently clamped k to effective_n and let
    # effective_n=0 fall through to a Beta(prior_alpha, prior_beta) "winner"
    # — both behaviours could produce a fake winner backed by zero data.
    #
    # Validation runs BEFORE the logging block so that invalid inputs never
    # make it into the research dataset (otherwise the "click Run with bad
    # data, get blocked" path would still leak a row to the Sheet first).
    input_errors = []
    for s in strategies_data:
        effective_n = s['n'] - s['invalid']
        if effective_n <= 0:
            input_errors.append(
                f"**{s['label']}**: 0 valid applications after removing noise "
                f"(Total {s['n']} − Noise {s['invalid']} ≤ 0). "
                f"Reduce Noise or add more applications before running the analysis."
            )
        elif s['k'] > effective_n:
            input_errors.append(
                f"**{s['label']}**: Interviews ({s['k']}) exceed valid applications "
                f"({effective_n} = Total {s['n']} − Noise {s['invalid']}). "
                f"Noise can only come from applications that did NOT result in an interview, "
                f"so Noise ≤ Total − Interviews."
            )

    if input_errors:
        for msg in input_errors:
            st.error(msg)
        st.stop()

    # --- Fingerprint (used by both logging and PNG cache, so always compute) ---
    current_fp = str(prior_mode) + str(strategies_data)
    now = time.time()

    # --- Logging (only when secrets are configured) ---
    # logging_enabled() short-circuits the whole debounce + autorefresh loop
    # when no credentials exist, so a self-hosted deploy without a Sheet
    # doesn't waste cycles re-checking on every rerun.
    if logging_enabled():
        # If the data changed since last rerun, reset the debounce clock
        if current_fp != st.session_state.pending_fp:
            st.session_state.pending_fp   = current_fp
            st.session_state.pending_since = now

        # Is there a pending change that hasn't been logged yet?
        unlogged = (current_fp != st.session_state.last_logged_fp)
        stable_for = now - st.session_state.pending_since

        if unlogged and stable_for >= DEBOUNCE_SECONDS:
            # Data has been stable for long enough — write it. Only advance
            # the fingerprint if the write actually succeeded; otherwise we'd
            # silently lose the row AND skip retrying on the next stable window.
            if log_event(prior_mode, prior_alpha, prior_beta, strategies_data):
                st.session_state.last_logged_fp = current_fp
        elif unlogged:
            # Still within debounce window — schedule a rerun to re-check in 0.5s.
            # Streamlit only reruns on user interaction, so we need this tick
            # to actually notice "2.5s of inactivity have passed".
            st_autorefresh(interval=500, limit=20, key="debounce_tick")

    # Storage for results
    results = []

    # X-axis for PDF — avoid the 0 / 1 endpoints because Beta(α<1, β) and
    # Beta(α, β<1) have infinite density at the boundary (e.g. Jeffreys
    # Beta(0.5, 0.5) prior). Plotly handles inf badly: the y-axis collapses
    # and the curve disappears.
    _eps = 1e-6
    x = np.linspace(_eps, 1.0 - _eps, 1000)

    # --- Calculation Loop ---
    rng = np.random.default_rng(222)
    for s in strategies_data:
        # Validation above guarantees: effective_n > 0 and k <= effective_n.
        effective_n = s['n'] - s['invalid']
        success = s['k']

        # Bayesian Update
        post_alpha = prior_alpha + success
        post_beta = prior_beta + (effective_n - success)

        # Statistics
        mean_rate = post_alpha / (post_alpha + post_beta)

        # 95% Equal-Tailed Credible Interval
        # NOTE: not HDI — for skewed Beta posteriors these differ. ETI is simpler
        # and more common; HDI would require arviz.hdi or a custom solver.
        ci_lower, ci_upper = stats.beta.ppf([0.025, 0.975], post_alpha, post_beta)

        # Generate Distribution Curve. Replace any residual inf (extreme priors
        # close to the boundary even after the eps offset) with NaN so Plotly
        # skips that point instead of compressing the y-axis to accommodate it.
        pdf = stats.beta.pdf(x, post_alpha, post_beta)
        pdf = np.where(np.isfinite(pdf), pdf, np.nan)

        # Calculate Survival Function (SF) = 1 - CDF
        sf = stats.beta.sf(x, post_alpha, post_beta)

        # Monte Carlo Sampling
        samples = rng.beta(post_alpha, post_beta, 10000)

        results.append({
            "data": s,
            "effective_n": effective_n, # Store for table
            "post_alpha": post_alpha,
            "post_beta": post_beta,
            "mean": mean_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pdf_y": pdf,
            "sf_y": sf,
            "samples": samples
        })

    # Find Winner (by highest mean)
    sorted_results = sorted(results, key=lambda x: x['mean'], reverse=True)
    winner = sorted_results[0]

    # Determine dynamic X-axis range for better visualization
    # Find the highest upper bound among all strategies to set the view
    max_upper_bound = max(r['ci_upper'] for r in results)
    # Add some padding (e.g., 20%) but cap at 1.0
    view_range_max = min(1.0, max_upper_bound * 1.5)
    # Ensure minimum range of 0.2 so it doesn't look too zoomed in for low data
    view_range_max = max(0.2, view_range_max)

    # Calculate "Probability of Being Best" (Monte Carlo).
    # Earlier versions only compared the highest-mean strategy against the
    # runner-up, which overstates dominance when there are 3+ arms — beating
    # the second-best is not the same as being best overall.
    # Here we draw paired samples from each posterior and ask, "in what
    # fraction of joint draws is `winner` the maximum across ALL arms?"
    if len(results) > 1:
        sample_matrix = np.vstack([r['samples'] for r in results])  # (n_strat, 10000)
        best_idx_by_draw = np.argmax(sample_matrix, axis=0)
        winner_idx = results.index(winner)
        prob_best = float(np.mean(best_idx_by_draw == winner_idx))
        win_msg = (
            f"**{winner['data']['label']}** has the highest posterior mean "
            f"and a **{prob_best:.1%}** posterior probability of being the best across all "
            f"{len(results)} strategies."
        )
    else:
        win_msg = f"**{winner['data']['label']}** is your current baseline."

    # -----------------------------------------------------------------------------
    # 4. Visualizations (Tabs)
    # -----------------------------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions (PDF)", "🌳 Forest Plot (Comparison)", "⏳ Effort Survival (Simulation)", "🎯 Reverse Goal Calculator"])

    # --- TAB 1: PDF Curves ---
    with tab1:
        st.subheader("Probability Density Function (PDF)")
        st.caption("Higher peaks = more certainty. Further right = better conversion rate.")

        fig_pdf = go.Figure()
        for res in results:
            s = res['data']
            fig_pdf.add_trace(go.Scatter(
                x=x, y=res['pdf_y'],
                customdata=res['sf_y'],
                mode='lines',
                name=f"{s['label']} ({res['mean']:.1%})",
                line=dict(color=s['color'], width=2.5),
                fill='tozeroy',
                fillcolor=hex_to_rgba_str(s['color'], 0.1),
                hovertemplate="Chance > %{x:.1%}: %{customdata:.1%}<extra></extra>"
            ))

        fig_pdf.update_layout(
            title=dict(text="Probability Density Function (True Conversion Rate)", x=0.5, xanchor='center'),
            xaxis_title="True Conversion Rate",
            yaxis_title="Probability Density",
            hovermode="x unified",
            xaxis=dict(tickformat=".0%", range=[0, view_range_max]),  # Dynamic range
            margin=dict(l=80, r=50, t=80, b=50),
            height=450
        )
        st.plotly_chart(fig_pdf, use_container_width=True)
        st.info(win_msg)
        
        st.markdown("""
        #### 💡 How to read this chart:
        *   **X-axis (True Conversion Rate)**: The possible "real" success rate of your resume.
        *   **Y-axis (Height)**: How likely that rate is. A taller, narrower peak means we are more sure.
        *   **Hover Info ("Chance > X")**: This is the most useful number!
            *   It tells you: *"What is the probability that my true skill is **better** than this number?"*
            *   *Example:* If you hover at **10%** and see **"Chance > 10%: 80%"**, it means there is an **80% probability** that your actual conversion rate is higher than 10%.
        """)
        # -----------------------------------------------------------------------------
        # Pairwise P(A > B) Matrix
        # -----------------------------------------------------------------------------
        if len(results) >= 2:
            st.markdown("### 🔀 Pairwise Probability Matrix")
            st.caption(
                "Probability that the **row** strategy's true conversion rate exceeds the **column**'s, "
                "based on 10,000 posterior samples. Values near 50% mean the data can't distinguish them; "
                "values above 95% are strong evidence."
            )

            labels = [r['data']['label'] for r in results]
            n_strat = len(results)
            matrix = np.full((n_strat, n_strat), np.nan)
            for i in range(n_strat):
                for j in range(n_strat):
                    if i != j:
                        matrix[i, j] = (results[i]['samples'] > results[j]['samples']).mean()

            prob_df = pd.DataFrame(matrix, index=labels, columns=labels)
            st.dataframe(
                prob_df.style.format("{:.1%}", na_rep="—").background_gradient(
                    cmap="OrRd", vmin=0, vmax=1, axis=None
                ).highlight_null(color="#F8F9FA"),
                use_container_width=True,
            )

        else:
            st.info("Add a second strategy in the sidebar to enable pairwise comparison.")

        # --- Summary Table ---
        st.markdown("### 📋 Detailed Statistics")
        df_table = pd.DataFrame([{
            "Strategy": r['data']['label'],
            "Valid Apps": r['effective_n'],
            "Interviews": r['data']['k'],
            "Mean Rate": f"{r['mean']:.1%}",
            "95% CrI Lower": f"{r['ci_lower']:.1%}",
            "95% CrI Upper": f"{r['ci_upper']:.1%}",
        } for r in results])

        st.dataframe(df_table.reset_index(drop=True), use_container_width=True)

    # --- TAB 2: Forest Plot ---
    with tab2:
        st.subheader("Forest Plot: 95% Credible Intervals")
        st.caption(
            "Compare the ranges. If intervals overlap significantly, the strategies might be statistically similar.")

        fig_forest = go.Figure()

        for res in reversed(results):
            s = res['data']
            fig_forest.add_trace(go.Scatter(
                x=[res['mean']],
                y=[s['label']],
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[res['ci_upper'] - res['mean']],
                    arrayminus=[res['mean'] - res['ci_lower']],
                    color=s['color'],
                    thickness=3,
                    width=10
                ),
                mode='markers',
                marker=dict(color=s['color'], size=12),
                name=s['label'],
                hovertemplate=f"<b>{s['label']}</b><br>Mean: {res['mean']:.1%}<br>95% CrI: {res['ci_lower']:.1%} - {res['ci_upper']:.1%}<extra></extra>"
            ))

        fig_forest.update_layout(
            title=dict(text="Forest Plot: 95% Credible Intervals", x=0.5, xanchor='center'),
            xaxis=dict(title="Conversion Rate", tickformat=".0%",
                       range=[0, min(1.0, max([r['ci_upper'] for r in results]) * 1.2)]),
            yaxis=dict(title="Strategy"),
            showlegend=False,
            height=300 + (len(results) * 30),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # --- TAB 3: Effort Survival Analysis ---
    with tab3:
        st.subheader("Effort Simulation (Geometric Distribution)")
        st.caption(
            "Based on your conversion rate, what is the probability of getting AT LEAST ONE interview within the next N applications?")

        fig_surv = go.Figure()
        efforts = np.arange(0, 51)

        for i, res in enumerate(results):
            s = res['data']
            samples_p = res['samples']  # 10k posterior draws

            # Marginalize over the posterior: for each effort level,
            # average P(at least one interview) across all posterior draws
            prob_success = np.array([
                1 - np.mean((1 - samples_p) ** e) for e in efforts
            ])

            fig_surv.add_trace(go.Scatter(
                x=efforts,
                y=prob_success,
                mode='lines',
                name=s['label'],
                line=dict(color=s['color'], width=3, dash='solid'),
                hovertemplate=f"<b>{s['label']}</b><br>After %{{x}} apps: %{{y:.1%}} chance of ≥1 interview<extra></extra>"
            ))

            try:
                idx_90 = next(j for j, val in enumerate(prob_success) if val >= 0.9)

                y_offset = -30 - (i % 3) * 25

                fig_surv.add_annotation(
                    x=idx_90, y=0.9,
                    text=f"90% chance at {idx_90} apps",
                    showarrow=True, arrowhead=1,
                    ax=0, ay=y_offset,  # <--- 使用動態高度
                    font=dict(color=s['color'], size=10)
                )
            except StopIteration:
                pass

        fig_surv.update_layout(
            title=dict(text="Effort Simulation (Probability of ≥1 Interview)", x=0.5, xanchor='center'),
            xaxis_title="Number of Future Applications",
            yaxis_title="Probability of ≥1 Interview",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            hovermode="x unified",
            margin=dict(l=80, r=50, t=80, b=50),
            height=450
        )

        fig_surv.add_hline(y=0.5, line_dash="dot", annotation_text="50% Probability", annotation_position="bottom right")
        fig_surv.add_hline(y=0.9, line_dash="dot", annotation_text="90% Probability", annotation_position="bottom right")

        st.plotly_chart(fig_surv, use_container_width=True)
        st.markdown("""
        **How to read this:**
        * If the line crosses **90% at 20 apps**, it means: *"If I send 20 more applications, I am 90% sure I'll get an interview."*
        * **Steeper curve** = Better strategy (Less effort required).
        """)

    # --- TAB 4: Reverse Goal Calculator ---
    with tab4:
        st.subheader("🎯 Reverse Goal Calculator (Action Plan)")
        st.markdown(
            "**How many applications do I need to send to be reasonably sure I'll get my target offers?** "
            "The number returned below is the smallest *N* such that the marginal probability of "
            "reaching the goal in *N* applications meets your chosen confidence level. "
            "This integrates **two** sources of uncertainty: posterior uncertainty about your true "
            "app→interview rate, *and* binomial sampling noise in actual outcomes — so it doesn't "
            "fall into the trap of `1 / rate` thinking, where a 5% rate naïvely says \"20 apps for "
            "1 offer\" but actually gives only ~64% probability of success."
        )

        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            target_offers = st.number_input("Target Number of Offers", min_value=1, value=1, step=1)
        with col_input2:
            offer_rate = st.slider(
                "Estimated Interview-to-Offer Rate", 0.0, 1.0, 0.2, 0.05, format="%.0f%%",
                help="If you get 5 interviews, how many offers do you expect? (0.2 = 1 offer per 5 interviews)"
            )
        with col_input3:
            confidence = st.slider(
                "Desired probability of reaching target", 0.50, 0.99, 0.80, 0.05, format="%.0f%%",
                help="Higher confidence ⇒ more applications recommended."
            )

        st.divider()

        # Search grid bounded for performance. ~1000 apps × scipy.binom.cdf
        # (vectorised over 10k posterior draws per call) is well under a
        # second per strategy.
        MAX_APPS = 1000
        apps_grid = np.arange(1, MAX_APPS + 1)

        for res in results:
            s = res['data']
            samples_p = res['samples']  # 10k posterior draws of app→interview rate
            mean_rate = res['mean']
            overall_mean = mean_rate * offer_rate

            st.markdown(f"### Strategy: {s['label']}")

            if offer_rate <= 0:
                st.warning("Cannot compute — Interview-to-Offer rate is 0.")
                st.markdown("---")
                continue

            # Overall conversion rate per posterior draw: app → interview → offer.
            overall_samples = samples_p * offer_rate

            # For each candidate N_apps, marginal P(≥ target_offers in N apps),
            # averaging over the posterior. Inner call is C-vectorised over the
            # 10k draws, outer loop is just 1000 scalars — fast in practice.
            prob_reach_goal = np.array([
                float(np.mean(1.0 - stats.binom.cdf(target_offers - 1, n, overall_samples)))
                for n in apps_grid
            ])

            meets = np.where(prob_reach_goal >= confidence)[0]
            if meets.size > 0:
                apps_needed = int(apps_grid[meets[0]])
                apps_needed_str = str(apps_needed)
            else:
                apps_needed = None
                apps_needed_str = f">{MAX_APPS}"

            c1, c2, c3 = st.columns(3)
            c1.metric("App→Interview Rate (mean)", f"{mean_rate:.1%}")
            c2.metric("Overall Success Rate (mean)", f"{overall_mean:.1%}",
                      help="Posterior-mean app→interview × interview→offer.")
            c3.metric(
                f"Apps for {confidence:.0%} chance",
                apps_needed_str,
                help=(
                    f"Smallest N such that the marginal probability of getting at least "
                    f"{target_offers} offer{'s' if target_offers > 1 else ''} in N apps is ≥ {confidence:.0%}."
                ),
            )

            if apps_needed is not None:
                st.info(
                    f"To have a **{confidence:.0%} probability** of getting "
                    f"**≥ {target_offers} offer{'s' if target_offers > 1 else ''}** with this strategy, "
                    f"send approximately **{apps_needed} applications**. "
                    f"This is the marginal probability over both posterior uncertainty in your true "
                    f"conversion rate and the binomial randomness of actual outcomes."
                )
            else:
                st.warning(
                    f"Even {MAX_APPS} applications give only a {prob_reach_goal[-1]:.1%} probability "
                    f"of reaching this goal with this strategy. Consider lowering the target, "
                    f"raising the offer-rate estimate, or accepting a lower confidence level."
                )
            st.markdown("---")




    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Powered by Bayesian Inference. · "
        "If the operator has configured anonymous logging, clicking \"Run Bayesian Analysis\" "
        "may record numeric inputs (application counts, interview counts, prior settings) "
        "for aggregate usage research. No names, emails, IPs, cookies, or identifying "
        "information are stored; rows cannot be linked back to you or your session. "
        "Logging is debounced — rows are only written after ~2.5 seconds of input stability, "
        "so editing your values within that window prevents the row from being saved. "
        "Legal basis: GDPR Art. 6(1)(f) — legitimate interest in improving the tool."
    )

    # -----------------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------------
    st.markdown("### 💾 Export")

    # Side-by-side layout so the PNG expander doesn't span the full page width
    # — st.expander always fills its parent container, so we bound the parent
    # via a column instead.
    col_csv, col_png = st.columns([1, 2])

    with col_csv:
        # CSV is cheap — always ready
        csv_bytes = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download table (CSV)",
            data=csv_bytes,
            file_name="resume_conversion_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_png:
        # PNGs are expensive (kaleido spawns headless Chromium) — lazy-load
        # behind a button. We also key the render cache by a fingerprint so
        # the same fingerprint never re-renders even if the user toggles the
        # button off and on.
        with st.expander("🖼️ Download charts as PNG", expanded=False):
            st.caption(
                "Generating PNGs spins up a headless browser and takes ~1–2 seconds. "
                "Click below only when you actually want to download."
            )

            png_fp_key = f"png_ready_{current_fp}"  # one flag per data fingerprint
            if png_fp_key not in st.session_state:
                st.session_state[png_fp_key] = False

            if not st.session_state[png_fp_key]:
                if st.button("Generate PNG downloads", key=f"gen_png_{current_fp}"):
                    st.session_state[png_fp_key] = True
                    st.rerun()
            else:
                try:
                    with st.spinner("Rendering charts…"):
                        pdf_png_bytes = render_png(fig_pdf.to_json())
                        forest_png_bytes = render_png(fig_forest.to_json())
                        surv_png_bytes = render_png(fig_surv.to_json())

                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        st.download_button(
                            "📈 PDF chart",
                            data=pdf_png_bytes,
                            file_name="resume_conversion_pdf.png",
                            mime="image/png",
                        )
                    with dl_col2:
                        st.download_button(
                            "🌳 Forest plot",
                            data=forest_png_bytes,
                            file_name="resume_conversion_forest.png",
                            mime="image/png",
                        )
                    with dl_col3:
                        st.download_button(
                            "⏳ Survival chart",
                            data=surv_png_bytes,
                            file_name="resume_conversion_survival.png",
                            mime="image/png",
                        )
                except Exception as e:
                    st.error(
                        "PNG export requires `kaleido` — make sure `kaleido==0.2.1` is in requirements.txt. "
                        f"(Error: {type(e).__name__})"
                    )
