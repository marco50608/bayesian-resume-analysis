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
        # --- user opt-out (GDPR Art. 21) ---
    if st.session_state.get("opt_out_logging"):
        return False
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
st.set_page_config(page_title="Bayesian Resume Conversion Analysis", page_icon="📊", layout="wide")

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

with st.expander("ℹ️ New here? Start with this — what this tool does, in plain English"):
    st.markdown("""
#### Start here: the story this tool came out of

In late 2025 I was applying for jobs in Germany with a fairly awkward profile — **A1 German** (which is to say, none), a **Philosophy and Law** background I was moving out of, and one year at Amazon. Over about three months I sent 58 applications.

I did not set out to run an experiment. I just kept rewriting my CV.

**Version 1 — my normal CV.** The standard English one-pager I would have used in Taiwan or the US. Clean, competent, the format everyone tells you to use. I sent it out and waited.

> **21 applications. Zero interviews.** Not one reply.

**Version 2 — the same CV, rebuilt to look German.** Identical person, identical jobs, identical content: same degree, same Amazon internship, same bullet points. The only thing I changed was **how it looked** — German section headings (*Lebenslauf*, *Berufserfahrung*), grades on the German 1.0–5.0 scale, the conservative layout German recruiters expect.

> **14 applications. Five interviews.**

Which looks like a spectacular result. 0% became 36%, just by changing the formatting.

And that is exactly the moment you should get suspicious. **Fourteen applications is nothing.** If I had got two interviews instead of five, the number would have been 14%. Streaks like this happen by luck all the time. So: is this real, or did I just get lucky at the right moment?

That question is the entire reason this tool exists. Type those four numbers into the sidebar — 21 and 0, then 14 and 5 — press **🚀 Run Bayesian Analysis**, and here is what comes back:

| | Applications | Interviews | Best guess | Honest range |
|---|---|---|---|---|
| **Version 1** (English CV) | 21 | 0 | **4.3%** | 0.1% – 15.4% |
| **Version 2** (German format) | 14 | 5 | **37.5%** | 16.3% – 61.6% |

**Version 2 is better, with 99.8% probability.**

---

#### What those three lines actually mean

**1. Zero interviews does not mean a zero rate.**
Version 1 got nothing at all, yet the tool says 4.3% rather than 0%. That is not the tool being polite. Twenty-one applications simply is not enough to prove something never happens: if my true rate had been 5%, sending 21 applications would have produced zero interviews about **a third of the time**. The honest reading of 0/21 is *"probably low, and I can't say more than that yet"* — which is what the range 0.1% – 15.4% says.

**2. The best guess is not the answer. The range is.**
Version 2's headline number is 36%, but the tool's real answer is *"somewhere between 16% and 62%."* That is an enormous span, and it is supposed to be. Five interviews out of fourteen pins down the truth very loosely. Anyone who quotes you a single confident percentage off a sample this size is showing you their arithmetic, not their evidence.

**3. You can be sure about the comparison while still being unsure about the number.**
This is the part that surprises people, and it is the whole point. The tool is **99.8% confident that Version 2 beats Version 1** — while flatly refusing to tell me whether Version 2's true rate is 20% or 55%. Those are two different questions. "Which one is better?" needs far less data than "exactly how good is it?", and only one of them was worth waiting for.

So the decision was easy, even on a laughably small sample: **keep using Version 2, and stop spending applications on Version 1 to find out.** That is a conclusion I could act on the same afternoon. Pinning down the exact rate would have taken hundreds more applications, and I did not need it.

*(These are my real numbers. The sidebar's **📥 Load example data** button loads the full three-version original, including the version I panic-switched to in the middle and should not have.)*

---

#### What counts as a different "version"?

This matters more than anything else on this page, and it is the easiest thing to get wrong.

A version is a CV you decided on once and then sent out **unchanged** across a batch of applications. The test: if you handed two of your applications to a stranger, would they say *"same CV"* or *"two different CVs"*?

**Counts as a new version**

- A different layout or template
- A different section order — education first vs experience first
- Switching to local conventions: section headings in the local language, the local grade scale, photo added or removed
- One page vs two pages

**Does not count**

- Swapping keywords to match each job description
- Rewording a bullet or the summary line for each posting
- Renaming the file

Per-application tailoring is normal and you should keep doing it — it just isn't a *version*. If you treat every tweak as its own version, you end up with twenty versions of one or two applications each, and the tool can tell you nothing useful about any of them.

⚠️ **The rule that is easiest to break:** if you changed the layout halfway through a batch, that batch is really two versions. Split it. Otherwise you are averaging two different things, and the number you get back describes neither of them.

---

#### The problem this tool solves

You sent some job applications. A few of them led to interviews; most didn't. You're staring at a number like *"5 interviews out of 14 applications = 36%"* and asking: **how confident can I actually be in that 36%?** If you'd sent only 3 more applications and got 1 more interview, the number would have been 38%. If you'd had 1 fewer success, it would be 29%. The raw rate is wobbly because the sample is small.

This tool gives you the honest answer: **a range, not a single number**. Instead of saying *"your conversion rate is 36%"*, it says something like *"your true underlying rate is most likely somewhere between 16% and 62%, with the best single guess around 38%."* That range is wide because 14 applications is a small sample — and the tool doesn't pretend otherwise.

#### How it works (Bayesian inference in 30 seconds)

1. **Prior** — what you believed *before* seeing your data. By default the tool starts from "I have no idea" (every conversion rate from 0% to 100% is equally plausible).
2. **Data** — your actual application counts.
3. **Posterior** — your updated belief *after* combining the prior with the data. This is the bell-shaped curve you see on the "Distributions" tab.

The math is a standard *Beta-Binomial conjugate update*: if your prior is `Beta(α, β)` and you observed `k` interviews out of `n` valid applications, your posterior is `Beta(α + k, β + n − k)`. You don't need to understand the formula to use the tool — but if a stats reviewer asks, that's what's happening under the hood.

#### What the tabs show you

- **📈 Distributions (PDF)** — your belief about each strategy's true rate as a curve. **Taller and narrower = more certain. Further right = better.** Hover anywhere to see *"how likely is the true rate to be at least X?"*.

- **🌳 Forest Plot** — each strategy as a horizontal line showing its 95% credible interval (the range that contains the true rate with 95% probability). **If two lines barely overlap, those strategies are probably different. If they overlap a lot, the data can't tell them apart.**

- **⏳ Effort Survival** — answers "how many applications until I'm X% sure of getting at least one interview?" Steeper curve = better strategy (less effort needed).

- **🎯 Reverse Goal Calculator** — answers "how many applications do I need to send for an N% chance of getting *Y* offers?" Takes both posterior uncertainty AND binomial sampling randomness into account, so it doesn't fall into the trap of thinking "1 / rate = number of applications needed".

- **🔀 Pairwise Probability Matrix** (under the Distributions tab) — for any two strategies A and B, shows `P(A's true rate > B's true rate)`. **Near 50% means the data can't distinguish them. Above 95% is strong evidence that A is better.**

#### Things to be honest about

- **Small samples produce wide intervals.** With 14 applications and 0 interviews, the tool will tell you the true rate is somewhere between 0% and ~22%. That's the actual answer — narrowing it down requires more data, not better math.
- **This tool can't tell you *why* a resume works.** It compares effectiveness across versions; it doesn't explain causation. A high rate could come from the resume, the timing, the role mix, or luck.
- **"True rate" is a model concept, not a physical constant.** The Bayesian "true rate" is the parameter of the Binomial distribution we assume generated your observations. It's the most defensible estimate given the data — but it's still an estimate, not a measurement.

#### Customising the prior (advanced — skip if it doesn't matter to you)

The sidebar lets you change the prior:
- **Slider (1, 1)** = Flat / uniform prior. *"I have no prior belief about what conversion rates are typical."* This is the default.
- **Jeffreys Beta(0.5, 0.5)** = an objective reference prior with slight U-shape (pulls slightly toward 0 and 1). Often used by statisticians as a "non-informative" default.
- **Slider (1, 20)** or similar = a pessimistic prior. *"I believe conversion rates are typically low; the data must work hard to convince me otherwise."*

Changing the prior shifts the posterior somewhat, but with reasonable data the data dominates. The notebook (linked below) sweeps the whole prior class and shows the conclusion is robust across a wide range.

---

Full write-up: [GitHub repo](https://github.com/marco50608/bayesian-resume-analysis) · Medium post: link TBA after publication.
    """)

# -----------------------------------------------------------------------------
# 2. Sidebar: User Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# GDPR Art. 21 gives a right to object to processing based on legitimate
# interest, so the interface has to offer a way to exercise it.
if logging_enabled():
    st.sidebar.checkbox(
        "Don't log this analysis",
        value=False,
        key="opt_out_logging",
        help="Skips the anonymous numeric logging described in the footer. "
             "Your inputs stay in your browser.",
    )

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

st.session_state.setdefault("ns", _qp_int("ns", 2, 1, 5))
num_strategies = st.number_input(
    "How many versions to compare?",
    min_value=1,
    max_value=5,
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

    # Seed defaults through session_state instead of `value=`, so the
    # "Load example data" button can overwrite them without Streamlit warning
    # that the widget has both a default and a Session State value.
    st.session_state.setdefault(f"name_{i}", _label_default)
    st.session_state.setdefault(f"n_{i}",    _n_default)
    st.session_state.setdefault(f"k_{i}",    _k_default)
    st.session_state.setdefault(f"inv_{i}",  _inv_default)

    with col1:
        label = st.text_input("Name", key=f"name_{i}", max_chars=40)
        n_apps = st.number_input(
            "Total Apps", min_value=1, max_value=10000, key=f"n_{i}",
        )

    with col2:
        k_interviews = st.number_input(
            "Interviews", min_value=0, max_value=10000, key=f"k_{i}",
            help="Number of interviews/first-stage responses you actually received.",
        )
        n_invalid = st.number_input(
            "Noise (Invalid)", min_value=0, max_value=10000, key=f"inv_{i}",
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
    # 4. Overview — plain-language summary, shown before the detailed tabs
    # -----------------------------------------------------------------------------
    st.markdown("## 📋 Overview — what your numbers actually say")

    _n_total = sum(r["effective_n"] for r in results)
    _k_total = sum(r["data"]["k"] for r in results)
    _runner = sorted_results[1] if len(sorted_results) > 1 else None
    if _runner is not None:
        _p_vs_runner = float((winner["samples"] > _runner["samples"]).mean())

    _c1, _c2, _c3 = st.columns(3)
    _wl = winner["data"]["label"]
    _c1.metric("Best performing version",
               _wl if len(_wl) <= 22 else _wl[:21] + "…",
               help=_wl if len(_wl) > 22 else None)
    if len(results) > 1:
        _c2.metric(
            "Chance it really is the best", f"{prob_best:.0%}",
            help="Probability that this version's TRUE rate is the highest of every "
                 "version you entered — not just that it looked best this time.",
        )
    else:
        _c2.metric("Chance it really is the best", "—",
                   help="Add a second version in the sidebar to enable comparison.")
    _c3.metric(
        "Evidence behind it", f"{_k_total} / {_n_total}",
        help="Total interviews / total valid applications across all versions.",
    )

    # --- Per-version readout, plain language --------------------------------
    st.markdown("#### Your versions, one line each")
    for r in sorted_results:
        s_ = r["data"]
        st.markdown(
            f"- **{s_['label']}** — {s_['k']} interview"
            f"{'' if s_['k'] == 1 else 's'} from {r['effective_n']} valid "
            f"application{'' if r['effective_n'] == 1 else 's'}. "
            f"Best guess **{r['mean']:.1%}**; the data is consistent with anything "
            f"from **{r['ci_lower']:.1%}** to **{r['ci_upper']:.1%}**."
        )

    # --- What it means, adapted to how decisive the result actually is ------
    st.markdown("#### What this means")

    _widest = max(r["ci_upper"] - r["ci_lower"] for r in results)

    if len(results) == 1:
        st.info(
            f"With one version there is nothing to compare against, so the useful "
            f"output is the range: your true rate is most likely between "
            f"**{winner['ci_lower']:.1%}** and **{winner['ci_upper']:.1%}**. "
            f"Add a second version in the sidebar to find out whether a change you "
            f"made actually helped."
        )
    elif prob_best >= 0.95:
        st.success(
            f"**The data can tell your versions apart.** "
            f"{winner['data']['label']} comes out on top in {prob_best:.0%} of the "
            f"plausible worlds consistent with what you observed. That is strong "
            f"enough to act on: keep using it, and stop spending applications on the "
            f"others to find out.\n\n"
            f"It is *not* a promise that its true rate is {winner['mean']:.1%}. "
            f"The honest claim is the range "
            f"[{winner['ci_lower']:.1%}, {winner['ci_upper']:.1%}] — the ranking is "
            f"solid, the exact number is not."
        )
    elif prob_best >= 0.80:
        st.warning(
            f"**Leaning, but not settled.** {winner['data']['label']} is ahead, and "
            f"it is the best of your versions in {prob_best:.0%} of plausible worlds "
            f"— which also means roughly a **{1 - prob_best:.0%} chance the ranking "
            f"is wrong** and something else is actually better.\n\n"
            f"That is usually worth acting on provisionally while you gather more "
            f"data, but it is not worth writing a blog post about yet."
        )
    else:
        st.error(
            f"**Your data cannot yet tell these versions apart.** "
            f"{winner['data']['label']} has the highest average, but only a "
            f"{prob_best:.0%} chance of genuinely being the best — which leaves a "
            f"**{1 - prob_best:.0%} chance one of the others is actually better**. "
            f"An ordering that shaky can reverse on a handful of applications.\n\n"
            f"The honest conclusion right now is *\"I don't know yet\"*. That is a "
            f"real finding, not a failure: it stops you from switching strategy on "
            f"noise. Send more applications with the current front-runners before "
            f"drawing a conclusion."
        )

    if _runner is not None and 0.05 < _p_vs_runner < 0.95:
        st.caption(
            f"⚖️ Head-to-head, **{winner['data']['label']}** beats "
            f"**{_runner['data']['label']}** in only {_p_vs_runner:.0%} of plausible "
            f"worlds. Those two in particular are not distinguishable yet."
        )

    if _widest > 0.35:
        st.caption(
            f"📏 Your widest range spans {_widest:.0%} percentage points. That is what "
            f"a small sample looks like — the width is the honest answer, and it "
            f"narrows with more applications, not with better maths."
        )

    # --- Where to look next -------------------------------------------------
    st.markdown("#### Where to look next")
    st.markdown(
        "Each question below is answered on one of the tabs directly underneath "
        "this section.\n\n"
        "| If you want to know… | Open this tab |\n"
        "|---|---|\n"
        "| How certain each rate is, and whether two versions overlap | **📈 Distributions (PDF)** |\n"
        "| Every version's range side by side, at a glance | **🌳 Forest Plot (Comparison)** |\n"
        "| The exact odds of any version beating any other | **📈 Distributions (PDF)** → *Pairwise Probability Matrix* |\n"
        "| How many more applications until you'd expect at least one interview | **⏳ Effort Survival (Simulation)** |\n"
        "| How many applications to send to hit a target number of offers | **🎯 Reverse Goal Calculator** |"
    )

    st.divider()

    # -----------------------------------------------------------------------------
    # 5. Visualizations (Tabs)
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
        #### 💡 How to read this chart

        Each curve is one strategy's **belief distribution** about its true conversion rate.
        Think of it as: *"if you had to bet on what the true rate is, this curve is your bet."*

        - **X-axis** — possible true conversion rates from 0% to whatever the chart shows.
        - **Y-axis** — how plausible each rate is. **A tall narrow peak = "I'm very sure the rate is around here." A wide flat curve = "I really don't know yet."**
        - **Width of the curve** = uncertainty. Small samples → wide curves. More data → narrower curves.

        **The hover info is the most useful number.** Hover over the chart at any x-value (e.g., 10%):
        - You'll see *"Chance > 10%: 80%"* — meaning **there's an 80% probability the true rate is above 10%**.
        - This is what Bayesian inference is good at: **direct probability statements**, not p-values.

        **Comparing two curves visually:**
        - If two curves barely overlap → those strategies are probably different.
        - If two curves overlap a lot → the data can't reliably tell them apart yet (you'd need more applications).
        """)
        # -----------------------------------------------------------------------------
        # Pairwise P(A > B) Matrix
        # -----------------------------------------------------------------------------
        if len(results) >= 2:
            st.markdown("### 🔀 Pairwise Probability Matrix")
            st.caption(
                "Read **row vs column**: each cell shows the posterior probability that the **row** strategy's "
                "true rate is higher than the **column** strategy's. So the cell at row V2, column V3 reading "
                "*99%* means *\"there's a 99% probability that V2's true conversion rate is higher than V3's.\"* "
                "**Near 50% = the data can't tell the two strategies apart.** "
                "**Above 95% = strong evidence the row strategy is better.** "
                "**Below 5% = strong evidence the column strategy is better.** "
                "Computed from 10,000 paired posterior samples per pair."
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
            "Each strategy is shown as a dot (best single guess of the true rate) with a horizontal bar "
            "(95% credible interval — the range that contains the true rate with 95% posterior probability). "
            "If two bars overlap a lot, the data can't reliably tell those strategies apart. "
            "If a bar is entirely to the right of another, that strategy is very likely better.")

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
                       range=[0, view_range_max]),
            yaxis=dict(title="Strategy"),
            showlegend=False,
            height=300 + (len(results) * 30),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # --- TAB 3: Effort Survival Analysis ---
    with tab3:
        st.subheader("Effort Simulation: how many applications until ≥1 interview?")
        st.caption(
            "Reads as: 'if I send N more applications using this strategy, what's the probability that at least one of them "
            "produces an interview?' The curve climbs from 0% (zero applications) toward 100% (lots of applications). "
            "Each strategy gets its own curve. A steeper curve means fewer applications needed for the same probability. "
            "Two dotted reference lines mark the 50% and 90% probability thresholds — read off the x-axis to see "
            "how many applications you'd need to cross each threshold.")

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

                y_offset = -30 - i * 25

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
        **How to read this**

        - **Where a curve crosses the 90% line** = how many applications you'd need to send for a 90% chance of at least one interview with that strategy. E.g. if a curve hits 90% at 20 applications, sending 20 more applications gives you a 90% chance of at least one interview.
        - **Steeper curve = better strategy** (fewer applications needed for the same confidence).
        - The "annotation arrow" labelled *"90% chance at X apps"* points at the crossing point automatically. If no curve reaches 90% within 50 applications, no annotation appears for that strategy — meaning you'd need more than 50 applications to hit 90% confidence.
        - The curve **marginalises over posterior uncertainty**: it averages the survival probability across all plausible true rates, weighted by how likely each rate is given your data. So a flat-looking, low-converging curve usually means *"the data is consistent with a low true rate, so even many applications can't guarantee an interview."*
        """)

    # --- TAB 4: Reverse Goal Calculator ---
    with tab4:
        st.subheader("🎯 Reverse Goal Calculator (Action Plan)")
        st.markdown(
            "**How many applications do I need to send to be reasonably sure I'll get my target offers?**\n\n"
            "Set three things on the sliders below:\n"
            "1. **Target number of offers** — how many job offers you want.\n"
            "2. **Interview-to-offer rate** — your estimate of how many interviews convert to actual offers "
            "(e.g. 20% = 1 offer per 5 interviews).\n"
            "3. **Desired probability of reaching the target** — how confident you want to be that this many "
            "applications will be enough (e.g. 80% = \"I want to be 80% sure I'll hit my target\").\n\n"
            "The calculator then returns the **smallest number of applications N such that the probability of "
            "reaching your target in N applications is at least your chosen confidence**.\n\n"
            "**Why this is more than just `target / rate`.** Naïvely, if your offer rate is 5%, you'd think "
            "1 / 0.05 = 20 applications give you 1 offer. But that 20 only gives you a ~64% chance of getting "
            "at least one offer — not the 100% the simple division suggests. This calculator avoids that trap "
            "by integrating **both** sources of randomness: (1) uncertainty about your true conversion rate "
            "(your real rate could be higher or lower than your point estimate), and (2) the binomial sampling "
            "noise of whether each application actually converts."
        )

        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            target_offers = st.number_input("Target Number of Offers", min_value=1, value=1, step=1)
        with col_input2:
            # Slider is integer 0-100 so the printf format "%d%%" displays
            # the value as a real percentage (e.g. 20%). The float-range
            # version 0.0-1.0 with "%.0f%%" formatting was a bug — it
            # printed 0.2 as "0%" and 0.8 as "1%" because printf has no
            # built-in "scale by 100" specifier.
            offer_rate_pct = st.slider(
                "Estimated Interview-to-Offer Rate", 0, 100, 20, 1, format="%d%%",
                help="If you get 5 interviews, how many offers do you expect? (20 = 1 offer per 5 interviews)"
            )
            offer_rate = offer_rate_pct / 100.0
        with col_input3:
            confidence_pct = st.slider(
                "Desired probability of reaching target", 1, 99, 80, 1, format="%d%%",
                help="Higher target probability ⇒ more applications recommended."
            )
            confidence = confidence_pct / 100.0

        st.divider()

        # Search grid bounded for performance. ~1000 apps × scipy.binom.cdf
        # (vectorised over 10k posterior draws per call) is well under a
        # second per strategy.
        MAX_APPS = 1000

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
            # P(reach goal) increases monotonically in N, so bisect instead of
            # evaluating all MAX_APPS grid points: ~10 evaluations, not 1000.
            def _p_reach(n_apps_):
                return float(np.mean(
                    1.0 - stats.binom.cdf(target_offers - 1, n_apps_, overall_samples)
                ))

            prob_at_max = _p_reach(MAX_APPS)
            if prob_at_max < confidence:
                apps_needed = None
                apps_needed_str = f">{MAX_APPS}"
            else:
                lo, hi = 1, MAX_APPS       # invariant: _p_reach(hi) >= confidence
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _p_reach(mid) >= confidence:
                        hi = mid
                    else:
                        lo = mid + 1
                apps_needed = lo
                apps_needed_str = str(apps_needed)

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
                    f"Even {MAX_APPS} applications give only a {prob_at_max:.1%} probability "
                    f"of reaching this goal with this strategy. Consider lowering the target, "
                    f"raising the offer-rate estimate, or accepting a lower target probability."
                )
            st.markdown("---")

    # -----------------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------------
    st.markdown("### 💾 Export")

    # CSV button + PNG generator share one row, each occupying 1/3 of the
    # page width. The third column stays empty.
    col_csv, col_png, _spacer = st.columns([1, 1, 1])

    with col_csv:
        csv_bytes = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download table (CSV)",
            data=csv_bytes,
            file_name="resume_conversion_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_png:
        # One fixed key holding WHICH fingerprint the user last generated for,
        # instead of a new session_state entry per input change.
        if st.session_state.get("png_ready_fp") != current_fp:
            if st.button(
                "🖼️ Download charts as PNG",
                key="gen_png",
                use_container_width=True,
                help="Generates 3 PNGs (PDF, Forest, Survival). Takes ~1–2 seconds.",
            ):
                st.session_state["png_ready_fp"] = current_fp
                st.rerun()
        else:
            try:
                with st.spinner("Rendering charts…"):
                    pdf_png_bytes = render_png(fig_pdf.to_json())
                    forest_png_bytes = render_png(fig_forest.to_json())
                    surv_png_bytes = render_png(fig_surv.to_json())

                st.download_button(
                    "📈 PDF chart",
                    data=pdf_png_bytes,
                    file_name="resume_conversion_pdf.png",
                    mime="image/png",
                    use_container_width=True,
                )
                st.download_button(
                    "🌳 Forest plot",
                    data=forest_png_bytes,
                    file_name="resume_conversion_forest.png",
                    mime="image/png",
                    use_container_width=True,
                )
                st.download_button(
                    "⏳ Survival chart",
                    data=surv_png_bytes,
                    file_name="resume_conversion_survival.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(
                    "PNG export requires `kaleido` — make sure `kaleido==0.2.1` is in requirements.txt. "
                    f"(Error: {type(e).__name__})"
                )

    # -----------------------------------------------------------------------------
    # Footer (moved to actual end — was incorrectly above the Export section)
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
