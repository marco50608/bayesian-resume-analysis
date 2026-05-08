# Bayesian Resume Conversion Analysis

A Bayesian analysis of a personal resume A/B "experiment": 58 applications across three CV versions in the German job market, Oct–Dec 2025. Uses a Beta-Binomial conjugate model to quantify uncertainty in conversion rates on a small sample, and to compare strategies via posterior simulation instead of frequentist tests.

This is a personal case study (N = 49 valid applications after excluding German-required roles), not a generalisable result. The point of the repo is the method, not the headline number.

**Companion write-up:** Medium post — link TBA after publication.
**Interactive tool:** Streamlit app — link TBA after deployment. Drop in your own application counts and see the posteriors for your strategies.

---

## TL;DR — What's in here

| Version | Description | Valid N | Interviews | Raw rate |
|---|---|---|---|---|
| V1 | Standard Taiwan/US English CV | 21 | 0 | 0% |
| V2 | English content, German UX (*Lebenslauf*, *Berufserfahrung*, 1.0–5.0 GPA) | 14 | 5 | 36% |
| V3 | "Traditional" English student format | 14 | 0 | 0% |

Under a flat Beta(1,1) prior, P(V2 > V3) ≈ 99%, 95% ETI for V2 ≈ [16%, 62%]. The effect survives a deliberately pessimistic Beta(1,50) prior, which is the main stress test in the notebook.

---

## Methodology, in one paragraph

Each strategy is modelled as Binomial(n, θ) with a Beta prior, giving a closed-form Beta(α+k, β+n−k) posterior. Comparison between strategies is done by drawing 100,000 samples from each posterior and computing pairwise P(θ_A > θ_B) and the expected uplift distribution. Application-count forecasts in the Streamlit app integrate over the full posterior rather than using a point estimate, so prediction intervals reflect parameter uncertainty rather than just sampling noise at a fixed rate. A sensitivity analysis (Part 9) repeats the comparison under a deliberately pessimistic Beta(1, 50) prior, and a separate test (Part 10, Test 1B) uses a Jeffreys reference prior to compare V2 against a pooled V1+V3 baseline — both confirm the result isn't an artefact of the flat-prior choice. Two further robustness checks (Parts 13–14) sweep the full Beta(α, β) prior class and run a power-scaling sensitivity diagnostic, both confirming the directional conclusion is structurally stable.

---

## What's *not* in here (and why)

- **No frequentist hypothesis test as the headline.** With N = 14 per arm and two zero-success arms, a p-value is either meaningless or misleading depending on which exact test you pick. The Bayesian framing answers the question I actually care about — "how much should I update my belief?" — directly.
- **No fitted "alpha curve" / difficulty-scoring model.** The Medium post's Part 11 ("Alpha Analysis") uses finance vocabulary as a framing device — the 2% / 10% / decay-curve constants are illustrative anchors, not parameters fitted to data. The notebook flags this explicitly. None of the headline conclusions depend on those numbers.
- **No attempt to control for JD fit, company size, or posting timing.** I don't have enough data, and pretending I do would be worse than admitting I don't.
- **No applicant PII.** No names, emails, addresses, or full résumé text appear in the repo. Companies that interviewed me are named in Part 7's timeline plot, consistent with the Medium post.

---

## Repo layout

```
.
├── README.md                          # you are here
├── LICENSE                            # MIT
├── .gitignore
├── requirements.txt                   # pinned dependencies
├── app.py                             # Streamlit interactive tool
├── Resume Conversion.ipynb            # full Bayesian analysis, end-to-end
└── .streamlit/
    └── secrets.toml.example           # template for optional anonymous logging
```

---

## Reproducing the analysis

```bash
git clone https://github.com/marco50608/bayesian-resume-analysis.git
cd bayesian-resume-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab "Resume Conversion.ipynb"
```

All stochastic steps (Monte Carlo posterior comparison, posterior-predictive noodle draws, sensitivity analysis) are seeded with `np.random.seed(222)` and `np.random.default_rng(222)`, so the figures should reproduce bit-for-bit.

### Running the Streamlit app

```bash
streamlit run app.py
```

The app runs without `secrets.toml`. Anonymous event logging to a Google Sheet is optional and silently disabled when the secrets file is absent — see `.streamlit/secrets.toml.example` for the schema if you want to wire it up.

---

## Key files to inspect first

- `Resume Conversion.ipynb` — Parts 0–14. Start here.
  - **Part 2:** Beta-Binomial conjugate update, per-strategy posteriors
  - **Part 3:** Posterior-predictive ("noodle") plot — Beta-Binomial 95% prediction interval, not just rate × n
  - **Part 5:** Inference — pairwise P(V2 > V3) and 95% credible interval for V2
  - **Part 6:** Forest plot and posterior difference plot (V2 − V3)
  - **Part 9:** Sensitivity analysis under a deliberately pessimistic Beta(1, 50) prior
  - **Part 10:** Statistical validation — frequentist binomial test (Test 1A) and Bayesian comparison against the pooled V1+V3 baseline under a Jeffreys prior (Test 1B)
  - **Part 11:** Risk-adjusted "alpha" framing — explicitly labelled as narrative, not statistical inference
  - **Part 13:** Robustness frontier — sweeps a 30×31 grid of (α, β) priors and plots contours of P(V2 > V3), so the conclusion's robustness is shown across the whole prior class instead of one alternative point
  - **Part 14:** Power-scaling sensitivity diagnostic (Kallioinen et al. 2024 style, closed-form Beta-Binomial adaptation) — quantifies how much each posterior depends on the prior versus the likelihood
- `app.py` — Streamlit version of the same model with arbitrary user-supplied data; adds an effort-survival simulation and a reverse-goal calculator that both marginalise over the posterior.

---

## Known limitations

1. **N = 49 is small.** The credible intervals are wide on purpose — that's the honest answer, not a flaw in the method.
2. **The sample is one person.** No claim that "German UX" generalises. The hypothesis is stated as a hypothesis in the write-up, not as a finding.
3. **No causal identification.** The three versions weren't randomised across a fixed job pool; they were applied sequentially. Time-of-posting, JD drift, and personal learning effects are all confounds I can't separate.
4. **The cognitive-load story is a hypothesis.** I find it the most plausible explanation, but the data doesn't prove mechanism.

---

## License

MIT. Do whatever you want with the code; attribution appreciated but not required.

## Contact

Art — MSc Applied Data Science, Frankfurt School
If you want to poke holes in the methodology, please do — issues and PRs welcome.