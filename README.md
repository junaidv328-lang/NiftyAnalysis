# Nifty 50 Breadth Dashboard

A Streamlit app that pulls OHLC for the top-20 weighted Nifty 50
constituents via Angel One SmartAPI, runs Bulkowski pattern detection
on each, and aggregates the results into a **weighted directional
breadth score**.

## What this is — and what it isn't

This is a **breadth indicator**, not a Nifty forecast.

- It tells you what % of the index (by weight) is currently in
  confirmed bullish vs bearish Bulkowski patterns.
- Most useful as a **confirmation / divergence signal** alongside
  your existing gamma / GEX / max-pain workflow.
- Bulkowski's stats are calibrated for **daily** charts. On
  4h/weekly/monthly the *direction* is still informative; the
  probability numbers are approximations — the UI flags this.
- The "implied Nifty range" is a **mechanical projection** of weighted
  pattern targets, not a forecast. Real hit rates are well below 100%.

---

## Changes in this build (review fixes)

This version corrects bugs found in the previous one. Worth knowing
because two of them silently affected output:

1. **Implied-range feature was dead.** `app.py` read
   `fc.get("move_to_t1")`; the engine emits `move_to_t1_pct`. The key
   never matched, so `implied_pct_move` was always `None` and the
   implied-range card could never appear. **Fixed.**

2. **Direction-agreement flaw (found by the self-test).** A pattern
   that has already blown past its measure-rule target reports a
   *negative* `move_to_t1_pct` while still BULLISH. The old code fed
   that contradictory number in, where it was silently discarded.
   Selection now requires the projected move to agree with the stock
   verdict and picks the nearest still-ahead target. **Fixed.**

3. **TLS verification was disabled** (`ssl.CERT_NONE`) on every Angel
   One call, including the one carrying your JWT. Now **strict by
   default**; only disabled if you explicitly set
   `ANGEL_INSECURE_SSL=1` (local-only escape hatch).

4. **Two modules were missing** from the previous upload
   (`nifty_weights.py`, `cred_store.py` — the filenames were swapped
   with other modules' contents). Both reconstructed; the app now
   actually imports.

5. **Hardening:** constant-time password compare
   (`hmac.compare_digest`); a loud **partial-coverage warning** so a
   half-fetched run can't masquerade as a full-index reading; an
   engine-contract assertion that raises loudly instead of silently
   scoring zero if the engine's dict keys ever drift.

Run `python selftest.py` to confirm the pipeline produces real,
non-zero output end to end. Delete `selftest.py` before deploy if you
prefer (it is not imported by the app).

---

## Architecture

```
nifty_breadth/
├── app.py                       # Streamlit UI
├── selftest.py                  # pipeline correctness harness (optional)
├── core/
│   ├── __init__.py
│   ├── bulkowski_engine.py      # pattern detection + forecast
│   ├── aggregator.py            # weighted score + breadth aggregation
│   ├── angel_client.py          # SmartAPI login + OHLC fetch
│   ├── nifty_weights.py         # top-20 weighted constituents
│   └── cred_store.py            # OS-keyring credential store
├── requirements.txt
├── runtime.txt
├── .gitignore
└── .streamlit/
    ├── config.toml              # theme
    └── secrets.toml.example     # copy to secrets.toml locally
```

---

## Local development

```bash
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml — set app_password (and optionally
# pre-fill Angel One credentials)

streamlit run app.py
```

Optional: run the correctness harness first.

```bash
python selftest.py        # expect "ALL CHECKS PASSED"
```

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo. **No secrets are in the repo**
   (`.gitignore` excludes `.streamlit/secrets.toml`).
2. <https://share.streamlit.io> → **New app** → point at the repo,
   main file `app.py`.
3. **App → Settings → Secrets**, paste:

   ```toml
   app_password = "your-strong-password-here"

   # optional pre-fill
   angel_api_key   = "..."
   angel_client_id = "..."
   angel_password  = "..."
   angel_totp_key  = "..."
   ```

4. Deploy. Share the URL only with people who have the password.

Note: the OS keyring "Remember me" works only on a local machine. On
Streamlit Cloud there is no keyring backend — the app detects this and
falls back to secrets / manual entry automatically.

---

## Updating Nifty weights (quarterly)

NSE rebalances semi-annually and weights drift daily with prices. The
values in `core/nifty_weights.py` are static, representative of
mid-2026, and adequate for a *breadth* read but not live.

1. Fetch current weights from
   <https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-50>
2. Update `WEIGHTS` (top-20 by weight) in `core/nifty_weights.py`.
3. Bump `WEIGHTS_LAST_UPDATED`.
4. If you change which symbols are used, also update `ANGEL_TOKENS`
   in `core/angel_client.py` so tokens stay in sync. The import-graph
   check in the README's self-test will flag any symbol missing a
   token.

## Security notes

- Code is visible if the repo is public — fine, no secrets in code.
- API keys live in Streamlit's encrypted secrets store, not the repo.
- The app is password-gated (constant-time compare).
- TLS verification is strict by default.
- If the password leaks, change `app_password` in the secrets manager
  — instant rotation, no redeploy.

## License / Author

For Junaid · VAR Fisheries / FishyBiz Research Tools.
