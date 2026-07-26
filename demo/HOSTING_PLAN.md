# Hosting plan — Viyog vs `pytorch-ood` OOD-vs-ADV leaderboard

**Goal.** A free, public, visually appealing, interactive page that shows how
**Viyog** compares to standard `pytorch-ood` detectors on the **OOD-vs-ADV**
separation task (and ID-vs-ADV, ID-vs-OOD) across **20 architectures** and
**3 datasets** (CIFAR-100, CIFAR-10, GTSRB).

**TL;DR recommendation — HuggingFace Spaces (Gradio SDK).** It is the ML-native
home for OOD/robustness leaderboards, free with no credit card, gives a stable
public URL (`https://huggingface.co/spaces/<user>/viyog-ood-adv`), needs zero
DevOps, and renders interactive Plotly charts + sortable tables out of the box.
Everything is **already scaffolded and smoke-tested** in this `demo/` folder —
deploying is a 3-command push. A pure-static **GitHub Pages** fallback is
described in §6 if you ever want to drop the Python runtime entirely.

---

## 1. Why this is cheap and easy: the data is tiny

The app never touches the multi-GB `data/`, `weights/`, or `results/` trees.
`export_leaderboard.py` distills the aggregated analysis CSVs into **~38 KB** of
tidy files (`demo/data/`):

| file | rows | what |
|---|---|---|
| `leaderboard.csv` | 30 | detector × dataset, ranked, with T1/T2/T3 + state-memory + compute cost |
| `permodel_t3.csv` | 270 | per-architecture T3 for every detector (20 models) |
| `ood_difficulty.csv` | 324 | T3 split by far / near / texture OOD |
| `meta.json` | — | model / detector / dataset inventory + provenance |

Because the payload is static and tiny, **any free tier is comfortably enough**
(no GPU, no model load, sub-second cold start).

---

## 2. Option comparison (all free)

| Option | URL / persistence | Interactivity | Effort | "ML-native" | Verdict |
|---|---|---|---|---|---|
| **HF Spaces — Gradio** ✅ | `*.hf.space`, permanent | Plotly + sortable tables, tabs, dropdowns | **Lowest** (push 4 files) | **Yes** (lives next to the models) | **Recommended** |
| HF Spaces — Streamlit | same | Great multi-tab dashboards | Low | Yes | Fine alternative; heavier runtime |
| Streamlit Community Cloud | `*.streamlit.app` | Same as Streamlit | Low–med (needs a public GitHub repo) | Neutral | Good if you prefer GitHub-driven deploys |
| GitHub Pages (static) | `*.github.io`, permanent | Plotly.js / ECharts from a JSON | Med (write HTML/JS) | No | Best **fallback**: zero runtime, fastest load |

**Why Gradio over Streamlit here:** the app is a leaderboard + a few linked
charts, not a long scrolling dashboard. Gradio's `Tab`/`Radio`/`Dataframe`/`Plot`
match that exactly, the theme looks polished with no CSS, and HF shows the
Space card (this folder's `README.md`) as the landing description.

---

## 3. What the app shows (already built — `app.py`)

Five tabs, all driven by a dataset selector:

1. **🏆 Leaderboard** — detectors ranked by **T3 (OOD-vs-ADV AUROC)**, with T1,
   T2, and persistent-state memory. Viyog rows green, logit blue, distance
   orange. Headline: *Viyog-D leads T3 (0.824) at 0.28 KB; logit detectors are
   blind to ADV (T2 ≈ 0.63 vs Viyog 0.966); distance detectors need 7–25 MB.*
2. **⚖️ Cost vs accuracy** — log-x scatter of state-memory vs T3. Viyog (green ★)
   sits **top-left** = cheap **and** accurate; the Pareto story in one glance.
3. **🧩 Per-architecture** — grouped bars of per-model T3, **Viyog-D vs any
   baseline you pick**, sorted, across all 20 backbones.
4. **🎯 OOD difficulty** — far / near / texture split; shows the TV read holds
   far/near and the HF complement rescues texture-OOD (mirrors paper Fig. 11).
5. **ℹ️ About** — plain-language definitions of T1/T2/T3, the mechanism, and the
   paper provenance.

Palette is the paper's Okabe-Ito colourblind-safe set, so the demo and the
manuscript figures look like one body of work.

---

## 4. Deploy steps (free, no credit card) — ~5 minutes

```bash
# 0. one-time: a free HF account + a write token (huggingface.co/settings/tokens)
pip install -U huggingface_hub

# 1. (re)build the data the app serves
cd /mnt/data1/asing725/viyog
VIYOG_ROOT=$PWD uv run --frozen --project Seperating_OOD_and_ADV \
    python demo/export_leaderboard.py

# 2. create the Space (sdk=gradio) and push the demo/ folder
huggingface-cli login                      # paste the write token
huggingface-cli repo create viyog-ood-adv --type space --space_sdk gradio -y
cd demo
git init && git remote add origin https://huggingface.co/spaces/<user>/viyog-ood-adv
git add app.py requirements.txt README.md data/ && git commit -m "Viyog OOD-vs-ADV leaderboard"
git push -u origin main                    # or master, per the repo default
```

The Space auto-builds from `requirements.txt` (pinned `gradio==5.50.0`, smoke-
tested) and goes live at `https://huggingface.co/spaces/<user>/viyog-ood-adv`.
Alternative to the git push: drag-drop the four items in the Space's **Files**
web UI.

---

## 5. Maintenance

When results change, regenerate and re-push only the data:

```bash
VIYOG_ROOT=/path/to/viyog uv run --frozen --project Seperating_OOD_and_ADV \
    python demo/export_leaderboard.py
cd demo && git add data/ && git commit -m "refresh leaderboard data" && git push
```

No app changes needed — the UI reads whatever is in `data/`.

---

## 6. Fallback — GitHub Pages (zero runtime, static)

If you ever want a no-Python, instant-load public page:

1. `export_leaderboard.py` already emits CSVs; add a 10-line `--json` path (or
   `pandas.to_json`) to also write `data/leaderboard.json`.
2. A single `index.html` with **Plotly.js** (CDN `<script>`) `fetch()`es the
   JSON and renders the same four charts client-side.
3. Push to a `gh-pages` branch (or `/docs`); enable Pages in repo settings.
   Live at `https://<user>.github.io/viyog`.

Trade-off: more hand-written HTML/JS, no server-side filtering, but the fastest
possible load and nothing to keep running.

---

## 7. Optional Phase 2 — a live "try an image" tab

A stretch goal that still fits a free **CPU** Space: ship one small checkpoint
(e.g. ResNet-18, ~45 MB) and the `viyog` scorer; let a user upload an image, run
one forward pass, and display the dormant-band roughness `V(x)` with its
OOD/ADV routing decision. CPU-feasible for a single small model. Keep it behind
its own tab so the leaderboard stays instant. Defer until the static leaderboard
is live.

---

### Status

- [x] Data export written and **run on real results** (`demo/data/`, 38 KB).
- [x] Gradio app written and **headless smoke-tested** on gradio 5.50.0 (all 5
      views, all 3 datasets).
- [x] `requirements.txt` pinned to the tested version; Space card `README.md`
      with HF metadata header ready.
- [ ] **You:** create the Space and push (needs your free HF account/token).
