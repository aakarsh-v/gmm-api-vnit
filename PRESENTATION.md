# Alloy Property Prediction — Models and Pipeline (Overview)

This document gives a non-technical overview of the project: what it does, which models are used, what goes in and what comes out, and how the steps fit together. The pipeline is **wrought alloys only** (one dataset and one synthetic search pool).

---

## Pipeline Architecture

The workflow has three main phases:

1. **Tune and save settings** — We automatically choose the best prediction model and its settings for each property (e.g. strength, conductivity) and save them to a central config file. We also tune a separate model that describes how real alloy compositions look, so we can generate new, realistic recipes later.
2. **Build a large table of synthetic alloys** — Using those saved settings, we create a big table (e.g. 50,000 rows) of hypothetical wrought alloys: each row has a composition (recipe) and predicted properties. This table is the “search pool” for the next phase.
3. **Search by your targets** — You specify the properties you want (e.g. “I need UTS = 550 MPa and Yield = 400 MPa”). The system filters and sorts the pool and returns the top few candidate alloys that best match your targets.

There is also an optional step that lets you evaluate predictions on real wrought data or type in a composition and get predicted properties directly (no search).

### Flow diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  Central config file (saved “best settings”)             │
                    │  • Best prediction model per property                    │
                    │  • Composition-space model (GMM) settings                 │
                    └─────────────────────────────────────────────────────────┘
                                     ▲                    ▲
                     save            │                    │            save
    ┌────────────────────────────────┴────┐    ┌───────────┴────────────────────┐
    │  Step 1: Tune prediction models    │    │  Step 2: Tune composition model │
    │  (XGBoost / Random Forest / GB)    │    │  (GMM — for sampling recipes)   │
    └────────────────────────────────┬──┘    └───────────┬────────────────────┘
                                      │                   │
                                      ▼                   ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  Step 3: Generate synthetic wrought table                  │
                    │  • Sample 50k compositions (GMM)                          │
                    │  • Predict properties (saved models) → synthetic_wrought.csv │
                    └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 4: Backward search (wrought)                                        │
    │  You set target properties → system returns top-k candidate alloys        │
    └─────────────────────────────────────────────────────────────────────────┘

    Optional: Step 5 — Forward prediction notebook: evaluate or predict from composition.
```

### Execution order

1. **Step 1 — Forward tuning:** Choose the best prediction model (and its settings) for each property on wrought data. Results are saved to the config file.
2. **Step 2 — GMM tuning:** Tune the composition-space model (GMM) on real wrought compositions. Results are saved to the config file.
3. **Step 3 — Generate synthetic wrought pool:** Build the 50k-row table (composition + predicted properties). Writes `synthetic_wrought.csv`.
4. **Step 4 — Backward search:** Load that pool; you set target properties; get top-k candidate alloys.
5. **Step 5 (optional) — Forward prediction:** Load config and wrought data; train/evaluate or predict properties for compositions you provide.

---

## Models Used

### 1. XGBoost / Random Forest / Gradient Boosting (property prediction)

**What it is:** These are three different “tree-based” prediction models. Each one learns from past data: “when the composition looks like X, the property (e.g. strength) is usually Y.” The pipeline tries all three for every property and keeps the one that performs best.

**Input:** Alloy composition — the amounts of 12 elements (Al, Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ga, V, Ti), usually as percentages.

**Output:** A single predicted number for one property — e.g. UTS (Ultimate Tensile Strength) in MPa, or electrical conductivity in % IACS.

**Where it’s used:** Step 1 (tuning); Step 3 (predict properties for the 50k synthetic compositions); optional forward notebook.

---

### 2. Gaussian Mixture Model — GMM (composition space)

**What it is:** A statistical model that captures how real wrought compositions are distributed. It does **not** predict properties. It is used only to **generate** new, realistic-looking compositions that are then passed to the property-prediction models.

**Input:** Many real alloy compositions — the 12 element columns only.

**Output:** A tuned model that can sample new compositions (e.g. 50,000) that resemble real alloys.

**Where it’s used:** Step 2 (tuning); Step 3 (sampling compositions for the synthetic table).

---

### 3. Search / filter (backward step)

**What it is:** This is **not** a machine-learning model. We take the pre-built table of synthetic alloys and predicted properties, filter and sort by your target values, and return the top few candidate recipes.

**Input:** `synthetic_wrought.csv` and your target property values.

**Output:** Top-k candidate alloys: composition (recipe) and predicted properties.

**Where it’s used:** Step 4 (`05_backward_wrought`).

---

## Inputs and Outputs — Summary Table

| Step | Name (short)              | Input                                          | Output                                                                 |
|------|---------------------------|------------------------------------------------|------------------------------------------------------------------------|
| 1    | Forward tuning            | Wrought data (composition + properties)        | Best model type and settings per property → config file               |
| 2    | GMM tuning                | Wrought composition data (12 elements)         | GMM settings → config file                                            |
| 3    | Generate synthetic pool   | Config + wrought data                          | `synthetic_wrought.csv` (e.g. 50k rows)                                 |
| 4    | Backward search           | Synthetic CSV + your target properties         | Top-k candidate alloys (recipe + properties)                          |
| 5    | Forward prediction (opt.) | Config + wrought data + composition(s)          | Predicted or evaluated properties                                      |

---

## What You Can Do With This

- **Forward (prediction):** “I have a composition (recipe). What strength, conductivity, etc. will it have?”
- **Backward (search):** “I need certain properties (e.g. UTS and yield strength). Which alloy recipes in the synthetic pool best match those targets?”

---

## Using This Document in Word

- You can **open this file in Microsoft Word** (File → Open, then select `PRESENTATION.md`) and then use “Save As” to save it as a `.docx` document.
- Alternatively, copy and paste the sections you need into a new Word document.
- If you have **Pandoc** installed: `pandoc PRESENTATION.md -o PRESENTATION.docx`

For technical details (e.g. run times, notebook names), see **PROJECT_SUMMARY.md**.
