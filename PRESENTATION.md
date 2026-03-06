# Alloy Property Prediction — Models and Pipeline (Overview)

This document gives a non-technical overview of the project: what it does, which models are used, what goes in and what comes out, and how the steps fit together. The project supports two types of alloys — **wrought** and **cast** — each with its own data and the same pipeline pattern.

---

## Pipeline Architecture

The workflow has three main phases:

1. **Tune and save settings** — We automatically choose the best prediction model and its settings for each property (e.g. strength, conductivity) and save them to a central config file. We also tune a separate model that describes how real alloy compositions look, so we can generate new, realistic recipes later.
2. **Build a large table of synthetic alloys** — Using those saved settings, we create a big table (e.g. 50,000 rows) of hypothetical alloys: each row has a composition (recipe) and predicted properties. This table is the “search pool” for the next phase.
3. **Search by your targets** — You specify the properties you want (e.g. “I need UTS = 550 MPa and Yield = 400 MPa”). The system filters and sorts the pool and returns the top few candidate alloys that best match your targets.

There are also optional steps that let you type in a single composition and get predicted properties directly (no search).

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
                    │  Step 3–4: Generate synthetic alloy table                 │
                    │  • Sample 50k compositions (GMM)                         │
                    │  • Predict properties (saved models) → CSV files         │
                    └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 5–6: Backward search                                               │
    │  You set target properties → system returns top-k candidate alloys       │
    └─────────────────────────────────────────────────────────────────────────┘

    Optional: Step 7–8 — Give one composition → get predicted properties.
```

### Execution order

1. **Step 1 — Forward tuning:** Choose the best prediction model (and its settings) for each property, for both wrought and cast data. Results are saved to the config file.
2. **Step 2 — GMM tuning:** Tune the composition-space model (GMM) on real alloy compositions only. Results are saved to the config file.
3. **Step 3 — Generate synthetic wrought pool:** Build the 50k-row table for wrought alloys (composition + predicted properties). Writes `synthetic_wrought.csv`.
4. **Step 4 — Generate synthetic cast pool:** Same for cast alloys. Writes `synthetic_cast.csv`.
5. **Step 5 — Backward search (wrought):** Load the wrought pool; you set target properties; get top-k candidate wrought alloys.
6. **Step 6 — Backward search (cast):** Same using the cast pool.
7. **Step 7–8 (optional) — Forward prediction:** Load config and data; give a composition; get predicted properties for that composition.

---

## Models Used

### 1. XGBoost / Random Forest / Gradient Boosting (property prediction)

**What it is:** These are three different “tree-based” prediction models. Each one learns from past data: “when the composition looks like X, the property (e.g. strength) is usually Y.” The pipeline tries all three for every property and keeps the one that performs best. So for one property we might use XGBoost, for another Random Forest, and so on — chosen automatically per property.

**Input:** Alloy composition — the amounts of 12 elements (Al, Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ga, V, Ti), usually as percentages. Aluminum (Al) is the main base; the rest are alloying elements.

**Output:** A single predicted number for one property — e.g. UTS (Ultimate Tensile Strength) in MPa, or electrical conductivity in % IACS.

**Where it’s used:** Step 1 (tuning: which model and which settings per property); Step 3–4 (to predict properties for the 50k synthetic compositions); and optionally Step 7–8 (to predict properties for any composition you provide).

---

### 2. Gaussian Mixture Model — GMM (composition space)

**What it is:** A statistical model that captures how real alloy compositions are distributed — which combinations of the 12 elements are typical in your dataset. It does **not** predict properties. It is used only to **generate** new, realistic-looking compositions (recipes) that we then pass to the property-prediction models above.

**Input:** Many real alloy compositions — just the 12 element columns (no property columns). The model learns the patterns in these recipes.

**Output:** A tuned model that can “sample” new compositions — e.g. 50,000 — that look like plausible alloys. Those compositions are then fed into the prediction models to get properties.

**Where it’s used:** Step 2 (tuning: how many mixture components and covariance type); Step 3–4 (sampling the 50k compositions that go into the synthetic table).

---

### 3. Search / filter (backward step)

**What it is:** This is **not** a machine-learning model. We already have a large table (e.g. 50,000 rows) of synthetic alloys with their predicted properties. You tell the system what you want (e.g. “UTS = 550 MPa, Yield = 400 MPa”). The system filters and sorts this table by how close each row is to your targets and returns the top few candidate alloys — each with its recipe (composition) and predicted properties.

**Input:** The synthetic table (composition + properties) and your target property values (e.g. desired UTS, yield strength, conductivity).

**Output:** The top-k candidate alloys: for each candidate, the composition (recipe) and the predicted properties. You can then take these as suggestions for which alloys might meet your requirements.

**Where it’s used:** Step 5 (wrought pool) and Step 6 (cast pool).

---

## Inputs and Outputs — Summary Table

| Step | Name (short)              | Input                                          | Output                                                                 |
|------|---------------------------|------------------------------------------------|------------------------------------------------------------------------|
| 1    | Forward tuning            | Wrought/cast data (composition + properties)   | Best model type and settings per property → saved in config file      |
| 2    | GMM tuning                | Composition data only (12 elements)            | GMM settings → saved in config file                                   |
| 3–4  | Generate synthetic pool   | Config + real data                             | `synthetic_wrought.csv` / `synthetic_cast.csv` (e.g. 50k rows each)    |
| 5–6  | Backward search           | Synthetic CSV + your target properties         | Top-k candidate alloys (recipe + predicted properties)                 |
| 7–8  | Forward prediction (opt.) | Config + real data + new composition(s)       | Predicted properties for the given composition(s)                     |

---

## What You Can Do With This

- **Forward (prediction):** “I have a composition (recipe). What strength, conductivity, etc. will it have?” — The system uses the saved models to predict those properties.
- **Backward (search):** “I need certain properties (e.g. UTS and yield strength). Which alloy recipes best match those targets?” — The system searches the pre-built table and returns the top candidate alloys with their recipes and predicted values.

---

## Using This Document in Word

- You can **open this file in Microsoft Word** (File → Open, then select `PRESENTATION.md`) and then use “Save As” to save it as a `.docx` document.
- Alternatively, copy and paste the sections you need into a new Word document.
- If you have **Pandoc** installed, you can generate a Word file from the command line:  
  `pandoc PRESENTATION.md -o PRESENTATION.docx`

For technical details (e.g. run times, notebook names, file paths), see the project’s technical summary document.
