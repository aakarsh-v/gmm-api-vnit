const API_BASE = 'https://gmm-api-vnit.onrender.com';

const PROPERTIES = [
  { key: 'UTS (MPa)', label: 'UTS (MPa)', default: '550' },
  { key: 'YS (MPa)', label: 'YS (MPa)', default: '400' },
  { key: 'Fatigue Strength (MPa)', label: 'Fatigue Strength (MPa)', default: '' },
  { key: 'Shear Strength (MPa)', label: 'Shear Strength (MPa)', default: '' },
  { key: 'Y (GPa)', label: 'Y (GPa)', default: '' },
  { key: 'G (GPa)', label: 'G (GPa)', default: '' },
  { key: 'Density (g/cc)', label: 'Density (g/cc)', default: '' },
  { key: 'Cp (J/kg-K)', label: 'Cp (J/kg-K)', default: '' },
  { key: 'TC (W/m-K)', label: 'TC (W/m-K)', default: '' },
  { key: 'TE Coeff', label: 'TE Coeff', default: '' },
  { key: 'Thermal Diffusivity ', label: 'Thermal Diffusivity', default: '' },
  { key: 'EC Volume (% IACS)', label: 'EC Volume (% IACS)', default: '' },
  { key: 'EC Weight (% IACS)', label: 'EC Weight (% IACS)', default: '' },
];

const ELEMENTS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti'];

function initForm() {
  const grid = document.getElementById('property-inputs');
  PROPERTIES.forEach((prop) => {
    const div = document.createElement('div');
    div.className = 'property-field';
    const id = `prop-${prop.key.replace(/\W+/g, '-')}`;
    div.innerHTML = `
      <label for="${id}">${prop.label}</label>
      <input type="number" step="any" id="${id}" name="${prop.key}" 
        data-key="${prop.key}" placeholder="—" value="${prop.default}" />
    `;
    grid.appendChild(div);
  });
}

function collectTargets() {
  const targets = {};
  document.querySelectorAll('#property-inputs input').forEach((input) => {
    const val = input.value.trim();
    if (val !== '') {
      const num = parseFloat(val);
      if (!Number.isNaN(num)) {
        targets[input.dataset.key] = num;
      }
    }
  });
  return targets;
}

async function checkHealth() {
  const el = document.getElementById('api-status');
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.backward_pool_loaded) {
      el.textContent = `API ready · ${data.pool_rows?.toLocaleString() ?? '?'} alloys in search pool`;
      el.className = 'api-status ok';
    } else {
      el.textContent = 'API running but search pool not loaded. Run notebook 06 to generate synthetic_wrought.csv.';
      el.className = 'api-status warn';
    }
  } catch (err) {
    el.textContent = 'Cannot reach API. Start server: uvicorn api.main:app --reload --port 8000';
    el.className = 'api-status error';
  }
}

async function searchAlloys(targets, topK) {
  const res = await fetch(`${API_BASE}/api/backward/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ targets, top_k: topK }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText || 'Request failed';
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

function renderComposition(composition) {
  const rows = ELEMENTS.filter((el) => (composition[el] ?? 0) > 0.05)
    .map(
      (el) =>
        `<tr><td>${el}</td><td>${Number(composition[el]).toFixed(2)}%</td></tr>`
    )
    .join('');
  return rows
    ? `<table><thead><tr><th>Element</th><th>wt%</th></tr></thead><tbody>${rows}</tbody></table>`
    : '<p class="empty-state">No composition data</p>';
}

function renderProperties(properties, targetKeys) {
  const keys = targetKeys.length ? targetKeys : Object.keys(properties);
  const rows = keys
    .map(
      (k) =>
        `<tr><td>${k}</td><td>${properties[k] != null ? Number(properties[k]).toFixed(2) : '—'}</td></tr>`
    )
    .join('');
  return `<table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderResults(data) {
  const container = document.getElementById('results');
  const targetKeys = Object.keys(data.targets || {});

  if (!data.candidates || data.candidates.length === 0) {
    container.innerHTML = '<p class="empty-state">No candidates returned.</p>';
    return;
  }

  container.innerHTML = data.candidates
    .map((c, i) => {
      const rank = i + 1;
      return `
        <article class="candidate-card">
          <h3>Candidate #${rank}</h3>
          <p class="error-score">Total error: ${Number(c.total_error).toFixed(4)} (lower is better)</p>
          <p class="recipe">${escapeHtml(c.recipe)}</p>
          <h4 style="margin:0.5rem 0 0.25rem;font-size:0.85rem;color:var(--muted)">Composition</h4>
          ${renderComposition(c.composition)}
          <h4 style="margin:0.75rem 0 0.25rem;font-size:0.85rem;color:var(--muted)">Matched properties</h4>
          ${renderProperties(c.properties, targetKeys)}
        </article>
      `;
    })
    .join('');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

document.getElementById('search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const targets = collectTargets();
  const resultsEl = document.getElementById('results');
  const btn = document.getElementById('submit-btn');

  if (Object.keys(targets).length === 0) {
    resultsEl.innerHTML =
      '<p class="error-msg">Enter at least one target property value.</p>';
    return;
  }

  const topK = parseInt(document.getElementById('top-k').value, 10) || 3;
  btn.disabled = true;
  resultsEl.innerHTML = '<p class="loading">Searching alloy pool…</p>';

  try {
    const data = await searchAlloys(targets, topK);
    renderResults(data);
  } catch (err) {
    resultsEl.innerHTML = `<p class="error-msg">${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
});

initForm();
checkHealth();
