// Backend API discovery:
// 1. If running locally on FastAPI dev server (port 8000), use same-origin relative path ''
// 2. If hosted directly on Render, use same-origin relative path ''
// 3. For any other frontend host (Vercel, GitHub Pages, Netlify, VS Code Live Server, file://),
//    route requests to the deployed Render backend:
const RENDER_BACKEND_URL = 'https://gmm-api-vnit.onrender.com';

function determineApiBase() {
  const host = window.location.hostname || '';
  const port = window.location.port || '';

  if ((host === 'localhost' || host === '127.0.0.1') && port === '8000') {
    return '';
  }
  if (host.includes('onrender.com')) {
    return '';
  }
  return RENDER_BACKEND_URL;
}

let API_BASE = determineApiBase();

const PROPERTIES = [
  { key: 'UTS (MPa)', label: 'UTS (MPa)', range: '100 - 570', placeholder: 'e.g. 450' },
  { key: 'YS (MPa)', label: 'YS (MPa)', range: '70 - 480', placeholder: 'e.g. 350' },
  { key: 'Fatigue Strength (MPa)', label: 'Fatigue Strength (MPa)', range: '35 - 180', placeholder: 'e.g. 120' },
  { key: 'Shear Strength (MPa)', label: 'Shear Strength (MPa)', range: '55 - 325', placeholder: 'e.g. 200' },
  { key: 'Y (GPa)', label: 'Y (GPa)', range: '67 - 72', placeholder: 'e.g. 69' },
  { key: 'G (GPa)', label: 'G (GPa)', range: '25 - 27', placeholder: 'e.g. 26' },
  { key: 'Density (g/cc)', label: 'Density (g/cc)', range: '2.68 - 3.10', placeholder: 'e.g. 2.75' },
  { key: 'Cp (J/kg-K)', label: 'Cp (J/kg-K)', range: '850 - 910', placeholder: 'e.g. 890' },
  { key: 'TC (W/m-K)', label: 'TC (W/m-K)', range: '115 - 230', placeholder: 'e.g. 170' },
  { key: 'TE Coeff', label: 'TE Coeff', range: '22.0 - 24.1', placeholder: 'e.g. 23.5' },
  { key: 'Thermal Diffusivity ', label: 'Thermal Diffusivity', range: '45 - 95', placeholder: 'e.g. 65' },
  { key: 'EC Volume (% IACS)', label: 'EC Volume (% IACS)', range: '21 - 62', placeholder: 'e.g. 45' },
  { key: 'EC Weight (% IACS)', label: 'EC Weight (% IACS)', range: '75 - 205', placeholder: 'e.g. 140' },
];

const PRESETS = [
  {
    name: 'High Strength (7xxx)',
    targets: { 'UTS (MPa)': 540, 'YS (MPa)': 460 },
  },
  {
    name: 'Medium / Ductile (6xxx)',
    targets: { 'UTS (MPa)': 260, 'YS (MPa)': 190 },
  },
  {
    name: 'High Thermal Conductivity',
    targets: { 'TC (W/m-K)': 210, 'Thermal Diffusivity ': 85 },
  },
  {
    name: 'Electrical Conductor',
    targets: { 'EC Volume (% IACS)': 55, 'EC Weight (% IACS)': 180 },
  },
  {
    name: 'Lightweight Alloy',
    targets: { 'Density (g/cc)': 2.70, 'UTS (MPa)': 250 },
  },
];

const ELEMENTS = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti'];

function initForm() {
  const grid = document.getElementById('property-inputs');
  grid.innerHTML = '';

  PROPERTIES.forEach((prop) => {
    const div = document.createElement('div');
    div.className = 'property-field';
    const id = `prop-${prop.key.replace(/\W+/g, '-')}`;
    div.innerHTML = `
      <label for="${id}">
        <span>${escapeHtml(prop.label)}</span>
        <span class="prop-hint">${prop.range}</span>
      </label>
      <input type="number" step="any" id="${id}" name="${prop.key}" 
        data-key="${prop.key}" placeholder="${prop.placeholder}" value="" />
    `;
    grid.appendChild(div);
  });

  // Attach input listener to update styling and active property count
  grid.querySelectorAll('input').forEach((input) => {
    input.addEventListener('input', () => {
      input.classList.toggle('has-value', input.value.trim() !== '');
      updateActiveCount();
    });
  });

  initPresets();
}

function initPresets() {
  const container = document.getElementById('preset-buttons');
  if (!container) return;
  container.innerHTML = '';

  PRESETS.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn';
    btn.textContent = preset.name;
    btn.addEventListener('click', () => applyPreset(preset.targets));
    container.appendChild(btn);
  });
}

function applyPreset(presetTargets) {
  clearInputs();
  Object.entries(presetTargets).forEach(([key, val]) => {
    const input = document.querySelector(`#property-inputs input[data-key="${CSS.escape(key)}"]`);
    if (input) {
      input.value = val;
      input.classList.add('has-value');
    }
  });
  updateActiveCount();
  executeSearch();
}

function clearInputs() {
  document.querySelectorAll('#property-inputs input').forEach((input) => {
    input.value = '';
    input.classList.remove('has-value');
  });
  updateActiveCount();
  const summaryEl = document.getElementById('results-summary');
  if (summaryEl) summaryEl.textContent = '';
}

function updateActiveCount() {
  const targets = collectTargets();
  const count = Object.keys(targets).length;
  const countEl = document.getElementById('active-count');
  if (countEl) {
    countEl.textContent = `${count} ${count === 1 ? 'property' : 'properties'} active`;
  }
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

let healthCheckAttempt = 0;

async function checkHealth() {
  const el = document.getElementById('api-status');
  if (!el) return;

  el.textContent = 'Connecting to backend…';
  el.className = 'api-status warn';

  let res, data;
  let usedBase = API_BASE;

  try {
    res = await fetch(`${usedBase}/health`);
    data = await res.json();
  } catch (err1) {
    // If initial fetch failed and we weren't targeting the Render URL, fallback to Render URL
    if (usedBase !== RENDER_BACKEND_URL) {
      try {
        usedBase = RENDER_BACKEND_URL;
        API_BASE = RENDER_BACKEND_URL;
        res = await fetch(`${usedBase}/health`);
        data = await res.json();
      } catch (err2) {
        handleHealthError(el);
        return;
      }
    } else {
      handleHealthError(el);
      return;
    }
  }

  if (data && data.backward_pool_loaded) {
    const label = usedBase === '' ? 'Backend API' : 'Remote API';
    el.textContent = `${label} ready · ${data.pool_rows?.toLocaleString() ?? '?'} alloys loaded`;
    el.className = 'api-status ok';
  } else {
    el.textContent = 'API connected · Loading search pool…';
    el.className = 'api-status warn';
    setTimeout(checkHealth, 3000);
  }
}

function handleHealthError(el) {
  healthCheckAttempt += 1;
  el.textContent = 'Waking up backend server… (Render free tier may take ~30s on cold start)';
  el.className = 'api-status warn';
  if (healthCheckAttempt < 10) {
    setTimeout(checkHealth, 4000);
  } else {
    el.textContent = 'Backend offline or unreachable. Please try again later.';
    el.className = 'api-status error';
  }
}

async function searchAlloys(targets, topK) {
  let url = `${API_BASE}/api/backward/search`;
  let res;

  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targets, top_k: topK }),
    });
  } catch (networkErr) {
    // If local/relative failed, attempt fallback to Render URL
    if (API_BASE !== RENDER_BACKEND_URL) {
      API_BASE = RENDER_BACKEND_URL;
      url = `${API_BASE}/api/backward/search`;
      res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targets, top_k: topK }),
      });
    } else {
      throw networkErr;
    }
  }

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
        `<tr><td><strong>${el}</strong></td><td>${Number(composition[el]).toFixed(2)}%</td></tr>`
    )
    .join('');
  return rows
    ? `<table><thead><tr><th>Element</th><th>Concentration (wt%)</th></tr></thead><tbody>${rows}</tbody></table>`
    : '<p class="empty-state">No composition data</p>';
}

function renderMatchedProperties(candidateProps, targets) {
  const targetKeys = Object.keys(targets);
  if (!targetKeys.length) return '';

  const rows = targetKeys
    .map((k) => {
      const targetVal = targets[k];
      const actualVal = candidateProps[k];
      if (actualVal == null) {
        return `<tr><td>${escapeHtml(k.trim())}</td><td>${targetVal}</td><td>—</td><td>—</td></tr>`;
      }
      const diff = actualVal - targetVal;
      const pctErr = targetVal !== 0 ? (diff / targetVal) * 100 : 0;
      const diffClass = Math.abs(pctErr) < 2 ? 'diff-match' : (diff > 0 ? 'diff-pos' : 'diff-neg');
      const diffSign = diff > 0 ? '+' : '';

      return `
        <tr>
          <td><strong>${escapeHtml(k.trim())}</strong></td>
          <td>${targetVal.toFixed(2)}</td>
          <td>${actualVal.toFixed(2)}</td>
          <td class="${diffClass}">${diffSign}${diff.toFixed(2)} (${diffSign}${pctErr.toFixed(1)}%)</td>
        </tr>
      `;
    })
    .join('');

  return `
    <table>
      <thead>
        <tr>
          <th>Target Property</th>
          <th>Requested</th>
          <th>Matched Value</th>
          <th>Deviation</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderOtherProperties(candidateProps, targets, candidateIndex) {
  const targetKeys = new Set(Object.keys(targets).map((k) => k.trim()));
  const otherKeys = Object.keys(candidateProps).filter(
    (k) => !targetKeys.has(k.trim())
  );

  if (otherKeys.length === 0) return '';

  const rows = otherKeys
    .map(
      (k) =>
        `<tr><td>${escapeHtml(k.trim())}</td><td>${candidateProps[k] != null ? Number(candidateProps[k]).toFixed(2) : '—'}</td></tr>`
    )
    .join('');

  const detailId = `other-props-${candidateIndex}`;
  return `
    <div class="other-properties-section">
      <button type="button" class="toggle-details-btn" onclick="toggleDetails('${detailId}')">
        Show other estimated properties (${otherKeys.length})
      </button>
      <div id="${detailId}" style="display: none; margin-top: 0.5rem;">
        <table>
          <thead>
            <tr><th>Property</th><th>Estimated Value</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  `;
}

window.toggleDetails = function(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const isHidden = el.style.display === 'none';
  el.style.display = isHidden ? 'block' : 'none';
  const btn = el.previousElementSibling;
  if (btn) {
    btn.textContent = isHidden ? 'Hide other estimated properties' : 'Show other estimated properties';
  }
};

function renderResults(data) {
  const container = document.getElementById('results');
  const summaryEl = document.getElementById('results-summary');
  const targets = data.targets || {};

  if (!data.candidates || data.candidates.length === 0) {
    container.innerHTML = '<p class="empty-state">No matching candidate alloys found for the given targets.</p>';
    if (summaryEl) summaryEl.textContent = '';
    return;
  }

  if (summaryEl) {
    summaryEl.textContent = `Found ${data.candidates.length} distinct candidate ${data.candidates.length === 1 ? 'alloy' : 'alloys'}`;
  }

  container.innerHTML = data.candidates
    .map((c, i) => {
      const rank = i + 1;
      return `
        <article class="candidate-card">
          <div class="candidate-header">
            <h3>Candidate #${rank}</h3>
            <span class="error-badge">Total Error: ${Number(c.total_error).toFixed(4)}</span>
          </div>
          <div class="recipe">${escapeHtml(c.recipe)}</div>
          
          <div class="section-title">Matched Target Properties</div>
          ${renderMatchedProperties(c.properties, targets)}

          <div class="section-title">Chemical Composition</div>
          ${renderComposition(c.composition)}

          ${renderOtherProperties(c.properties, targets, i)}
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

async function executeSearch() {
  const targets = collectTargets();
  const resultsEl = document.getElementById('results');
  const btn = document.getElementById('submit-btn');

  if (Object.keys(targets).length === 0) {
    resultsEl.innerHTML =
      '<p class="error-msg">Please enter at least one target property value or choose a Quick Preset above.</p>';
    return;
  }

  const topK = parseInt(document.getElementById('top-k').value, 10) || 3;
  if (btn) btn.disabled = true;
  resultsEl.innerHTML = '<p class="loading">Searching alloy pool for optimal compositions…</p>';

  try {
    const data = await searchAlloys(targets, topK);
    renderResults(data);
  } catch (err) {
    resultsEl.innerHTML = `<p class="error-msg">${escapeHtml(err.message)}</p>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

document.getElementById('search-form').addEventListener('submit', (e) => {
  e.preventDefault();
  executeSearch();
});

document.getElementById('clear-btn').addEventListener('click', () => {
  clearInputs();
  document.getElementById('results').innerHTML =
    '<p class="empty-state">Enter one or more target properties or choose a preset above, then click <strong>Find alloys</strong>.</p>';
});

initForm();
checkHealth();
