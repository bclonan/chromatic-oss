"use strict";

const VIEWS = [
  ["overview", "Overview", "Measured runs, representation behavior, and the evidence behind this reference implementation."],
  ["experiments", "Experiments", "Compare recorded runs. Every metric comes from the saved experiment artifacts."],
  ["tensor", "Tensor viewer", "Inspect numerical values, chromatic codes, and reconstruction error side by side."],
  ["color", "Color space", "Explore the color geometry of real encoded tensors. Display color and numerical meaning are separate properties."],
  ["codebook", "Codebook", "Inspect learned representatives and their observed token frequencies."],
  ["training", "Training", "Follow recorded loss curves and compare representations on the same task and seed."],
  ["benchmarks", "Benchmarks", "Read task metrics, storage budgets, and measured runtime together."],
  ["ablations", "Ablations", "Test codec choice and precision against conventional quantization with equal bit budgets."],
  ["architecture", "Architecture", "How tensors, codecs, operators, and PyTorch training connect in this implementation."],
  ["conclusions", "Research conclusions", "What the current measurements support, where they fall short, and what they do not establish."],
  ["cube", "Cube lab", "A full multidimensional color cube. Turn a layer, inspect its hidden dimensions, or decode its distributed spectral field."],
];
const ICONS = [
  '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  '<path d="M9 3h6M10 3v7L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3M8 14h8"/>',
  '<path d="M3 7l9-4 9 4v10l-9 4-9-4V7zm0 0l9 4 9-4M12 11v10"/>',
  '<circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18M6 6l12 12M6 18L18 6"/>',
  '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 3v18M11 7h5M11 11h5M11 15h5"/>',
  '<path d="M3 3v18h18M6 16l5-6 4 2 6-7"/>',
  '<path d="M5 21V11h4v10M11 21V3h4v18M17 21V7h4v14"/>',
  '<path d="M3 7h18M3 17h18M7 4v6M17 14v6"/><circle cx="7" cy="7" r="2"/><circle cx="17" cy="17" r="2"/>',
  '<rect x="8" y="3" width="8" height="5" rx="1"/><rect x="2" y="16" width="8" height="5" rx="1"/><rect x="14" y="16" width="8" height="5" rx="1"/><path d="M12 8v4M6 16v-4h12v4"/>',
  '<path d="M5 3h11l3 3v15H5V3zM8 8h8M8 12h8M8 16h5"/>',
  '<path d="M3 7l9-4 9 4v10l-9 4-9-4V7zm0 0l9 4 9-4M12 11v10M7.5 5l9 4M7.5 9l9-4M7.5 9v10M16.5 9v10M3 12l9 4 9-4"/>',
];
const COLORS = ["#7454db", "#2e9e87", "#dc9861", "#478bc5", "#c35c93", "#9da44b", "#ae7058", "#576789"];
const state = { data: null, view: "overview", selectedExperiment: null, sample: null, sampleIndex: 0, tokenIndex: 0, taskFilter: "all", repFilter: "all", query: "", colorMode: "rgb", bookIndex: 0, bookToken: 0, bookPage: 0, ablationDistribution: "uniform", ablationDomain: "linear", inputShape: "4, 4", inputValues: "[-1, -0.75, -0.5, -0.25, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]", inputCodec: "rgbd" };
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite = value => typeof value === "number" && Number.isFinite(value);
const fmt = value => finite(value) ? (Math.abs(value) >= 1000 ? value.toLocaleString("en-US", {maximumFractionDigits: 2}) : Math.abs(value) > 0 && Math.abs(value) < .0001 ? value.toExponential(3) : Number(value.toPrecision(5)).toString()) : value == null ? "Not recorded" : String(value);
const human = text => String((typeof text === "object" && text !== null ? text.name ?? JSON.stringify(text) : text) ?? "").replace(/_/g, " ");
const localTime = value => { const date=new Date(value); return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat("en-US",{timeZone:"America/New_York",month:"short",day:"numeric",year:"numeric",hour:"numeric",minute:"2-digit",timeZoneName:"short"}).format(date); };
const codecLabel = item => typeof item === "object" && item !== null ? item.name === "uniform" ? `Uniform ${item.bits}-bit` : human(item.name) : human(item);
const metricValue = (e, ...keys) => keys.map(k => e.metrics?.[k] ?? e.execution?.[k]).find(v => v != null);
const experiments = () => state.data?.experiments ?? [];
const samples = () => state.data?.tensor_samples ?? [];
const books = () => state.data?.codebooks ?? [];
const colorSymbol = rgb => Array.isArray(rgb) && rgb.length === 3 && rgb.every(v => Number.isInteger(v) && v >= 0 && v <= 255)
  ? `COLOR_${rgb.map(v => v.toString(16).padStart(2,"0").toUpperCase()).join("")}` : "RGB not recorded";
const currentExperiment = () => experiments().find(e => e.id === state.selectedExperiment) ?? experiments()[0];
const svgIcon = index => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[index]}</svg>`;
const card = (title, subtitle, body, tag = "") => `<section class="card"><div class="card-header"><div><h2>${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div>${tag}</div><div class="card-body">${body}</div></section>`;
const facts = entries => `<dl class="facts">${entries.map(([name, value]) => `<div><dt>${esc(name)}</dt><dd class="number">${esc(fmt(value))}</dd></div>`).join("")}</dl>`;
const chip = (text, cls = "") => `<span class="chip ${cls}">${esc(text)}</span>`;
const empty = (title, text) => `<div class="empty"><h2>${esc(title)}</h2><p>${esc(text)}</p><code class="code-command">python scripts/run_benchmarks.py</code></div>`;
const options = (items, selected) => items.map(([value, label]) => `<option value="${esc(value)}"${String(selected) === String(value) ? " selected" : ""}>${esc(label)}</option>`).join("");
const metricKey = experiment => Object.keys(experiment?.metrics ?? {}).find(k => /accuracy|test_mse|val_mse|perplexity|task_metric|mse/.test(k)) ?? Object.keys(experiment?.metrics ?? {})[0];

function table(headers, rows, caption = "", rowHeader = false) {
  return `<div class="table-wrapper"><table>${caption ? `<caption>${esc(caption)}</caption>` : ""}<thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((value, i) => i === 0 && rowHeader ? `<th scope="row">${value}</th>` : `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function genericTable(rows, caption) {
  if (!rows.length) return `<p class="hint">No measurements recorded for this section.</p>`;
  const keys = [...new Set(rows.flatMap(row => Object.keys(row).filter(k => typeof row[k] !== "object" && !["artifact_path", "curves"].includes(k))))];
  return table(keys.map(human), rows.map(row => keys.map(k => `<span class="${finite(row[k]) ? "number" : ""}">${esc(fmt(row[k]))}</span>`)), caption);
}

function chart(series, yLabel = "Loss", logarithmic = false, xLabel = "recorded epochs") {
  const valid = series.map(s => ({...s, points: s.points.filter(p => finite(p[0]) && finite(p[1]) && (!logarithmic || p[1] > 0))})).filter(s => s.points.length);
  if (!valid.length) return `<p class="hint">No curve measurements available.</p>`;
  const width = 620, height = 245, pad = {left: 55, right: 16, top: 12, bottom: 32};
  const points = valid.flatMap(s => s.points);
  let xMin = Math.min(...points.map(p => p[0])), xMax = Math.max(...points.map(p => p[0]));
  let yMin = Math.min(...points.map(p => logarithmic ? Math.log10(p[1]) : p[1])), yMax = Math.max(...points.map(p => logarithmic ? Math.log10(p[1]) : p[1]));
  if (xMin === xMax) xMax += 1;
  if (yMin === yMax) { yMin -= .1; yMax += .1; }
  const yMargin = (yMax - yMin) * .07; yMin -= yMargin; yMax += yMargin;
  if (!logarithmic && points.every(p => p[1] >= 0)) yMin = Math.max(0, yMin);
  const x = v => pad.left + (v - xMin) / (xMax - xMin) * (width - pad.left - pad.right);
  const y = v => height - pad.bottom - ((logarithmic ? Math.log10(v) : v) - yMin) / (yMax - yMin) * (height - pad.top - pad.bottom);
  let markup = `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img"><title>${esc(yLabel)} across ${esc(xLabel)}</title>`;
  for (let n = 0; n <= 4; n++) {
    const yt = yMin + (yMax - yMin) * n / 4, py = height - pad.bottom - n / 4 * (height - pad.top - pad.bottom);
    markup += `<line class="gridline" x1="${pad.left}" y1="${py}" x2="${width-pad.right}" y2="${py}"/><text x="${pad.left-9}" y="${py+3}" text-anchor="end">${esc(fmt(logarithmic ? 10 ** yt : yt))}</text>`;
    const xt = xMin + (xMax - xMin) * n / 4;
    markup += `<text x="${x(xt)}" y="${height-12}" text-anchor="middle">${esc(fmt(xt))}</text>`;
  }
  valid.forEach((s, i) => {
    const color = s.color ?? COLORS[i % COLORS.length];
    markup += `<path d="${s.points.map((p, j) => `${j ? "L" : "M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2.3"${s.dashed ? ' stroke-dasharray="5 4"' : ""}/>`;
    const p = s.points.at(-1); markup += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${color}"/>`;
  });
  markup += `</svg>`;
  const legend = `<div class="legend">${valid.map((s,i) => `<span><i style="background:${s.color ?? COLORS[i % COLORS.length]}"></i>${esc(s.name)}${s.dashed ? " · dashed" : ""}</span>`).join("")}</div>`;
  const curveRows = valid.flatMap(s => s.points.map(p => [esc(s.name), `<span class="number">${fmt(p[0])}</span>`, `<span class="number">${fmt(p[1])}</span>`]));
  return legend + markup + `<details><summary class="hint">View chart data</summary>${table(["Series", "Epoch", yLabel], curveRows, "Recorded chart values")}</details>`;
}

function experimentRows(list, full = false) {
  return list.map(e => {
    const key = metricKey(e);
    const row = [
      `<button class="table-link" data-experiment="${esc(e.id)}">${esc(human(e.task))}</button>`,
      chip(human(e.representation), e.representation === "fp32" ? "neutral" : ""),
      `<span class="number">${esc(e.seed)}</span>`,
      `<span class="number">${esc(fmt(e.metrics?.[key]))}</span><small class="hint"> ${esc(human(key))}</small>`,
      `<span class="number">${esc(fmt(metricValue(e, "training_seconds", "train_seconds")))}</span>`,
    ];
    if (full) row.push(`<span class="number">${esc(fmt(metricValue(e, "bytes_model", "parameter_bytes", "storage_bytes")))}</span>`, e.artifact_path ? `<a href="/api/artifact?path=${encodeURIComponent(e.artifact_path)}" download>Artifact</a>` : "Not recorded");
    return row;
  });
}

function summaryMetrics() {
  const list = experiments(), tasks = new Set(list.map(e => e.task));
  const reps = new Set(list.map(e => e.representation));
  const measures = state.data?.precision?.length ?? 0;
  const metrics = [["Recorded experiments", list.length, `${tasks.size} tasks with recorded outcomes`, 1], ["Representations", reps.size, "Conventional and chromatic runs", 3], ["Precision measurements", measures, "Measured codec reconstruction", 7], ["Inspectable tensors", samples().length, "Numerical values and encoded colors", 2]];
  return `<div class="metrics-grid">${metrics.map(([label,value,note,icon]) => `<div class="metric-card"><div class="metric-label">${esc(label)}<span class="metric-glyph">${svgIcon(icon)}</span></div><div class="metric-value">${value}</div><div class="metric-note">${esc(note)}</div></div>`).join("")}</div>`;
}

function overview() {
  if (!state.data) return card("Start with a measured run", "The dashboard reads results/latest.json", empty("No recorded experiments", "Run the benchmark command in your project terminal. Refresh results when it finishes. The live tensor inspector also works before a benchmark run."));
  const first = currentExperiment(), task = first?.task;
  const preferred = ["fp32", "chroma8", "int8", "rgb24", "fixed24", "fp16"];
  const peers = experiments().filter(e => e.task === task && e.seed === first?.seed && e.curves?.length).sort((a,b) => (preferred.includes(a.representation) ? preferred.indexOf(a.representation) : 99) - (preferred.includes(b.representation) ? preferred.indexOf(b.representation) : 99)).slice(0, 6);
  const curves = peers.map((e,i) => ({name: human(e.representation), color: COLORS[i], points: (e.curves ?? []).map(p => [p.epoch,p.val_loss])}));
  const environment = state.data.environment ?? {};
  const sample = samples().find(s => (s.codec?.name ?? s.codec) === "rgb") ?? samples()[0];
  const rgb = sample?.rgb?.slice(0, 128) ?? [];
  const palette = rgb.length ? `<div class="palette">${rgb.map((color,i) => `<button class="swatch" style="background:${rgbCSS(color)}" data-overview-sample="${i}" data-overview-sample-index="${samples().indexOf(sample)}" aria-label="Inspect tensor value ${i}"></button>`).join("")}</div><p class="hint">${esc(sample.name)} · shape ${esc((sample.shape ?? []).join(" × "))} · ${esc(codecLabel(sample.codec))}</p><a href="#tensor">Open tensor viewer →</a>` : `<p class="hint">This run did not record tensor samples.</p>`;
  const recent = table(["Task", "Representation", "Seed", "Metric", "Train s"], experimentRows(experiments().slice(-7).reverse()), "Latest recorded experiments");
  return summaryMetrics() + `<div class="grid">${card("Validation loss", task ? `${human(task)} · seed ${first.seed}` : "Recorded epochs", chart(curves), chip("Measured"))}${card("Run environment", "Reproducibility context", facts([["Run", state.data.run_id], ["Status", state.data.status ?? "Recorded"], ["Created", localTime(state.data.created_at)], ...Object.entries(environment).slice(0,7).map(([k,v]) => [human(k), typeof v === "object" ? JSON.stringify(v) : v])]), chip("Local", "good"))}</div><div class="grid">${card("Experiment ledger", "Select a run to inspect its training record", recent)}${card("Chromatic tensor", "A color field from the recorded run", palette)}</div>${card("Research notes", "Recorded conclusions, including negative results", conclusionsMarkup((state.data.conclusions ?? []).slice(0, 4)))}`;
}

function experimentsView() {
  const list = experiments();
  if (!list.length) return card("Experiment ledger", "Results appear after a benchmark run", empty("No experiment records", "This view lists actual measurements and links to each saved artifact."));
  const tasks = [...new Set(list.map(e => e.task))], reps = [...new Set(list.map(e => e.representation))];
  const filtered = list.filter(e => (state.taskFilter === "all" || e.task === state.taskFilter) && (state.repFilter === "all" || e.representation === state.repFilter) && JSON.stringify([e.task,e.representation,e.id,e.seed]).toLowerCase().includes(state.query.toLowerCase()));
  return `<div class="toolbar"><div class="field"><label for="search">Search recorded experiments</label><input id="search" type="search" value="${esc(state.query)}" placeholder="Task, representation, or seed"></div><div class="field compact"><label for="task-filter">Task</label><select id="task-filter">${options([["all","All tasks"],...tasks.map(t => [t,human(t)])],state.taskFilter)}</select></div><div class="field compact"><label for="rep-filter">Representation</label><select id="rep-filter">${options([["all","All representations"],...reps.map(r => [r,human(r)])],state.repFilter)}</select></div></div>${card("Recorded experiments", `${filtered.length} of ${list.length} runs`, table(["Task","Representation","Seed","Task metric","Train s","Code + metadata bytes","Record"],experimentRows(filtered,true),"Actual measurements by experiment"),chip(`${filtered.length} runs`))}`;
}

function tensorView() {
  const sample = state.sample ?? samples()[state.sampleIndex];
  const picker = samples().length ? `<div class="field"><label for="sample-select">Recorded tensor</label><select id="sample-select">${options(samples().map((s,i) => [i,`${s.name} · ${codecLabel(s.codec)}`]),state.sampleIndex)}</select></div>` : sample ? "" : `<p class="hint">No recorded tensor samples. Encode values below to inspect a real tensor.</p>`;
  const controls = `<form id="encode-form"><div class="toolbar"><div class="field"><label for="codec-select">Codec</label><select id="codec-select">${options([["rgb","RGB scalar · 24 bits"],["rgbd","RGBD · 32-bit byte token"],["hsv","HSV scalar"],["hue","Circular hue · 2π period"],["learned","Learned codebook · fit on input"],["learned_rgbd","Learned RGBD · residual modifier"],["hierarchical","Hierarchical · 3 × 16 codes"],["int8","Uniform INT8 baseline"],["int4","Uniform INT4 baseline"],["fixed24","Fixed-point 24-bit baseline"]],state.inputCodec)}</select></div><div class="field compact"><label for="input-shape">Tensor shape</label><input id="input-shape" required value="${esc(state.inputShape)}" aria-describedby="encode-hint"></div></div><div class="field"><label for="input-values">Numerical values, JSON array</label><textarea id="input-values" rows="3" required aria-describedby="encode-hint">${esc(state.inputValues)}</textarea><p id="encode-hint" class="hint">Up to 4,096 finite values. Shape must match the array. Range-based codecs use the input's minimum and maximum. Hue wraps modulo 2π.</p></div><button type="submit" class="button primary" id="encode-button">Encode with Python codec</button><p id="encode-status" class="hint section-gap" aria-live="polite"></p></form>`;
  const inspector = sample ? `<div class="toolbar">${picker}<span class="chip">${esc((sample.shape ?? []).join(" × "))}</span>${state.sample ? '<button class="button small" id="reset-sample">Return to recorded tensors</button>' : ""}</div><div class="tensor-grid"><figure><h3>Numerical tensor</h3><canvas id="numeric-canvas" class="tensor-canvas" role="img" tabindex="0" aria-label="Numerical tensor. Use arrow keys to inspect adjacent values."></canvas><figcaption>Grayscale normalized to this tensor's range.</figcaption></figure><figure><h3>Chromatic tensor</h3><canvas id="rgb-canvas" class="tensor-canvas" role="img" tabindex="0" aria-label="Chromatic tensor. Use arrow keys to inspect adjacent values."></canvas><figcaption>Actual RGB display codes from the codec.</figcaption></figure><figure><h3>Reconstruction error</h3><canvas id="error-canvas" class="tensor-canvas" role="img" tabindex="0" aria-label="Reconstruction error. Use arrow keys to inspect adjacent values."></canvas><figcaption>Absolute error, scaled to the largest error. Hue uses circular error.</figcaption></figure></div><p class="hint">Hover a cell or use arrow keys on a tensor image. Every value also appears in the data table.</p>` : picker;
  return `<div class="stack">${card("Tensor inspection",sample?.experiment_id ? `${sample.name} · diagnostic encoding` : sample?.name ?? "Live input",inspector,sample ? chip(codecLabel(sample.codec)) : "")}<div class="grid equal">${card("Selected value","Color, code, and reconstruction",'<div id="token-inspector">Select a tensor value to inspect its code.</div>')}${card("Encode your own tensor","This calls the installed reference implementation",controls)}</div>${sample ? card("Tensor values","A numerical alternative to the color fields",'<div id="tensor-data"></div>') : ""}</div>`;
}

function rgbCSS(color) {
  return `rgb(${(color ?? [128,128,128]).slice(0,3).map(v => Math.max(0,Math.min(255,Math.round(Number(v))))).join(" ")})`;
}

function sampleError(sample, index) {
  if (finite(sample.errors?.[index])) return sample.errors[index];
  const error = Math.abs(sample.values[index] - sample.decoded[index]);
  if ((sample.codec?.name ?? sample.codec) === "hue") { const period = sample.metadata?.period ?? sample.codec?.period ?? Math.PI*2; return Math.min(error % period, period - error % period); }
  return error;
}

function rgbToHSV(rgb) {
  const [r,g,b] = rgb.map(c => c/255), max = Math.max(r,g,b), min = Math.min(r,g,b), delta = max-min;
  let hue = delta === 0 ? 0 : max === r ? ((g-b)/delta)%6 : max === g ? (b-r)/delta+2 : (r-g)/delta+4;
  return [(hue/6+1)%1, max===0 ? 0 : delta/max, max];
}

function token(sample,index) { const item = sample.tokens?.[index]; return Array.isArray(item) ? item.join("/") : item ?? (sample.codes?.[index] ?? []).join("/") ?? "Not recorded"; }

function updateInspector(sample, index, announce = false) {
  if (!sample?.values?.length) return;
  state.tokenIndex = Math.max(0,Math.min(sample.values.length-1,index));
  const n = state.tokenIndex, rgb = sample.rgb?.[n] ?? [128,128,128], hsv = sample.hsv?.[n] ?? rgbToHSV(rgb);
  const tok = token(sample,n), frequency = sample.frequencies?.[n] ?? (sample.tokens ?? []).filter(t => JSON.stringify(t) === JSON.stringify(sample.tokens[n])).length;
  const entries = [["Flat index",n],["Token / code",tok],["Original",sample.values[n]],["Decoded",sample.decoded[n]],["Absolute error",sampleError(sample,n)],["RGB",rgb.map(fmt).join(", ")],["HSV",hsv.map(fmt).join(", ")],["Token frequency",`${frequency} / ${sample.values.length}`]];
  if ((sample.codec?.name ?? sample.codec) === "rgbd") entries.splice(6,0,["D exponent byte",sample.codes?.[n]?.[3]]);
  $("token-inspector").innerHTML = `<div class="inspector"><div class="inspector-color" style="background:${rgbCSS(rgb)}"></div>${facts(entries)}</div>${sample.provenance ? `<p class="hint section-gap">${esc(sample.provenance)}</p>` : ""}${(sample.metadata ?? (typeof sample.codec === "object" ? sample.codec : null)) ? `<p class="hint section-gap">Codec metadata ${esc(JSON.stringify(sample.metadata ?? sample.codec))}</p>` : ""}`;
  for (const id of ["numeric-canvas", "rgb-canvas", "error-canvas"]) {
    const canvas = $(id);
    if (!canvas?._tensorImage) continue;
    const ctx = canvas.getContext("2d"), cols = canvas.width / 16, x = n % cols * 16, y = Math.floor(n / cols) * 16;
    ctx.putImageData(canvas._tensorImage, 0, 0);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(x+2, y+2, 12, 12);
    ctx.strokeStyle = "#111"; ctx.lineWidth = 1; ctx.strokeRect(x+3, y+3, 10, 10);
  }
  if (announce) $("announcer").textContent = `Value ${n}. Original ${fmt(sample.values[n])}, decoded ${fmt(sample.decoded[n])}, error ${fmt(sampleError(sample,n))}, token ${tok}.`;
}

function drawTensors() {
  const sample = state.sample ?? samples()[state.sampleIndex];
  if (!sample?.values?.length || !$("rgb-canvas")) return;
  const n = sample.values.length, cols = sample.shape?.at(-1) > 1 && sample.shape.at(-1) <= 128 ? sample.shape.at(-1) : Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
  const minimum = Math.min(...sample.values), maximum = Math.max(...sample.values), errors = sample.values.map((v,i) => sampleError(sample,i)), maxError = Math.max(...errors);
  for (const [id,mode] of [["numeric-canvas","number"],["rgb-canvas","rgb"],["error-canvas","error"]]) {
    const canvas = $(id), ctx = canvas.getContext("2d");
    canvas.width = cols * 16; canvas.height = rows * 16; canvas.style.aspectRatio = `${cols} / ${rows}`;
    ctx.fillStyle = "#d4d8df"; ctx.fillRect(0,0,canvas.width,canvas.height);
    for (let i=0;i<n;i++) {
      const normalized = maximum===minimum ? .5 : (sample.values[i]-minimum)/(maximum-minimum);
      const color = mode === "rgb" ? rgbCSS(sample.rgb?.[i]) : mode === "number" ? `rgb(${Math.round(normalized*220+20)} ${Math.round(normalized*220+20)} ${Math.round(normalized*220+20)})` : `rgb(${Math.round(245-135*(maxError?errors[i]/maxError:0))} ${Math.round(244-195*(maxError?errors[i]/maxError:0))} ${Math.round(250-66*(maxError?errors[i]/maxError:0))})`;
      ctx.fillStyle = color; ctx.fillRect(i%cols*16,Math.floor(i/cols)*16,16,16);
    }
    canvas._tensorImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
    canvas.addEventListener("pointermove",event => { const rect=canvas.getBoundingClientRect(); updateInspector(sample,Math.floor((event.clientY-rect.top)/rect.height*rows)*cols+Math.floor((event.clientX-rect.left)/rect.width*cols)); });
    canvas.addEventListener("focus",()=>updateInspector(sample,state.tokenIndex,true));
    canvas.addEventListener("keydown",event=> { const move={ArrowRight:1,ArrowLeft:-1,ArrowDown:cols,ArrowUp:-cols,Home:-state.tokenIndex,End:n-1-state.tokenIndex}[event.key]; if(move != null) { event.preventDefault(); updateInspector(sample,state.tokenIndex+move,true); } });
  }
  updateInspector(sample,state.tokenIndex);
  $("tensor-data").innerHTML=table(["Index","Original","Decoded","Abs. error","Token","RGB"],sample.values.map((value,index)=>[`<button class="table-link" data-token="${index}">${index}</button>`,... [value,sample.decoded[index],errors[index]].map(v=>`<span class="number">${esc(fmt(v))}</span>`),`<span class="number">${esc(token(sample,index))}</span>`,`<span class="number">${esc((sample.rgb?.[index] ?? []).join(", "))}</span>`]),"Every encoded tensor value");
  document.querySelectorAll("[data-token]").forEach(button=>button.addEventListener("click",()=>updateInspector(sample,Number(button.dataset.token),true)));
}

function colorView() {
  const sample = state.sample ?? samples()[state.sampleIndex];
  const picker = `<div class="toolbar"><div class="field"><label for="color-sample">Tensor</label><select id="color-sample">${options(samples().map((s,i)=>[i,`${s.name} · ${codecLabel(s.codec)}`]),state.sampleIndex)}</select></div><div class="field compact"><label for="color-mode">Projection</label><select id="color-mode">${options([["rgb","RGB · red / green"],["hsv","HSV · hue / value"],["hue","Circular hue"]],state.colorMode)}</select></div></div>`;
  const visualization = sample ? `${picker}<figure style="margin:0"><canvas id="color-canvas" class="plot-canvas" role="img" aria-label="Color distribution for the selected tensor"></canvas><figcaption class="hint section-gap" id="color-caption"></figcaption></figure>` : `<p class="hint">Record a tensor sample or encode an input in the tensor viewer.</p>`;
  const geometry = `<p class="hint">RGB distance is a display-space distance. It does not necessarily track decoded scalar distance.</p><p class="hint">Hue wraps at 0 and 1. Circular tasks must use circular error, so values near the seam remain neighbors.</p><p class="hint">The plotted colors are the codec's actual display colors. They may differ from its numerical token identifiers.</p>`;
  return `<div class="grid">${card("Chromatic distribution",sample?.name ?? "Encoded tensors",visualization)}${card("Reading this projection","Geometry depends on the codec",geometry + (sample ? facts([["Values",sample.values.length],["Codec",codecLabel(sample.codec)],["Shape",sample.shape.join(" × ")],["Projection",human(state.colorMode)]]) : ""))}</div>${sample ? card("Color coordinates","The numerical data behind the plot",table(["Index","Red","Green","Blue","Hue","Saturation","Value"],sample.rgb.map((rgb,i)=>[i,...rgb,...rgbToHSV(rgb)].map(v=>`<span class="number">${esc(fmt(v))}</span>`)),"Recorded RGB and calculated HSV coordinates")) : ""}`;
}

function drawColorSpace() {
  const sample=state.sample ?? samples()[state.sampleIndex], canvas=$("color-canvas"); if(!sample || !canvas) return;
  const width=650,height=310,pad=45; canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext("2d"), dark=matchMedia("(prefers-color-scheme: dark)").matches;
  ctx.clearRect(0,0,width,height);ctx.strokeStyle=dark?"#354053":"#dce0e5";ctx.fillStyle=dark?"#a5b0c4":"#646b76";ctx.font="11px Segoe UI";
  if(state.colorMode==="hue") {
    const radius=110,cx=width/2,cy=height/2; ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.stroke();
    sample.rgb.forEach(rgb=>{const [h,s,v]=rgbToHSV(rgb),r=radius*(.2+.8*s);ctx.fillStyle=rgbCSS(rgb);ctx.beginPath();ctx.arc(cx+Math.cos(h*Math.PI*2)*r,cy+Math.sin(h*Math.PI*2)*r,Math.max(3,v*6),0,Math.PI*2);ctx.fill();ctx.strokeStyle=dark?"#ffffff50":"#00000030";ctx.stroke();});
    $("color-caption").textContent=`${sample.values.length} actual RGB colors. Angle is hue, radius is saturation, marker size is value.`;return;
  }
  for(let i=0;i<=4;i++){const x=pad+i/4*(width-pad*2),y=height-pad-i/4*(height-pad*2);ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,height-pad);ctx.moveTo(pad,y);ctx.lineTo(width-pad,y);ctx.stroke();ctx.fillText(String(state.colorMode==="rgb"?Math.round(i/4*255):i/4),x-4,height-pad+20);ctx.fillText(String(state.colorMode==="rgb"?Math.round(i/4*255):i/4),10,y+3);}
  sample.rgb.forEach(rgb=>{const hsv=rgbToHSV(rgb),x=state.colorMode==="rgb"?rgb[0]/255:hsv[0],y=state.colorMode==="rgb"?rgb[1]/255:hsv[2];ctx.fillStyle=rgbCSS(rgb);ctx.beginPath();ctx.arc(pad+x*(width-pad*2),height-pad-y*(height-pad*2),4.3,0,Math.PI*2);ctx.fill();ctx.strokeStyle=dark?"#ffffff50":"#00000030";ctx.stroke();});
  ctx.fillStyle=dark?"#a5b0c4":"#646b76";ctx.fillText(state.colorMode==="rgb"?"Red channel":"Hue",width/2-25,height-4);
  $("color-caption").textContent=`${sample.values.length} actual RGB colors. Horizontal ${state.colorMode==="rgb"?"red":"hue"}, vertical ${state.colorMode==="rgb"?"green":"value"}. Use the coordinate table for exact values.`;
}

function codebookView() {
  const all=books(); if(!all.length) return card("Chromatic vocabulary","Learned representatives and observed usage",empty("No codebook recorded","The benchmark runner records learned vocabulary values and usage frequencies after fitting."));
  const book=all[state.bookIndex] ?? all[0], pageSize=128, start=state.bookPage*pageSize, end=Math.min(book.values.length,start+pageSize), frequencies=book.frequencies ?? [], maximum=Math.max(1,...frequencies), total=frequencies.reduce((a,b)=>a+b,0), used=frequencies.filter(f=>f>0).length;
  const picker=`<div class="toolbar"><div class="field"><label for="book-select">Recorded codebook</label><select id="book-select">${options(all.map((b,i)=>[i,b.name]),state.bookIndex)}</select></div></div>`;
  const palette=`<div class="palette">${(book.rgb ?? []).slice(start,end).map((rgb,offset)=>{const i=start+offset;return `<button class="swatch" data-book-token="${i}" style="background:${rgbCSS(rgb)}" aria-label="Code ${i}, value ${fmt(book.values[i])}, frequency ${frequencies[i] ?? 0}"></button>`;}).join("")}</div><p class="hint">Select a color to inspect its learned numerical representative.</p><div id="book-inspector"></div>`;
  const rows=book.values.slice(start,end).map((value,offset)=>{const i=start+offset;return [`<button class="table-link" data-book-token="${i}">${esc(colorSymbol(book.rgb?.[i]))}</button>`,`<span class="number">${i}</span>`,`<span class="number">${esc(fmt(value))}</span>`,`<span class="number">${esc(fmt(frequencies[i] ?? 0))}</span><div class="bar-track"><div class="bar-fill" style="width:${(frequencies[i]??0)/maximum*100}%"></div></div>`,`<span class="number">${esc((book.rgb?.[i]??[]).join(", "))}</span>`];});
  const pagination=`<div class="toolbar"><button class="button small" id="book-previous"${start===0?" disabled":""}>Previous codes</button><span class="hint">Codes ${start} to ${end-1} of ${book.values.length}</span><button class="button small" id="book-next"${end>=book.values.length?" disabled":""}>Next codes</button><form id="book-jump" style="display:flex;gap:8px;align-items:center"><label for="book-entry" class="hint">Inspect code</label><input id="book-entry" type="number" required min="0" max="${book.values.length-1}" value="${state.bookToken}" style="width:100px"><button class="button small" type="submit">Go</button></form></div>`;
  return picker+`<div class="grid equal">${card("Codebook palette",book.name,palette,chip(`${book.values.length} codes`))}${card("Observed usage","Counts from the recorded encoding",facts([["Vocabulary size",book.values.length],["Codes with recorded usage",used],["Total assignments",total],["Unused codes",book.values.length-used]])+`<p class="hint section-gap">A learned scalar vocabulary is quantization. These measurements do not by themselves establish a benefit from color organization.</p>`)}</div>${card("Vocabulary entries","Learned scalar values and assignment counts",pagination+table(["RGB symbol","Palette ID","Representative","Frequency","Display RGB"],rows,"Recorded learned codebook",true))}`;
}

function showBookToken(index) {
  const book=books()[state.bookIndex] ?? books()[0];if(!book || !$("book-inspector")) return;
  const n=Math.min(book.values.length-1,Math.max(0,index));state.bookToken=n;
  $("book-inspector").innerHTML=facts([["RGB symbol",colorSymbol(book.rgb?.[n])],["Palette ID",n],["Representative",book.values[n]],["Frequency",book.frequencies?.[n]??0],["RGB",(book.rgb?.[n]??[]).join(", ")]]);
}

function trainingView() {
  const list=experiments(), e=currentExperiment();if(!e) return card("Training records","Loss curves come from completed runs",empty("No training records","Run a benchmark or a training command to save actual epochs, metrics, and execution details."));
  const peers=list.filter(p=>p.task===e.task&&p.seed===e.seed);
  const picker=`<div class="toolbar"><div class="field"><label for="experiment-select">Experiment</label><select id="experiment-select">${options(list.map(p=>[p.id,`${human(p.task)} · ${human(p.representation)} · seed ${p.seed}`]),e.id)}</select></div></div>`;
  const losses=[{name:"Training loss",color:COLORS[0],points:(e.curves??[]).map(p=>[p.epoch,p.train_loss])},{name:"Validation loss",color:COLORS[1],dashed:true,points:(e.curves??[]).map(p=>[p.epoch,p.val_loss])}];
  const comparison=peers.map((p,i)=>({name:human(p.representation),color:COLORS[i%COLORS.length],points:(p.curves??[]).map(p=>[p.epoch,p.val_loss])}));
  const metrics=Object.entries(e.metrics??{}).map(([k,v])=>[human(k),v]);
  return picker+`<div class="grid">${card("Training and validation loss",`${human(e.task)} · ${human(e.representation)}`,chart(losses),chip(`seed ${e.seed}`))}${card("Final recorded metrics",e.id,facts(metrics))}</div><div class="grid">${card("Same task, same seed",`${peers.length} recorded representations`,chart(comparison),chip("Validation"))}${card("Execution record","Reported by the experiment runner",facts(Object.entries(e.execution??{}).filter(([k,v])=>v===null||typeof v!=="object").map(([k,v])=>[human(k),v]))+`<details class="section-gap"><summary class="hint">View full execution metadata</summary><pre class="json">${esc(JSON.stringify(e.execution??{},null,2))}</pre></details>`+(e.artifact_path?`<p class="section-gap"><a href="/api/artifact?path=${encodeURIComponent(e.artifact_path)}" download>Download saved artifact →</a></p>`:""))}</div>${card("Epoch ledger","The recorded measurements behind the curves",genericTable(e.curves??[],"Recorded training epochs"))}${e.dataset ? `<div class="section-gap">${card("Dataset record","Sources, sample counts, and dataset identity",facts(Object.entries(e.dataset).map(([k,v])=>[human(k),typeof v==="object"?JSON.stringify(v):v])))}</div>` : ""}`;
}

function benchmarkView() {
  const list=experiments();if(!list.length) return card("Benchmark comparison","Accuracy, runtime, and storage",empty("No benchmark data","The runner records task-level results and codec measurements. Missing metrics stay marked as not recorded."));
  const group=new Map();list.forEach(e=>{const key=`${e.task}/${e.representation}`;if(!group.has(key))group.set(key,[]);group.get(key).push(e);});
  const rows=[...group.values()].map(runs=>{const e=runs[0],key=metricKey(e),values=runs.map(r=>r.metrics?.[key]).filter(finite),mean=values.reduce((a,b)=>a+b,0)/values.length,std=values.length>1?Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1)):null;const average=(...names)=>{const values=runs.map(r=>metricValue(r,...names)).filter(finite);return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;};return [esc(human(e.task)),chip(human(e.representation)),runs.length,`<span class="number">${esc(fmt(mean))}</span><small class="hint"> ${esc(human(key))}</small>`,`<span class="number">${std==null?"One seed":esc(fmt(std))}</span>`,...[average("training_seconds","train_seconds"),average("inference_batch_ms","inference_ms"),average("bytes_model","parameter_bytes"),average("bits_per_value","bits_per_parameter")].map(v=>`<span class="number">${esc(fmt(v))}</span>`)];});
  const warnings=`<p class="hint">Means combine only the recorded seeds. A single seed has no measured seed-to-seed variability.</p><p class="hint">Code + metadata bytes describe the recorded diagnostic encoding. Training uses numerical PyTorch tensors unless the execution record states otherwise. Checkpoint bytes, encoded bytes, and training memory are different measurements.</p>`;
  return card("Task comparison","Means and sample standard deviations across recorded seeds",table(["Task","Representation","Runs","Mean task metric","Seed std.","Train s","Inference ms","Code + metadata bytes","Bits / value"],rows,"Measured task benchmark summary"),chip("Actual runs"))+`<div class="grid equal section-gap">${card("Storage and precision","Compare error at an equal bit budget",genericTable(state.data.precision??[],"Codec reconstruction and storage measurements"))}${card("Measurement scope","Interpret timing and memory with the execution record",warnings+facts([["Run",state.data.run_id],["Environment",JSON.stringify(state.data.environment??{})],["Task records",list.length]]))}</div>`;
}

function ablationsView() {
  const precision=state.data?.precision??[],operators=state.data?.operators??[];
  if(!precision.length&&!operators.length) return card("Codec ablations","Reconstruction, storage, and operator behavior",empty("No ablation measurements","Run the benchmark suite to record measured precision sweeps and direct token-space operators."));
  const filtered=precision.filter(p=>(p.distribution??"uniform")===state.ablationDistribution&&(p.error_domain??"linear")===state.ablationDomain);
  const storedBits=p=>(p.total_bytes??(p.storage_bytes+p.metadata_bytes))*8/p.samples;
  const numericRows=filtered.filter(p=>finite(p.mae)&&p.samples>0&&finite(storedBits(p))),series=numericRows.map((p,i)=>({name:human(p.codec??p.representation??p.name??`codec ${i}`),color:COLORS[i%COLORS.length],points:[[storedBits(p),p.mae]]}));
  const controls=`<div class="toolbar"><div class="field"><label for="ablation-distribution">Input distribution</label><select id="ablation-distribution">${options([...new Set(precision.map(p=>p.distribution??"uniform"))].map(v=>[v,human(v)]),state.ablationDistribution)}</select></div><div class="field"><label for="ablation-domain">Error domain</label><select id="ablation-domain">${options([...new Set(precision.map(p=>p.error_domain??"linear"))].map(v=>[v,human(v)]),state.ablationDomain)}</select></div></div>`;
  return controls+`<div class="grid equal">${card("Error versus storage budget","Horizontal axis: stored bits per value, including codec metadata",chart(series,"Mean absolute error",numericRows.every(p=>p.mae>0),"stored bits per value including codec metadata"),chip("Codec sweep"))}${card("Fair comparison","Bits and meaning both matter",'<p class="hint">RGB scalar encoding uses 24 raw bits per value. Compare it with 24-bit fixed-point, alongside strong INT8 and INT4 baselines.</p><p class="hint">A color permutation preserves scalar quantization error. A topology claim requires a task that actually uses color relationships.</p><p class="hint">Hue is a periodic encoding. Its circular error answers a different question from ordinary scalar error.</p>')}</div>${card("Precision ledger","Recorded codec metrics, including negative results",genericTable(filtered,"Measured codec ablations"))}<div class="section-gap">${card("Operator ledger","Execution paths and measured operator error",genericTable(operators,"Measured chromatic operator behavior"))}</div>${state.data?.robustness?.length ? `<div class="section-gap">${card("Robustness ledger","Measured perturbations of numerical values and chromatic codes",genericTable(state.data.robustness,"Measured robustness experiments"))}</div>` : ""}`;
}

function architectureView() {
  const flow=[["Input","PyTorch tensor","Finite numerical values and original shape."],["Encoding","Chromatic codec","Discrete RGB, HSV, hue, learned, or residual codes."],["Storage","ChromaTensor","Integer codes, codec metadata, and tensor shape."],["Execution","Operators / layers","Token lookup or decoded numerical computation, recorded explicitly."],["Output","Decoded tensor","Numerical reconstruction for existing PyTorch code."]];
  const architecture=`<div class="architecture">${flow.map(([label,title,text])=>`<div class="architecture-item"><span>${esc(label)}</span><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`).join("")}</div>`;
  const modules=table(["Module","Responsibility"],[["chromatensor.codecs","Encode, decode, and map codes to display RGB."],["chromatensor.tensor","Store shape and code metadata, convert to and from PyTorch."],["chromatensor.operators","Record direct token lookup and decoded execution paths."],["chromatensor.nn","Quantize neural weights and activations during numerical training."],["scripts/run_benchmarks.py","Save measured task results, codec sweeps, and artifacts."],["dashboard.server","Read saved runs and execute bounded codec inspections."]].map(row=>row.map(esc)),"Implementation responsibilities",true);
  const boundaries=`<p class="hint">The representation is chromatic. The hardware still stores bits, and neural training still uses numerical arithmetic.</p><p class="hint">Straight-through gradient estimators use numerical gradients to train through discrete quantization. A display color does not create a differentiable color-native processor.</p><p class="hint">Lookup-table operators can work on discrete codes. Larger matrix operations can decode to conventional tensors. The operator record must identify the path.</p>`;
  return card("Representation pipeline","The actual encoding and interoperability boundary",architecture)+`<div class="grid equal section-gap">${card("Implementation map","Where each responsibility lives",modules)}${card("Computation boundary","How to read a chromatic training result",boundaries)}</div>${card("Saved run configuration","Configuration from the current result artifact",state.data?`<pre class="json">${esc(JSON.stringify({config:state.data.config??{},instrumentation:state.data.instrumentation??{}},null,2))}</pre>`:'<p class="hint">No saved configuration. Record a benchmark run to inspect its exact setup.</p>')}`;
}

function conclusionsMarkup(list) {
  return list.length?`<ol class="conclusion-list">${list.map((c,i)=>`<li><span class="finding-index">${String(i+1).padStart(2,"0")}</span><span>${esc(typeof c==="string"?c:JSON.stringify(c))}</span></li>`).join("")}</ol>`:'<p class="hint">The current result artifact contains no research conclusions.</p>';
}

function conclusionsView() {
  const conclusions=state.data?.conclusions??[];
  return `<div class="grid">${card("Recorded findings","Conclusions saved with this measured run",conclusionsMarkup(conclusions),state.data?chip("Run evidence"):chip("No run","warn"))}${card("Evidence boundaries","Claims that require more than a successful run",'<p class="hint">Useful learning behavior establishes that a quantized representation can train on the recorded tasks.</p><p class="hint">A storage reduction does not establish faster training. A lower reconstruction error does not establish better generalization.</p><p class="hint">Novelty, universal efficiency, and a benefit from color topology require separate evidence. Negative results remain part of the record.</p>')}</div>${card("Reproducibility record","Export the full run or inspect each experiment artifact",state.data?facts([["Run",state.data.run_id],["Created",state.data.created_at],["Schema",state.data.schema_version],["Experiments",experiments().length],["Recorded seeds",[...new Set(experiments().map(e=>e.seed))].join(", ")],["Configuration",JSON.stringify(state.data.config??{})]]):'<p class="hint">No recorded run. Research conclusions appear only after the benchmark runner writes measured results.</p>')}`;
}

function render(focus=false) {
  window.CubeLab?.unmount();
  const info=VIEWS.find(v=>v[0]===state.view)??VIEWS[0];state.view=info[0];
  document.title=`${info[1]} | ChromaTensor`;$("page-title").textContent=info[1];$("breadcrumb").textContent=info[1];$("page-description").textContent=info[2];
  document.querySelectorAll(".nav-link").forEach(a=>{if(a.dataset.view===state.view)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");});
  const views={overview,experiments:experimentsView,tensor:tensorView,color:colorView,codebook:codebookView,training:trainingView,benchmarks:benchmarkView,ablations:ablationsView,architecture:architectureView,conclusions:conclusionsView,cube:()=>window.CubeLab.markup()};
  $("view").innerHTML=views[state.view]();
  bindView();
  if(state.view==="tensor")drawTensors();if(state.view==="color")drawColorSpace();if(state.view==="codebook")showBookToken(state.bookToken);
  if(state.view==="cube")window.CubeLab.mount();
  if(focus) { window.scrollTo(0,0); $("page-title").focus({preventScroll:true}); }
}

function bindView() {
  document.querySelectorAll("[data-experiment]").forEach(button=>button.addEventListener("click",()=>{state.selectedExperiment=button.dataset.experiment;location.hash="training";}));
  document.querySelectorAll("[data-overview-sample]").forEach(button=>button.addEventListener("click",()=>{state.sample=null;state.sampleIndex=Number(button.dataset.overviewSampleIndex??0);state.tokenIndex=Number(button.dataset.overviewSample);location.hash="tensor";}));
  const bind=(id,event,fn)=>$(id)?.addEventListener(event,fn);
  bind("task-filter","change",event=>{state.taskFilter=event.target.value;render();});bind("rep-filter","change",event=>{state.repFilter=event.target.value;render();});
  bind("search","input",event=>{state.query=event.target.value;const pos=event.target.selectionStart;render();$("search").focus();$("search").setSelectionRange(pos,pos);});
  bind("sample-select","change",event=>{state.sample=null;state.sampleIndex=Number(event.target.value);state.tokenIndex=0;render();});bind("reset-sample","click",()=>{state.sample=null;state.tokenIndex=0;render();});
  bind("encode-form","submit",encodeInput);
  bind("color-sample","change",event=>{state.sample=null;state.sampleIndex=Number(event.target.value);render();});bind("color-mode","change",event=>{state.colorMode=event.target.value;render();});
  bind("book-select","change",event=>{state.bookIndex=Number(event.target.value);state.bookToken=0;state.bookPage=0;render();});document.querySelectorAll("[data-book-token]").forEach(button=>button.addEventListener("click",()=>showBookToken(Number(button.dataset.bookToken))));
  bind("book-previous","click",()=>{state.bookPage=Math.max(0,state.bookPage-1);render();});bind("book-next","click",()=>{state.bookPage++;render();});bind("book-jump","submit",event=>{event.preventDefault();state.bookToken=Number($("book-entry").value);state.bookPage=Math.floor(state.bookToken/128);render();});
  bind("ablation-distribution","change",event=>{state.ablationDistribution=event.target.value;render();});bind("ablation-domain","change",event=>{state.ablationDomain=event.target.value;render();});
  bind("experiment-select","change",event=>{state.selectedExperiment=event.target.value;render();});
}

async function encodeInput(event) {
  event.preventDefault();state.inputShape=$("input-shape").value;state.inputValues=$("input-values").value;state.inputCodec=$("codec-select").value;const status=$("encode-status"),button=$("encode-button");button.disabled=true;status.textContent="Encoding with the local Python implementation.";
  try {
    const values=JSON.parse($("input-values").value),shape=$("input-shape").value.split(/[\s,×x]+/).filter(Boolean).map(Number),codec=$("codec-select").value;
    const response=await fetch("/api/tensor",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({values,shape,codec})});const result=await response.json();if(!response.ok)throw new Error(result.error??"Encoding failed");
    state.sample=result;state.tokenIndex=0;render();$("encode-status").textContent=`Encoded ${result.values.length} values with ${human(codec)}. Measured MAE ${fmt(result.metrics.mae)}.`;$("announcer").textContent=$("encode-status").textContent;
  } catch(error) {status.textContent=error.message;button.disabled=false;}
}

async function loadResults() {
  const button=$("refresh");button.disabled=true;
  try {
    const response=await fetch("/api/results",{cache:"no-store"}),data=await response.json();
    if(!response.ok)throw new Error(data.error??"Results could not be loaded");state.data=data;window.CubeLab?.setResults(data);$("notice").hidden=true;$("connection").textContent="Recorded results loaded";
    $("run-meta").innerHTML=`<strong>${esc(data.run_id)}</strong>${esc(localTime(data.created_at))}${data.status ? ` · ${esc(data.status)}` : ""}`;$("announcer").textContent=`Loaded ${experiments().length} recorded experiments.`;
  } catch(error) {$("notice").textContent=error.message;$("notice").hidden=false;$("connection").textContent=state.data?"Refresh failed, previous run shown":"No recorded run";$("run-meta").innerHTML=state.data?`<strong>${esc(state.data.run_id)}</strong>Previous loaded results`:'<strong>Local experiment workspace</strong>Waiting for results/latest.json';}
  finally {button.disabled=false;$("export").disabled=!state.data;render();}
}

$("navigation").innerHTML=VIEWS.map(([id,label],i)=>`<a class="nav-link" href="#${id}" data-view="${id}">${svgIcon(i)}<span>${esc(label)}</span></a>`).join("");
$("refresh").addEventListener("click",loadResults);
$("export").addEventListener("click",()=>{if(!state.data)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.data,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`chromatensor-${String(state.data.run_id).replace(/[^a-z0-9_-]/gi,"_")}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
window.addEventListener("hashchange",()=>{state.view=location.hash.slice(1)||"cube";render(true);});
state.view=location.hash.slice(1)||"cube";render();loadResults();
