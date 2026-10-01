// Browser illustration only. The native library and its tests live in native/.
export const HUE_MODULUS = 391680;

export function encodePhase(phase) {
  if (!Number.isInteger(phase) || phase < 0 || phase >= HUE_MODULUS) {
    throw new RangeError('Phase must be an integer from 0 to 391679.');
  }
  const hue = Math.floor(phase / 256);
  const sector = Math.floor(hue / 255);
  const t = hue % 255;
  const rgb = [
    [255, t, 0], [255 - t, 255, 0], [0, 255, t],
    [0, 255 - t, 255], [t, 0, 255], [255, 0, 255 - t],
  ][sector];
  return [...rgb, phase % 256];
}

export function decodeToken(token) {
  if (!Array.isArray(token) || token.length !== 4
      || !token.every(value => Number.isInteger(value) && value >= 0 && value <= 255)) {
    throw new RangeError('RGBD must contain four bytes.');
  }
  const [r, g, b, d] = token;
  let hue;
  if (r === 255 && b === 0 && g < 255) hue = g;
  else if (g === 255 && b === 0 && r > 0) hue = 255 + 255 - r;
  else if (r === 0 && g === 255 && b < 255) hue = 510 + b;
  else if (r === 0 && b === 255 && g > 0) hue = 765 + 255 - g;
  else if (g === 0 && b === 255 && r < 255) hue = 1020 + r;
  else if (r === 255 && g === 0 && b > 0) hue = 1275 + 255 - b;
  else throw new RangeError('RGB must be a canonical saturated hue.');
  return hue * 256 + d;
}

// This tiny example uses ordinary JavaScript numbers. No training is performed.
export const HIDDEN_WEIGHTS = [[0.8, -0.4, 0.6], [-0.5, 0.9, 0.3], [0.4, 0.2, -0.7], [0.7, 0.5, -0.2]];
export const OUTPUT_WEIGHTS = [[0.6, -0.3, 0.8, 0.4], [-0.2, 0.7, 0.3, 0.9]];

export function evaluateNetwork(signal) {
  if (!Number.isFinite(signal) || signal < 0 || signal > 1) {
    throw new RangeError('Signal must be a number between 0 and 1.');
  }
  const inputs = [signal, 0.4, 0.8];
  const weightedSum = (values, weights) => values.reduce((sum, value, index) => sum + value * weights[index], 0);
  const sums = HIDDEN_WEIGHTS.map(weights => weightedSum(inputs, weights));
  const hidden = sums.map(value => Math.max(0, value));
  const outputs = OUTPUT_WEIGHTS.map(weights => weightedSum(hidden, weights));
  return { inputs, sums, hidden, outputs };
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const INK = '#202b2b';
const PAPER = '#f6f2e8';
const MUTED = '#596260';
const fmt = value => Math.abs(value) < 0.0005 ? '0.000' : value.toFixed(3);
const weightColor = value => `hsl(${Math.round((1 - value) * 125)} 76% 46%)`;

function svgElement(tag, attributes = {}, text = '') {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  if (text) element.textContent = text;
  return element;
}

function initNetwork() {
  const svg = document.getElementById('network-visual');
  if (!svg) return;
  const controls = Object.fromEntries([
    'mode-numeric', 'mode-color', 'play-toggle', 'step-animation',
    'signal-input', 'signal-value', 'frame-caption', 'weight-readout',
  ].map(id => [id, document.getElementById(id)]));
  if (Object.values(controls).some(element => !element)) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let colorMode = false;
  let playing = false;
  let clock = 0;
  let previousTime = null;
  let requestId = null;
  let lastStage = -1;
  let result = evaluateNetwork(Number(controls['signal-input'].value) / 100);
  const phaseDuration = 1700;
  const layers = [
    [{ x: 118, y: 102 }, { x: 118, y: 190 }, { x: 118, y: 278 }],
    [{ x: 460, y: 68 }, { x: 460, y: 149 }, { x: 460, y: 230 }, { x: 460, y: 311 }],
    [{ x: 802, y: 139 }, { x: 802, y: 244 }],
  ];

  svg.setAttribute('viewBox', '0 0 920 380');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-labelledby', 'network-title network-description');
  svg.replaceChildren();
  svg.append(svgElement('title', { id: 'network-title' }, 'One small network, two ways to display its weights'));
  const description = svgElement('desc', { id: 'network-description' });
  svg.append(description);
  svg.append(svgElement('rect', { width: 920, height: 380, fill: PAPER, rx: 18 }));
  const edgeLayer = svgElement('g', { 'aria-hidden': 'true' });
  const packetLayer = svgElement('g', { 'aria-hidden': 'true' });
  const labelLayer = svgElement('g', { 'aria-hidden': 'true' });
  const nodeLayer = svgElement('g', { 'aria-hidden': 'true' });
  svg.append(edgeLayer, packetLayer, labelLayer, nodeLayer);
  const edges = [];
  [HIDDEN_WEIGHTS, OUTPUT_WEIGHTS].forEach((matrix, layer) => {
    matrix.forEach((weights, to) => weights.forEach((weight, from) => {
      const a = layers[layer][from];
      const b = layers[layer + 1][to];
      const start = { x: a.x + 31, y: a.y };
      const end = { x: b.x - 31, y: b.y };
      const path = svgElement('path', {
        d: `M${start.x},${start.y} C${start.x + 110},${start.y} ${end.x - 110},${end.y} ${end.x},${end.y}`,
        fill: 'none', stroke: '#abb0a7', 'stroke-width': 1 + Math.abs(weight) * 2,
        'stroke-dasharray': weight < 0 ? '5 5' : 'none',
      });
      edgeLayer.append(path);
      const packet = svgElement(weight < 0 ? 'rect' : 'circle', weight < 0
        ? { width: 9, height: 9, rx: 1, stroke: INK, 'stroke-width': 1, visibility: 'hidden' }
        : { r: 5, stroke: INK, 'stroke-width': 1, visibility: 'hidden' });
      packetLayer.append(packet);
      const edge = { path, packet, weight, layer, from, to, start, end };
      edges.push(edge);
      if (layer === 0 && to === 0) {
        const x = 245 + from * 27;
        const y = [81, 133, 212][from];
        const group = svgElement('g');
        const background = svgElement('rect', { x: x - 26, y: y - 13, width: 52, height: 25, rx: 7, fill: PAPER, stroke: '#b9bfb6' });
        const dot = svgElement('circle', { cx: x - 15, cy: y, r: 4, fill: INK });
        const text = svgElement('text', { x: x + 3, y: y + 4, 'text-anchor': 'middle', fill: INK, 'font-size': 12, 'font-family': 'ui-monospace, monospace' }, `${weight > 0 ? '+' : ''}${weight.toFixed(1)}`);
        group.append(background, dot, text);
        labelLayer.append(group);
        edge.dot = dot;
      }
    }));
  });

  const nodes = layers.map((layer, layerIndex) => layer.map((point, index) => {
    const group = svgElement('g');
    const ring = svgElement('circle', { cx: point.x, cy: point.y, r: 31, fill: PAPER, stroke: INK, 'stroke-width': 1.5 });
    const inner = svgElement('circle', { cx: point.x, cy: point.y, r: 24, fill: PAPER });
    const value = svgElement('text', { x: point.x, y: point.y + 5, 'text-anchor': 'middle', fill: INK, 'font-size': 14, 'font-family': 'ui-monospace, monospace', 'font-weight': 650 });
    const name = `${['x', 'h', 'y'][layerIndex]}${index + 1}`;
    const label = svgElement('text', { x: point.x - 44, y: point.y + 4, 'text-anchor': 'end', fill: MUTED, 'font-size': 11, 'font-family': 'ui-monospace, monospace' }, name);
    group.append(ring, inner, value, label);
    nodeLayer.append(group);
    return { ring, value };
  }));
  [['Inputs', 118], ['Hidden layer / ReLU', 460], ['Linear outputs', 802]].forEach(([text, x]) => {
    svg.append(svgElement('text', { x, y: 365, 'text-anchor': 'middle', fill: MUTED, 'font-size': 13, 'font-family': 'system-ui, sans-serif' }, text));
  });

  function updateReadout() {
    const [a, b, c] = result.inputs;
    const terms = `${fmt(a)} × 0.8 + ${fmt(b)} × −0.4 + ${fmt(c)} × 0.6`;
    const calculation = `${terms} = ${fmt(result.sums[0])}`;
    controls['weight-readout'].textContent = `h1 = ReLU[${terms}] = ${fmt(result.hidden[0])}. Outputs: y1 = ${fmt(result.outputs[0])}, y2 = ${fmt(result.outputs[1])}. No biases. ReLU keeps positive sums and changes negative sums to zero.`;
    controls['signal-value'].textContent = fmt(a);
    controls['signal-input'].setAttribute('aria-valuetext', fmt(a));
    description.textContent = `Three inputs feed four hidden units with ReLU, then two linear outputs. Fixed weights, no biases, no training. First hidden unit: ${calculation}. Outputs ${result.outputs.map(fmt).join(' and ')}. ${colorMode ? 'Colors label the same scalar weights. All calculations still use JavaScript numbers.' : 'Weights appear as ordinary scalar numbers.'} Solid connections are positive; dashed connections are negative.`;
    [result.inputs, result.hidden, result.outputs].forEach((values, layer) => values.forEach((value, index) => {
      nodes[layer][index].value.textContent = fmt(value);
      nodes[layer][index].ring.setAttribute('fill', colorMode ? weightColor(Math.min(1, value)) : PAPER);
    }));
    for (const edge of edges) {
      edge.path.setAttribute('stroke', colorMode ? weightColor(edge.weight) : '#929b92');
      edge.packet.setAttribute('fill', colorMode ? weightColor(edge.weight) : INK);
      if (edge.dot) edge.dot.setAttribute('fill', colorMode ? weightColor(edge.weight) : INK);
    }
    controls['mode-numeric'].setAttribute('aria-pressed', String(!colorMode));
    controls['mode-color'].setAttribute('aria-pressed', String(colorMode));
  }

  function render() {
    const stage = Math.floor(clock / phaseDuration) % 4;
    const progress = (clock % phaseDuration) / phaseDuration;
    if (stage !== lastStage) {
      const captions = [
        `1 / 4 · Input values enter the network. x1 is ${fmt(result.inputs[0])}; x2 and x3 stay fixed.`,
        '2 / 4 · Each connection multiplies its input by a fixed weight. The hidden units sum those products and apply ReLU.',
        '3 / 4 · Hidden values travel through a second set of fixed weights to two linear outputs.',
        `4 / 4 · Output values are ${fmt(result.outputs[0])} and ${fmt(result.outputs[1])}. Changing the display keeps the result identical.`,
      ];
      controls['frame-caption'].textContent = captions[stage];
      lastStage = stage;
    }
    for (const edge of edges) {
      const active = stage === edge.layer + 1;
      edge.path.setAttribute('opacity', active ? '0.9' : '0.42');
      edge.packet.setAttribute('visibility', active ? 'visible' : 'hidden');
      if (!active) continue;
      const t = Math.min(0.97, Math.max(0.03, progress));
      const u = 1 - t;
      const x = u ** 3 * edge.start.x + 3 * u * u * t * (edge.start.x + 110)
        + 3 * u * t * t * (edge.end.x - 110) + t ** 3 * edge.end.x;
      const y = (u ** 3 + 3 * u * u * t) * edge.start.y + (3 * u * t * t + t ** 3) * edge.end.y;
      if (edge.weight < 0) edge.packet.setAttribute('transform', `translate(${x} ${y}) rotate(45) translate(-4.5 -4.5)`);
      else {
        edge.packet.setAttribute('cx', String(x));
        edge.packet.setAttribute('cy', String(y));
      }
    }
  }

  function tick(timestamp) {
    requestId = null;
    if (!playing || document.hidden) return;
    if (previousTime !== null) clock = (clock + Math.min(timestamp - previousTime, 80)) % (phaseDuration * 4);
    previousTime = timestamp;
    render();
    requestId = window.requestAnimationFrame(tick);
  }

  function setPlaying(next) {
    playing = next;
    previousTime = null;
    if (requestId !== null) window.cancelAnimationFrame(requestId);
    requestId = null;
    controls['play-toggle'].textContent = playing ? 'Pause animation' : 'Play animation';
    controls['play-toggle'].setAttribute('aria-pressed', String(playing));
    if (playing && !document.hidden) requestId = window.requestAnimationFrame(tick);
  }

  for (const [id, mode] of [['mode-numeric', false], ['mode-color', true]]) {
    controls[id].addEventListener('click', () => {
      colorMode = mode;
      updateReadout();
      render();
    });
  }
  controls['play-toggle'].addEventListener('click', () => setPlaying(!playing));
  controls['step-animation'].addEventListener('click', () => {
    setPlaying(false);
    clock = ((Math.floor(clock / phaseDuration) + 1) % 4) * phaseDuration + phaseDuration * 0.5;
    render();
  });
  controls['signal-input'].addEventListener('input', () => {
    result = evaluateNetwork(Number(controls['signal-input'].value) / 100);
    lastStage = -1;
    updateReadout();
    render();
  });
  reducedMotion.addEventListener('change', event => {
    if (event.matches) setPlaying(false);
  });
  document.addEventListener('visibilitychange', () => setPlaying(playing));
  // Screen readers can read the caption on demand without repeated announcements.
  controls['frame-caption'].setAttribute('aria-live', 'off');
  updateReadout();
  render();
  setPlaying(!reducedMotion.matches);
}

function initHueDemo() {
  const aInput = document.getElementById('hue-a');
  const bInput = document.getElementById('hue-b');
  const output = document.getElementById('hue-result');
  if (!aInput || !bInput || !output) return;
  const wheel = document.getElementById('hue-wheel');
  const markers = [];
  const tokenCells = ['token-a', 'token-b', 'token-result'].map(id => {
    const element = document.getElementById(id);
    if (!element) return null;
    const swatch = document.createElement('span');
    swatch.className = 'token-swatch';
    swatch.setAttribute('aria-hidden', 'true');
    const bytes = document.createElement('code');
    const label = document.createElement('small');
    element.replaceChildren(swatch, bytes, label);
    return { element, swatch, bytes, label };
  });
  const position = (phase, radius = 112) => {
    const angle = phase / HUE_MODULUS * Math.PI * 2 - Math.PI / 2;
    return { x: 150 + Math.cos(angle) * radius, y: 150 + Math.sin(angle) * radius };
  };
  if (wheel) {
    wheel.replaceChildren();
    wheel.append(svgElement('title', {}, 'Native hue phase circle. A plus B wraps after 391680 ticks.'));
    wheel.append(svgElement('circle', { cx: 150, cy: 150, r: 126, fill: 'none', stroke: '#6d7873', 'stroke-width': 1 }));
    for (let index = 0; index < 180; index += 1) {
      const phase = Math.floor(index * HUE_MODULUS / 180);
      const start = position(phase);
      const end = position(Math.floor((index + 1.1) * HUE_MODULUS / 180));
      const rgb = encodePhase(phase).slice(0, 3);
      wheel.append(svgElement('path', { d: `M${start.x},${start.y} A112,112 0 0 1 ${end.x},${end.y}`, fill: 'none', stroke: `rgb(${rgb.join(' ')})`, 'stroke-width': 17 }));
    }
    wheel.append(svgElement('text', { x: 150, y: 143, 'text-anchor': 'middle', fill: 'currentColor', 'font-family': 'ui-monospace, monospace', 'font-size': 22, 'font-weight': 650 }, '391,680'));
    wheel.append(svgElement('text', { x: 150, y: 166, 'text-anchor': 'middle', fill: 'currentColor', 'font-family': 'system-ui, sans-serif', 'font-size': 12 }, 'phase ticks / circle'));
    ['A', 'B', '='].forEach((label, index) => {
      const marker = svgElement('g');
      marker.append(svgElement(index === 2 ? 'rect' : 'circle', index === 2
        ? { x: -11, y: -11, width: 22, height: 22, rx: 3, fill: PAPER, stroke: INK, 'stroke-width': 2 }
        : { cx: 0, cy: 0, r: 11, fill: PAPER, stroke: INK, 'stroke-width': 2 }));
      marker.append(svgElement('text', { x: 0, y: 4, 'text-anchor': 'middle', fill: INK, 'font-family': 'system-ui, sans-serif', 'font-size': 11, 'font-weight': 700 }, label));
      wheel.append(marker);
      markers.push(marker);
    });
  }
  function update() {
    const a = Number(aInput.value);
    const b = Number(bInput.value);
    const sum = (a + b) % HUE_MODULUS;
    const phases = [a, b, sum];
    tokenCells.forEach((cell, index) => {
      if (!cell) return;
      const token = encodePhase(phases[index]);
      const color = `rgb(${token.slice(0, 3).join(' ')})`;
      cell.bytes.textContent = `[${token.join(', ')}]`;
      cell.label.textContent = `${['A', 'B', 'Result'][index]} / phase ${phases[index].toLocaleString('en-US')}`;
      cell.element.style.setProperty('--token-color', color);
      cell.swatch.style.backgroundColor = color;
    });
    for (const [id, value] of [['hue-a-value', a], ['hue-b-value', b]]) {
      const element = document.getElementById(id);
      if (element) element.textContent = value.toLocaleString('en-US');
    }
    aInput.setAttribute('aria-valuetext', `${a} phase ticks`);
    bInput.setAttribute('aria-valuetext', `${b} phase ticks`);
    output.textContent = `${a.toLocaleString('en-US')} + ${b.toLocaleString('en-US')} ≡ ${sum.toLocaleString('en-US')} mod 391,680`;
    markers.forEach((marker, index) => {
      // Separate marker tracks keep A, B and the result visible at equal phases.
      const point = position(phases[index], [92, 112, 136][index]);
      marker.setAttribute('transform', `translate(${point.x} ${point.y})`);
    });
  }
  aInput.addEventListener('input', update);
  bInput.addEventListener('input', update);
  update();
}

if (typeof document !== 'undefined') {
  initNetwork();
  initHueDemo();
}
