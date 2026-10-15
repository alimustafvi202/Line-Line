/**
 * CardioAI Web Application Logic
 * Real-Time Continuous ECG Oscilloscope, Beat Inspector, and AI Diagnostics
 */

// Application State
const state = {
  currentRecord: '100',
  currentModel: 'improved_1d_cnn',
  signalData: [],
  rPeaks: [],
  samplingRate: 360,
  isPlaying: true,
  playbackSpeed: 1.0,
  playHead: 0,
  animationId: null,
  cachedInference: null,
  selectedSampleBeat: 0
};

// Color Palette for AAMI Classes
const AAMI_COLORS = {
  'N': '#3b82f6', // Blue
  'S': '#f59e0b', // Amber
  'V': '#ef4444', // Red
  'F': '#10b981', // Green
  'Q': '#a855f7'  // Purple
};

// DOM Elements
const recordSelect = document.getElementById('recordSelect');
const modelSelect = document.getElementById('modelSelect');
const runAiBtn = document.getElementById('runAiBtn');
const playPauseBtn = document.getElementById('playPauseBtn');
const resetPlayBtn = document.getElementById('resetPlayBtn');
const speedButtons = document.querySelectorAll('.speed-btn');
const tabButtons = document.querySelectorAll('.tab-btn');

const ecgCanvas = document.getElementById('ecgCanvas');
const beatCanvas = document.getElementById('beatCanvas');
const ecgCtx = ecgCanvas.getContext('2d');
const beatCtx = beatCanvas.getContext('2d');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
  setupCanvasScaling();
  setupEventListeners();
  await loadRecordsAndModels();
  await loadRecordSignal(state.currentRecord);
  await loadBenchmarks();
  startOscilloscope();
});

function setupCanvasScaling() {
  // Handle high-DPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = ecgCanvas.getBoundingClientRect();
  ecgCanvas.width = rect.width * dpr;
  ecgCanvas.height = rect.height * dpr;
  ecgCtx.scale(dpr, dpr);
}

function setupEventListeners() {
  // Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  // Record Selection
  recordSelect.addEventListener('change', async (e) => {
    state.currentRecord = e.target.value;
    state.playHead = 0;
    await loadRecordSignal(state.currentRecord);
  });

  // Model Selection
  modelSelect.addEventListener('change', (e) => {
    state.currentModel = e.target.value;
  });

  // Run AI Diagnostics Button
  runAiBtn.addEventListener('click', async () => {
    await runInference();
  });

  // Playback Controls
  playPauseBtn.addEventListener('click', () => {
    state.isPlaying = !state.isPlaying;
    playPauseBtn.innerHTML = state.isPlaying ? '⏸ Pause' : '▶ Play';
  });

  resetPlayBtn.addEventListener('click', () => {
    state.playHead = 0;
  });

  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.playbackSpeed = parseFloat(btn.dataset.speed);
    });
  });
}

// --- API Data Fetching ---

async function loadRecordsAndModels() {
  try {
    const [recRes, modelRes] = await Promise.all([
      fetch('/api/records'),
      fetch('/api/models')
    ]);
    const recData = await recRes.json();
    const modelData = await modelRes.json();

    // Populate Records dropdown
    recordSelect.innerHTML = '';
    recData.records.forEach(rec => {
      const opt = document.createElement('option');
      opt.value = rec;
      opt.textContent = `Patient ${rec}`;
      if (rec === '100') opt.selected = true;
      recordSelect.appendChild(opt);
    });

    // Populate Models dropdown
    modelSelect.innerHTML = '';
    modelData.models.forEach(model => {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = model.replace(/_/g, ' ').toUpperCase();
      if (model === 'improved_1d_cnn') opt.selected = true;
      modelSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load initial metadata:', err);
  }
}

async function loadRecordSignal(recordId) {
  try {
    const res = await fetch(`/api/record-signal?record=${recordId}`);
    const data = await res.json();
    state.signalData = data.signal || [];
    state.rPeaks = data.beats || [];
    state.samplingRate = data.sampling_rate || 360;

    document.getElementById('leadDisplay').textContent = `Lead ${data.lead}`;
    document.getElementById('fsDisplay').textContent = `${data.sampling_rate} Hz`;
    document.getElementById('totalBeatsDisplay').textContent = `${data.total_record_beats.toLocaleString()} Beats`;
    document.getElementById('kpiStatusSub').textContent = `Patient ${recordId} loaded`;
  } catch (err) {
    console.error(`Error loading signal for record ${recordId}:`, err);
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    const data = await res.json();
    const tbody = document.getElementById('benchmarkBody');
    tbody.innerHTML = '';

    data.benchmarks.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row.Experiment || '--'}</strong></td>
        <td>${row.Dataset_Partition || '--'}</td>
        <td><code>${row.Train_Records || '--'}</code></td>
        <td><code>${row.Test_Records || '--'}</code></td>
        <td><span class="stat-val">${row.Overall_Accuracy || '--'}</span></td>
        <td><span class="stat-val highlight">${row.Macro_F1 || '--'}</span></td>
        <td><span class="stat-val ${parseFloat(row.Class_V_Recall) > 0 ? 'success' : ''}">${row.Class_V_Recall || '--'}</span></td>
        <td>${row.Class_N_F1 || '--'}</td>
        <td><small>${row.Notes || '--'}</small></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

// --- Live Oscilloscope Canvas Animation ---

function startOscilloscope() {
  const dpr = window.devicePixelRatio || 1;
  const width = ecgCanvas.width / dpr;
  const height = ecgCanvas.height / dpr;

  function render() {
    if (!ecgCtx) return;

    // 1. Draw Medical Grid Background
    drawMedicalGrid(ecgCtx, width, height);

    if (state.signalData.length > 0) {
      const windowSamples = state.samplingRate * 4; // Display 4 seconds at a time
      const startIndex = Math.floor(state.playHead);
      const endIndex = startIndex + windowSamples;

      // Extract current slice
      let slice = state.signalData.slice(startIndex, endIndex);
      if (slice.length < windowSamples) {
        // Wrap around loop
        state.playHead = 0;
        slice = state.signalData.slice(0, windowSamples);
      }

      // 2. Plot Glowing ECG Waveform
      ecgCtx.save();
      ecgCtx.strokeStyle = '#10b981';
      ecgCtx.lineWidth = 2.2;
      ecgCtx.shadowColor = '#10b981';
      ecgCtx.shadowBlur = 10;
      ecgCtx.beginPath();

      const centerY = height / 2;
      const yScale = height / 7; // Scale z-score signal to canvas height

      for (let i = 0; i < slice.length; i++) {
        const x = (i / windowSamples) * width;
        const y = centerY - (slice[i] * yScale);
        if (i === 0) ecgCtx.moveTo(x, y);
        else ecgCtx.lineTo(x, y);
      }
      ecgCtx.stroke();
      ecgCtx.restore();

      // 3. Draw R-Peak Annotation Markers
      state.rPeaks.forEach(beat => {
        if (beat.sample >= startIndex && beat.sample <= endIndex) {
          const beatX = ((beat.sample - startIndex) / windowSamples) * width;
          const beatY = centerY - (state.signalData[beat.sample] * yScale);
          const color = AAMI_COLORS[beat.class_name] || '#3b82f6';

          // Glowing dot at R-peak
          ecgCtx.save();
          ecgCtx.fillStyle = color;
          ecgCtx.beginPath();
          ecgCtx.arc(beatX, beatY, 4.5, 0, Math.PI * 2);
          ecgCtx.fill();

          // Class badge text
          ecgCtx.font = 'bold 11px JetBrains Mono';
          ecgCtx.fillText(beat.class_name, beatX - 4, beatY - 10);
          ecgCtx.restore();
        }
      });

      // 4. Update Playhead
      if (state.isPlaying) {
        state.playHead += (state.samplingRate / 60) * state.playbackSpeed;
      }
    }

    state.animationId = requestAnimationFrame(render);
  }

  render();
}

function drawMedicalGrid(ctx, width, height) {
  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, width, height);

  // Minor grid lines
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.06)';
  ctx.lineWidth = 1;
  const minorStep = 18;
  for (let x = 0; x < width; x += minorStep) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += minorStep) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Major grid lines
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.14)';
  ctx.lineWidth = 1.2;
  const majorStep = minorStep * 5;
  for (let x = 0; x < width; x += majorStep) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += majorStep) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Center zero line
  ctx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

// --- Run AI Diagnostics & Update UI ---

async function runInference() {
  runAiBtn.disabled = true;
  runAiBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: jsonBody({ record: state.currentRecord, model: state.currentModel })
    });
    const data = await res.json();
    state.cachedInference = data;

    // 1. Update KPI Cards
    document.getElementById('kpiAccuracy').textContent = `${(data.accuracy * 100).toFixed(2)}%`;
    document.getElementById('kpiAccuracySub').textContent = `Correct: ${(data.accuracy * data.total_beats).toFixed(0)} / ${data.total_beats}`;

    document.getElementById('kpiMacroF1').textContent = data.macro_f1.toFixed(4);
    
    const vRecall = (data.per_class.V?.recall_sensitivity * 100) || 0;
    document.getElementById('kpiVRecall').textContent = `${vRecall.toFixed(1)}%`;

    const statusEl = document.getElementById('kpiStatus');
    if (vRecall === 100) {
      statusEl.textContent = 'VEB CAPTURED';
      statusEl.className = 'kpi-value success';
    } else if (vRecall > 0) {
      statusEl.textContent = 'PARTIAL VEB';
      statusEl.className = 'kpi-value alert';
    } else {
      statusEl.textContent = 'VEB MISSED';
      statusEl.className = 'kpi-value alert';
    }

    // 2. Populate Confusion Matrix
    renderConfusionMatrix(data.confusion_matrix);

    // 3. Populate Sample Beat Selector Pills
    renderBeatSelector(data.sample_beats);

  } catch (err) {
    console.error('Inference error:', err);
    alert('Failed to run AI inference. Check server logs.');
  } finally {
    runAiBtn.disabled = false;
    runAiBtn.innerHTML = '<span class="btn-icon">⚡</span> Run AI Diagnostics';
  }
}

function jsonBody(obj) {
  return JSON.stringify(obj);
}

function renderConfusionMatrix(matrix) {
  const tbody = document.getElementById('matrixBody');
  tbody.innerHTML = '';
  const classes = ['N', 'S', 'V', 'F', 'Q'];

  matrix.forEach((row, i) => {
    const tr = document.createElement('tr');
    let rowHtml = `<td><strong>${classes[i]}</strong></td>`;
    row.forEach((val, j) => {
      const isDiag = (i === j);
      const cellClass = isDiag ? 'cell-diag' : (val > 0 ? 'cell-offdiag' : '');
      rowHtml += `<td class="${cellClass}">${val.toLocaleString()}</td>`;
    });
    tr.innerHTML = rowHtml;
    tbody.appendChild(tr);
  });
}

function renderBeatSelector(sampleBeats) {
  const bar = document.getElementById('beatSelectorBar');
  bar.innerHTML = '';

  sampleBeats.forEach((beat, idx) => {
    const btn = document.createElement('button');
    btn.className = `beat-pill ${idx === 0 ? 'active' : ''}`;
    btn.innerHTML = `#${beat.beat_index} (${beat.true_class} ➜ ${beat.pred_class})`;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.beat-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      inspectSingleBeat(beat);
    });
    bar.appendChild(btn);
  });

  if (sampleBeats.length > 0) {
    inspectSingleBeat(sampleBeats[0]);
  }
}

function inspectSingleBeat(beat) {
  document.getElementById('detailPredClass').textContent = `Class ${beat.pred_class}`;
  document.getElementById('detailTrueClass').textContent = `Class ${beat.true_class}`;

  // Update confidence bars
  const barsContainer = document.getElementById('confidenceBars');
  barsContainer.innerHTML = '';

  const classes = ['N', 'S', 'V'];
  classes.forEach(cls => {
    const prob = (beat.probabilities[cls] || 0) * 100;
    const item = document.createElement('div');
    item.className = 'conf-item';
    item.innerHTML = `
      <span class="conf-class">Class ${cls}:</span>
      <div class="bar-container"><div class="bar-fill" style="width: ${prob.toFixed(1)}%"></div></div>
      <span class="conf-pct">${prob.toFixed(1)}%</span>
    `;
    barsContainer.appendChild(item);
  });

  // Render single-beat waveform on beatCanvas
  drawSingleBeatWaveform(beat.waveform, beat.true_class, beat.pred_class);
}

function drawSingleBeatWaveform(waveform, trueCls, predCls) {
  const width = beatCanvas.width;
  const height = beatCanvas.height;

  beatCtx.fillStyle = '#020617';
  beatCtx.fillRect(0, 0, width, height);

  // Subtle grid
  beatCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  beatCtx.lineWidth = 1;
  for (let x = 0; x < width; x += 30) {
    beatCtx.beginPath();
    beatCtx.moveTo(x, 0);
    beatCtx.lineTo(x, height);
    beatCtx.stroke();
  }
  for (let y = 0; y < height; y += 30) {
    beatCtx.beginPath();
    beatCtx.moveTo(0, y);
    beatCtx.lineTo(width, y);
    beatCtx.stroke();
  }

  // R-peak reference line (sample 90 out of 200)
  const rPeakX = (90 / 200) * width;
  beatCtx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
  beatCtx.setLineDash([4, 4]);
  beatCtx.beginPath();
  beatCtx.moveTo(rPeakX, 0);
  beatCtx.lineTo(rPeakX, height);
  beatCtx.stroke();
  beatCtx.setLineDash([]);

  // Plot waveform
  beatCtx.save();
  const isMatch = (trueCls === predCls);
  beatCtx.strokeStyle = isMatch ? '#10b981' : '#ef4444';
  beatCtx.lineWidth = 2.5;
  beatCtx.shadowColor = isMatch ? '#10b981' : '#ef4444';
  beatCtx.shadowBlur = 8;
  beatCtx.beginPath();

  const centerY = height / 2;
  const scale = height / 5;

  for (let i = 0; i < waveform.length; i++) {
    const x = (i / waveform.length) * width;
    const y = centerY - (waveform[i] * scale);
    if (i === 0) beatCtx.moveTo(x, y);
    else beatCtx.lineTo(x, y);
  }
  beatCtx.stroke();
  beatCtx.restore();
}
