/**
 * app.js - UDFPS Design Studio Main Controller
 * 3-tab SPA: Summary / Detail / Explore
 * Backend: /api/inverse/predict, /api/inverse/amplitude_map, /api/inverse/search
 */
(function () {
  'use strict';

  var BASE = '';

  // ============ State ============
  var state = {
    d1: 0, d2: 0, w1: 10, w2: 10, theta_deg: 0,
    psf_7: null, skewness: null, mtf: null, throughput: null,
    amp_z40: null, int_z0: null, x_coords: null,
    amp_map: null,
    inverse: null,
    fingerprint: null,
    activeTab: 'summary',
  };

  // ============ API ============
  function apiPost(path, body) {
    return fetch(BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(function (r) { return r.json(); });
  }

  function fetchPredict() {
    return apiPost('/api/inverse/predict', {
      d1: state.d1, d2: state.d2, w1: state.w1, w2: state.w2,
      theta_deg: state.theta_deg,
    });
  }

  function fetchAmpMap() {
    return apiPost('/api/inverse/amplitude_map', {
      d1: state.d1, d2: state.d2, w1: state.w1, w2: state.w2,
      theta_deg: state.theta_deg,
    });
  }

  function fetchInverse(params) {
    return apiPost('/api/inverse/search', params);
  }

  function fetchFingerprint() {
    return apiPost('/api/inverse/fingerprint_sim', {
      d1: state.d1, d2: state.d2, w1: state.w1, w2: state.w2,
      theta_deg: state.theta_deg,
    });
  }

  // ============ Fingerprint simulation ============
  var fpTimer = null;
  function triggerFingerprint() {
    if (fpTimer) clearTimeout(fpTimer);
    fpTimer = setTimeout(function () {
      // Current design
      fetchFingerprint().then(function (r) {
        if (r.error) { console.warn('FP error:', r.error); return; }
        state.fingerprint = r;
        if (state.activeTab === 'summary') renderFingerprint();
      }).catch(function (e) { console.error('FP error:', e); });
    }, 600);
  }

  function renderFingerprint() {
    var fp = state.fingerprint;
    if (!fp) return;

    // Original (always same)
    var origEl = document.getElementById('fp-original');
    if (origEl && fp.original_image) {
      origEl.innerHTML = '<img src="data:image/png;base64,' + fp.original_image + '"/>';
    }

    // Current design
    var curEl = document.getElementById('fp-current');
    if (curEl && fp.processed_image) {
      curEl.innerHTML = '<img src="data:image/png;base64,' + fp.processed_image + '"/>';
    }
    var curInfo = document.getElementById('fp-current-info');
    if (curInfo) {
      curInfo.textContent = 'MTF ' + (fp.mtf * 100).toFixed(1) + '%';
    }

    // Best (inverse) — if available
    var bestBox = document.getElementById('fp-best-box');
    if (state.inverse && state.inverse.best) {
      bestBox.style.display = '';
    }
  }

  function renderBestFingerprint() {
    if (!state.inverse || !state.inverse.best) return;
    var b = state.inverse.best;
    // Fetch fingerprint for best candidate
    apiPost('/api/inverse/fingerprint_sim', {
      d1: b.d1, d2: b.d2, w1: b.w1, w2: b.w2, theta_deg: 0
    }).then(function (r) {
      if (r.error) return;
      var bestEl = document.getElementById('fp-best');
      if (bestEl) bestEl.innerHTML = '<img src="data:image/png;base64,' + r.processed_image + '"/>';
      var bestInfo = document.getElementById('fp-best-info');
      if (bestInfo) {
        bestInfo.textContent = 'MTF ' + (r.mtf * 100).toFixed(1) + '% | ' +
          'd1=' + b.d1.toFixed(1) + ' w1=' + b.w1.toFixed(1);
      }
      document.getElementById('fp-best-box').style.display = '';
    });
  }

  // ============ Debounced predict ============
  var predictTimer = null;
  function triggerPredict() {
    if (predictTimer) clearTimeout(predictTimer);
    predictTimer = setTimeout(function () {
      fetchPredict().then(function (r) {
        state.psf_7 = r.psf_7;
        state.skewness = r.skewness;
        state.mtf = r.mtf;
        state.throughput = r.throughput;
        state.amp_z40 = r.amplitude_profile_z40;
        state.int_z0 = r.intensity_profile_z0;
        state.x_coords = r.x_coords;
        renderTab();

        if (state.activeTab === 'detail') {
          fetchAmpMap().then(function (m) {
            state.amp_map = m.amplitude;
            renderDetail();
          });
        }
      }).catch(function (e) { console.error('Predict error:', e); });
    }, 200);
  }

  // ============ Render ============
  function renderTab() {
    if (state.activeTab === 'summary') renderSummary();
    else if (state.activeTab === 'detail') renderDetail();
    else if (state.activeTab === 'explore') renderExplore();
  }

  function renderSummary() {
    if (!state.psf_7) return;

    drawMetricsCard(document.getElementById('metrics-container'), {
      skewness: state.skewness, mtf: state.mtf, throughput: state.throughput,
    });

    renderFingerprint();

    var psfCanvas = document.getElementById('psf-canvas');
    if (psfCanvas) drawPsfBar(psfCanvas, state.psf_7);
  }

  function renderDetail() {
    if (!state.amp_z40) return;

    if (state.amp_map) {
      drawAmplitudeMap(
        document.getElementById('amp-map-canvas'),
        state.amp_map, { vmax: 0.85 }
      );
    }

    if (state.x_coords) {
      drawAmplitudeProfile(
        document.getElementById('detail-z40-canvas'),
        state.amp_z40, state.x_coords,
        { label: '|A| at z=40 (BM1)', d: state.d1, w: state.w1, vmax: 1.0 }
      );

      drawAmplitudeProfile(
        document.getElementById('detail-z0-canvas'),
        state.int_z0, state.x_coords,
        { label: '|A|^2 at z=0 (OPD)', color: '#2E7D32',
          fill: 'rgba(46,125,50,0.12)', d: 0, w: 10 }
      );
    }
  }

  function renderExplore() {
    var inv = state.inverse;
    if (!inv) return;

    document.getElementById('explore-status').style.display = 'none';
    document.getElementById('pareto-container').style.display = 'block';

    drawParetoScatter(document.getElementById('pareto-canvas'), inv.all_trials, inv.pareto);

    var container = document.getElementById('candidates-container');
    container.innerHTML = '';
    inv.pareto.slice(0, 6).forEach(function (c, i) {
      container.appendChild(createCandidateCard(c, i));
    });
  }

  // ============ PSF Bar chart ============
  function drawPsfBar(canvas, psf) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    var m = { top: 20, bottom: 30, left: 50, right: 20 };
    var pW = W - m.left - m.right;
    var pH = H - m.top - m.bottom;
    var n = psf.length;
    var barW = pW / n * 0.6;
    var gap = pW / n;
    var maxV = Math.max.apply(null, psf) * 1.15;
    if (maxV < 0.01) maxV = 1;

    for (var i = 0; i < n; i++) {
      var bh = (psf[i] / maxV) * pH;
      var x = m.left + i * gap + (gap - barW) / 2;
      var y = m.top + pH - bh;
      ctx.fillStyle = (i === 3) ? '#1428A0' : '#6B8CCE';
      ctx.fillRect(x, y, barW, bh);
      ctx.fillStyle = '#555';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(psf[i].toFixed(3), x + barW / 2, y - 4);
      ctx.fillText('OPD' + i, x + barW / 2, m.top + pH + 14);
    }

    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(m.left, m.top + pH);
    ctx.lineTo(m.left + pW, m.top + pH);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('PSF Intensity', m.left, m.top - 4);
  }

  // ============ Pareto scatter ============
  function drawParetoScatter(canvas, all, pareto) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    var m = { top: 20, bottom: 36, left: 56, right: 20 };
    var pW = W - m.left - m.right;
    var pH = H - m.top - m.bottom;

    var skewMax = 0.5, mtfMax = 1.0;

    function xPx(s) { return m.left + (s / skewMax) * pW; }
    function yPx(v) { return m.top + pH - (v / mtfMax) * pH; }

    ctx.fillStyle = 'rgba(150,150,150,0.3)';
    all.forEach(function (c) {
      ctx.beginPath();
      ctx.arc(xPx(c.skewness), yPx(c.mtf), 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = '#1428A0';
    pareto.forEach(function (c) {
      ctx.beginPath();
      ctx.arc(xPx(c.skewness), yPx(c.mtf), 6, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.strokeStyle = '#888';
    ctx.beginPath();
    ctx.moveTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH);
    ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH);
    ctx.stroke();

    ctx.fillStyle = '#555';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Skewness (lower better)', m.left + pW / 2, H - 6);
    ctx.save();
    ctx.translate(14, m.top + pH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('MTF (higher better)', 0, 0);
    ctx.restore();
  }

  // ============ Top 5 with fingerprint + stack ============
  function renderTop5() {
    if (!state.inverse || !state.inverse.pareto) return;
    var section = document.getElementById('top5-section');
    var container = document.getElementById('top5-container');
    if (!section || !container) return;

    var top5 = state.inverse.pareto.slice(0, 5);
    if (top5.length === 0) return;

    section.style.display = '';
    container.innerHTML = '';

    top5.forEach(function (c, i) {
      var card = document.createElement('div');
      card.className = 'top5-card' + (i === 0 ? ' best' : '');
      card.innerHTML =
        '<div class="top5-rank">#' + (i + 1) + '</div>' +
        '<div class="top5-fp" id="top5-fp-' + i + '"></div>' +
        '<div class="top5-metrics">' +
          'MTF <span class="val">' + (c.mtf * 100).toFixed(1) + '%</span><br>' +
          'Skew <span class="val">' + c.skewness.toFixed(3) + '</span><br>' +
          'Thru <span class="val">' + c.throughput.toFixed(1) + '</span>' +
        '</div>' +
        '<div class="top5-stack">' + renderStackMini(c) + '</div>';

      card.addEventListener('click', function () {
        state.d1 = c.d1; state.d2 = c.d2;
        state.w1 = c.w1; state.w2 = c.w2;
        updateSliderDisplays();
        triggerPredict();
        triggerFingerprint();
      });

      container.appendChild(card);

      // Fetch fingerprint for each candidate
      apiPost('/api/inverse/fingerprint_sim', {
        d1: c.d1, d2: c.d2, w1: c.w1, w2: c.w2, theta_deg: 0
      }).then(function (r) {
        if (r.error) return;
        var el = document.getElementById('top5-fp-' + i);
        if (el) el.innerHTML = '<img src="data:image/png;base64,' + r.processed_image + '"/>';
      });
    });
  }

  function renderStackMini(c) {
    var maxW = 20;
    function bar(w, color) {
      var pct = Math.max(10, (w / maxW) * 100);
      return '<div class="stack-bar" style="width:' + pct + '%;background:' + color + '"></div>';
    }
    return '' +
      '<div class="stack-layer">' +
        '<span class="stack-label">BM1</span>' +
        bar(c.w1, '#2563eb') +
        '<span class="stack-val">w=' + c.w1.toFixed(1) + ' d=' + c.d1.toFixed(1) + '</span>' +
      '</div>' +
      '<div class="stack-layer">' +
        '<span class="stack-label">ILD</span>' +
        '<div class="stack-bar" style="width:100%;background:#e2e8f0;height:3px"></div>' +
        '<span class="stack-val">20um</span>' +
      '</div>' +
      '<div class="stack-layer">' +
        '<span class="stack-label">BM2</span>' +
        bar(c.w2, '#7c3aed') +
        '<span class="stack-val">w=' + c.w2.toFixed(1) + ' d=' + c.d2.toFixed(1) + '</span>' +
      '</div>';
  }

  // ============ Candidate card ============
  function createCandidateCard(c, rank) {
    var div = document.createElement('div');
    div.className = 'candidate-card';
    div.innerHTML =
      '<div class="rank">#' + (rank + 1) + '</div>' +
      '<div class="params">' +
      'd1=' + c.d1.toFixed(1) + ', d2=' + c.d2.toFixed(1) + '<br>' +
      'w1=' + c.w1.toFixed(1) + ', w2=' + c.w2.toFixed(1) +
      '</div>' +
      '<div class="metrics">' +
      'Skew: ' + c.skewness.toFixed(4) + '<br>' +
      'MTF: ' + c.mtf.toFixed(4) + '<br>' +
      'Thru: ' + c.throughput.toFixed(3) +
      '</div>' +
      '<button class="load-btn" data-rank="' + rank + '">Load</button>';

    div.querySelector('.load-btn').addEventListener('click', function () {
      loadCandidate(rank);
    });
    return div;
  }

  function loadCandidate(rank) {
    var c = state.inverse.pareto[rank];
    state.d1 = c.d1; state.d2 = c.d2;
    state.w1 = c.w1; state.w2 = c.w2;
    updateSliderDisplays();
    switchTab('summary');
    triggerPredict();
  }

  // ============ Tab switching ============
  function switchTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll('.tab-button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-panel').forEach(function (p) {
      p.style.display = (p.dataset.tab === tab) ? 'block' : 'none';
    });
    renderTab();

    if (tab === 'detail' && !state.amp_map) {
      fetchAmpMap().then(function (m) {
        state.amp_map = m.amplitude;
        renderDetail();
      });
    }
  }

  // ============ Sliders ============
  var sliderKeys = ['d1', 'd2', 'w1', 'w2', 'theta_deg'];

  function updateSliderDisplays() {
    sliderKeys.forEach(function (key) {
      var input = document.getElementById('slider-' + key);
      var display = document.getElementById('value-' + key);
      if (input) input.value = state[key];
      if (display) display.textContent = state[key].toFixed(1);
    });
  }

  function wireSliders() {
    sliderKeys.forEach(function (key) {
      var input = document.getElementById('slider-' + key);
      if (!input) return;
      input.addEventListener('input', function () {
        state[key] = parseFloat(input.value);
        document.getElementById('value-' + key).textContent = state[key].toFixed(1);
        state.amp_map = null;
        triggerPredict();
        triggerFingerprint();
      });
    });
  }

  // ============ Inverse search ============
  function runInverse() {
    var btn = document.getElementById('inverse-run-btn');
    btn.disabled = true;
    btn.textContent = 'Searching...';

    fetchInverse({
      target_skewness: 0.0,
      target_mtf: parseFloat((document.getElementById('target-mtf') || {}).value || 0.8),
      target_throughput: 0.5,
      n_trials: parseInt((document.getElementById('n-trials') || {}).value || 50),
    }).then(function (r) {
      state.inverse = r;
      renderTab();
      renderBestFingerprint();
      renderTop5();
    }).catch(function (e) {
      console.error('Inverse error:', e);
      document.getElementById('explore-status').textContent = 'Error: ' + e.message;
    }).finally(function () {
      btn.disabled = false;
      btn.textContent = 'Find Optimal';
    });
  }

  // ============ Init ============
  function init() {
    document.querySelectorAll('.tab-button').forEach(function (btn) {
      btn.addEventListener('click', function () { switchTab(btn.dataset.tab); });
    });

    wireSliders();
    document.getElementById('inverse-run-btn').addEventListener('click', runInverse);
    triggerPredict();
    triggerFingerprint();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
