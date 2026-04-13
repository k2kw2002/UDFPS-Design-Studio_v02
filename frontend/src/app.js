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

    // Fingerprint views
    if (typeof drawFingerprintOriginal === 'function') {
      drawFingerprintOriginal('fp-original');
    }
    if (typeof drawFingerprint === 'function') {
      var metrics = {
        mtf_ridge: state.mtf,
        skewness: state.skewness,
        crosstalk_ratio: 0.1,
      };
      drawFingerprint('fp-current', state.psf_7, metrics, {
        showGrid: true, showInfo: true,
        label: 'Current', labelColor: '#90caf9'
      });

      // Best inverse result
      var bestContainer = document.getElementById('fp-best-container');
      if (state.inverse && state.inverse.best) {
        bestContainer.style.display = '';
        var b = state.inverse.best;
        var bestMetrics = {
          mtf_ridge: b.mtf,
          skewness: b.skewness,
          crosstalk_ratio: 0.1,
        };
        drawFingerprint('fp-best', b.psf_7, bestMetrics, {
          showGrid: true, showInfo: true,
          label: 'Best', labelColor: '#a5d6a7'
        });
      } else {
        bestContainer.style.display = 'none';
      }
    }

    var psfCanvas = document.getElementById('psf-canvas');
    drawPsfBar(psfCanvas, state.psf_7);

    if (state.amp_z40 && state.x_coords) {
      drawAmplitudeProfile(
        document.getElementById('profile-z40-canvas'),
        state.amp_z40, state.x_coords,
        { label: '|A| at z=40', d: state.d1, w: state.w1, vmax: 1.0 }
      );
    }
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
      });
    });
  }

  // ============ Inverse search ============
  function runInverse() {
    var btn = document.getElementById('inverse-run-btn');
    btn.disabled = true;
    btn.textContent = 'Searching...';

    fetchInverse({
      target_skewness: parseFloat(document.getElementById('target-skewness').value) || 0,
      target_mtf: parseFloat(document.getElementById('target-mtf').value) || 0.8,
      target_throughput: 0.5,
      n_trials: parseInt(document.getElementById('n-trials').value) || 50,
    }).then(function (r) {
      state.inverse = r;
      switchTab('explore');
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
  }

  document.addEventListener('DOMContentLoaded', init);
})();
