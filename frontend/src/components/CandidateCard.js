/**
 * CandidateCard.js - 후보 카드 컴포넌트
 * 가이드 12.7절 후보 카드 구성.
 * 각 카드: KPI + 설계변수 + COE 스택 + PSF 바 차트 + LT 내보내기
 */

/** KPI 색상 클래스 반환 (가이드 12.5) */
function getKpiClass(metric, value) {
  switch (metric) {
    case 'mtf':
      return value >= 0.60 ? 'kpi-green' : value >= 0.45 ? 'kpi-yellow' : 'kpi-red';
    case 'skewness':
      return value <= 0.10 ? 'kpi-green' : value <= 0.20 ? 'kpi-yellow' : 'kpi-red';
    case 'throughput':
      return value >= 0.60 ? 'kpi-green' : value >= 0.45 ? 'kpi-yellow' : 'kpi-red';
    case 'crosstalk':
      return value <= 0.10 ? 'kpi-green' : value <= 0.20 ? 'kpi-yellow' : 'kpi-red';
    default: return '';
  }
}

function deltaStr(current, baseline) {
  const diff = current - baseline;
  const pp = (diff * 100).toFixed(0);
  if (diff > 0) return `<span style="color:#22c55e">+${pp}pp</span>`;
  if (diff < 0) return `<span style="color:#ef4444">${pp}pp</span>`;
  return '<span style="color:#8b90a0">0pp</span>';
}

function deltaSkew(current, baseline) {
  const icon = current < baseline ? '\u25BC' : current > baseline ? '\u25B2' : '';
  const color = current <= baseline ? '#22c55e' : '#ef4444';
  return `<span style="color:${color}">${icon}</span>`;
}

/** 카드 HTML 생성 — 스택 + PSF 캔버스 포함 */
function renderCandidateCard(candidate, baseline, isSelected, index, prefix) {
  const c = candidate;
  const b = baseline || { mtf: 0.42, skew: 0.28, T: 0.48 };
  const idx = (prefix || '') + (index || 0);

  return `
    <div class="candidate-card ${isSelected ? 'selected' : ''}" data-id="${c.id}" data-idx="${idx}">
      <div class="card-top-row">
        <div class="card-info">
          <div class="card-header">
            <span class="card-label">${c.label}</span>
            <span class="card-rank">Pareto #${c.pareto_rank}</span>
          </div>
          <div class="card-metrics">
            <div>
              <span class="card-metric-label">MTF</span>
              <span class="card-metric-value ${getKpiClass('mtf', c.mtf_ridge)}">${(c.mtf_ridge * 100).toFixed(0)}%</span>
              <span class="card-metric-delta">${deltaStr(c.mtf_ridge, b.mtf)}</span>
            </div>
            <div>
              <span class="card-metric-label">Skew</span>
              <span class="card-metric-value ${getKpiClass('skewness', c.skewness)}">${c.skewness.toFixed(3)}</span>
              <span class="card-metric-delta">${deltaSkew(c.skewness, b.skew)}</span>
            </div>
            <div>
              <span class="card-metric-label">T</span>
              <span class="card-metric-value ${getKpiClass('throughput', c.throughput)}">${(c.throughput * 100).toFixed(0)}%</span>
              <span class="card-metric-delta">${deltaStr(c.throughput, b.T)}</span>
            </div>
            <div>
              <span class="card-metric-label">XT</span>
              <span class="card-metric-value ${getKpiClass('crosstalk', c.crosstalk_ratio)}">${(c.crosstalk_ratio * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div class="card-params">
            \u03B4\u2081=${c.params.delta_bm1.toFixed(1)}  \u03B4\u2082=${c.params.delta_bm2.toFixed(1)}  w\u2081=${c.params.w1.toFixed(1)}  w\u2082=${c.params.w2.toFixed(1)}
          </div>
        </div>
        <div class="card-stack">
          <canvas id="card-stack-${idx}" width="70" height="100"></canvas>
        </div>
      </div>
      <div class="card-psf-row">
        <canvas id="card-psf-${idx}" width="230" height="60"></canvas>
      </div>
      <div class="card-export">
        <button onclick="event.stopPropagation(); exportCandidate('${c.id}')">LightTools Export \u2192</button>
      </div>
    </div>
  `;
}

/** 미니 PSF 바 차트 (카드 내장용) */
function drawMiniPsf(canvasId, psf7) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !psf7 || psf7.length !== 7) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const pad = { left: 4, right: 4, top: 2, bottom: 14 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const maxVal = Math.max(...psf7, 0.001);
  const barW = plotW / 7 * 0.7;

  psf7.forEach((v, i) => {
    const x = pad.left + i * (plotW / 7) + (plotW / 7 * 0.15);
    const h = (v / maxVal) * plotH;
    const y = pad.top + plotH - h;

    const isRidge = i % 2 === 0;
    ctx.fillStyle = (i === 3) ? '#22c55e' : isRidge ? '#3b82f6' : '#8b5cf6';
    ctx.fillRect(x, y, barW, h);
  });

  // 축 라벨
  ctx.font = '8px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#8b90a0';
  const labels = ['R','V','R','C','R','V','R'];
  psf7.forEach((_, i) => {
    const x = pad.left + i * (plotW / 7) + (plotW / 7 * 0.5);
    ctx.fillText(labels[i], x, H - 2);
  });
}

/** 미니 COE 스택 (카드 내장용) */
function drawMiniStack(canvasId, params) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { delta_bm1 = 0, delta_bm2 = 0, w1 = 10, w2 = 10, d = 20 } = params || {};
  const pad = 3;
  const sW = W - 2 * pad, sH = H - 2 * pad;

  const layers = [
    { name: 'AR',     h: 0.03, color: '#6366f1' },
    { name: 'CG',     h: 0.50, color: '#1e40af' },
    { name: 'BM1',    h: 0.03, color: '#374151', isBM: true, w: w1, offset: delta_bm1 },
    { name: 'ILD',    h: 0.12, color: '#1e3a5f' },
    { name: 'BM2',    h: 0.03, color: '#374151', isBM: true, w: w2, offset: delta_bm2 },
    { name: 'Enc',    h: 0.12, color: '#164e63' },
    { name: 'OPD',    h: 0.05, color: '#065f46' },
  ];
  const totalR = layers.reduce((s, l) => s + l.h, 0);

  let y = pad;
  layers.forEach(layer => {
    const lh = (layer.h / totalR) * sH;
    if (layer.isBM) {
      ctx.fillStyle = layer.color;
      ctx.fillRect(pad, y, sW, lh);
      // aperture
      const aperW = (layer.w / 72) * sW * 0.8;
      const offPx = (layer.offset / 72) * sW * 0.8;
      const aperX = pad + sW / 2 - aperW / 2 + offPx;
      ctx.fillStyle = '#0f1117';
      ctx.fillRect(Math.max(pad, aperX), y, Math.min(aperW, sW), lh);
    } else {
      ctx.fillStyle = layer.color;
      ctx.fillRect(pad, y, sW, lh);
    }
    y += lh;
  });

  ctx.strokeStyle = '#2d3140';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(pad, pad, sW, sH);
}

/** 후보 리스트 렌더 (스택+PSF 포함) */
function renderCandidatesList(containerId, candidates, baseline, selectedId) {
  const el = document.getElementById(containerId);
  if (!el) return;

  // containerId 기반 prefix로 canvas id 충돌 방지
  const prefix = containerId.replace(/[^a-z0-9]/gi, '') + '_';

  if (!candidates || candidates.length === 0) {
    el.innerHTML = '<div style="color:#8b90a0;text-align:center;padding:20px;">Run reverse design to see candidates</div>';
    return;
  }

  el.innerHTML = candidates.map((c, i) =>
    renderCandidateCard(c, baseline, c.id === selectedId, i, prefix)
  ).join('');

  // 카드 내 스택 + PSF 렌더링 (DOM에 삽입된 후 실행)
  requestAnimationFrame(() => {
    candidates.forEach((c, i) => {
      const idx = prefix + i;
      drawMiniStack('card-stack-' + idx, c.params);
      // PINN PSF가 후보에 포함되어 있으면 사용
      const cardPsf = c.pinnPsf7 || [0.16,0.15,0.09,0.09,0.15,0.16,0.15];
      drawMiniPsf('card-psf-' + idx, cardPsf);
    });
  });

  // Click handler
  el.querySelectorAll('.candidate-card').forEach(card => {
    card.addEventListener('click', () => {
      DesignStore.selectCandidate(card.dataset.id);
    });
  });
}

/** LT Export 핸들러 */
async function exportCandidate(candidateId) {
  try {
    const result = await ApiClient.exportCandidate(candidateId);
    const blob = new Blob([result.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${candidateId}_lt_macro.py`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error('Export failed:', e);
    alert('Export failed: ' + e.message);
  }
}
