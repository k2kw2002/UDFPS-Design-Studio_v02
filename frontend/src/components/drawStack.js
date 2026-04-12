/**
 * drawStack.js - COE 스택 단면 렌더러
 * 가이드 27번: 탭 공용 COE 단면 렌더러 (cvEl 파라미터로 캔버스 교체)
 *
 * COE 스택 구조:
 *   AR Coating (상단)
 *   Cover Glass + OCR (550um)
 *   BM1 (0.1um) — aperture w1, offset delta_bm1
 *   ILD (d=20um)
 *   BM2 (0.1um) — aperture w2, offset delta_bm2
 *   Encap (20um)
 *   OPD (하단)
 */
function drawStack(canvasId, params, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { delta_bm1 = 0, delta_bm2 = 0, w1 = 10, w2 = 10, d = 20 } = params || {};
  const showRays = options.showRays !== false;

  const pad = 10;
  const stackW = W - 2 * pad;
  const stackH = H - 2 * pad;

  // Layer heights (proportional)
  const layers = [
    { name: 'AR', h: 0.03, color: '#6366f1' },
    { name: 'CG+OCR', h: 0.55, color: '#1e40af' },
    { name: 'BM1', h: 0.02, color: '#374151', isBM: true, w: w1, offset: delta_bm1 },
    { name: 'ILD', h: 0.10, color: '#1e3a5f' },
    { name: 'BM2', h: 0.02, color: '#374151', isBM: true, w: w2, offset: delta_bm2 },
    { name: 'Encap', h: 0.10, color: '#164e63' },
    { name: 'OPD', h: 0.05, color: '#065f46' },
  ];
  const totalRatio = layers.reduce((s, l) => s + l.h, 0);

  let y = pad;
  layers.forEach(layer => {
    const lh = (layer.h / totalRatio) * stackH;

    if (layer.isBM) {
      // BM layer with aperture
      ctx.fillStyle = layer.color;
      ctx.fillRect(pad, y, stackW, lh);

      // Aperture (transparent opening)
      const aperW = (layer.w / 72) * stackW * 0.8;
      const offsetPx = (layer.offset / 72) * stackW * 0.8;
      const aperX = pad + stackW / 2 - aperW / 2 + offsetPx;

      ctx.fillStyle = '#0f1117';
      ctx.fillRect(aperX, y, aperW, lh);

      // Aperture width indicator
      ctx.strokeStyle = '#eab308';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(aperX, y + lh + 2);
      ctx.lineTo(aperX + aperW, y + lh + 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.fillStyle = layer.color;
      ctx.fillRect(pad, y, stackW, lh);
    }

    // Label
    ctx.fillStyle = '#e1e4ea';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    if (lh > 12) {
      ctx.fillText(layer.name, pad + 4, y + lh / 2 + 3);
    }

    y += lh;
  });

  // Draw acceptance angle rays
  if (showRays) {
    const theta_eff = Math.atan(w1 / (2 * d)) * (180 / Math.PI);
    const bm1Y = pad + layers.slice(0, 2).reduce((s, l) => s + (l.h / totalRatio) * stackH, 0);
    const centerX = pad + stackW / 2;
    const rayLen = stackH * 0.4;
    const angleRad = theta_eff * Math.PI / 180;

    ctx.strokeStyle = '#f9731666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, bm1Y);
    ctx.lineTo(centerX - rayLen * Math.sin(angleRad), bm1Y - rayLen * Math.cos(angleRad));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(centerX, bm1Y);
    ctx.lineTo(centerX + rayLen * Math.sin(angleRad), bm1Y - rayLen * Math.cos(angleRad));
    ctx.stroke();

    // Angle label
    ctx.fillStyle = '#f97316';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`\u03B8=${theta_eff.toFixed(1)}\u00B0`, centerX, bm1Y - rayLen * 0.5 - 6);
  }

  // Border
  ctx.strokeStyle = '#2d3140';
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, stackW, stackH);
}
