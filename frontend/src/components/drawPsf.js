/**
 * drawPsf.js - PSF 7-OPD 바 차트 렌더러
 */
function drawPsf(canvasId, psf7, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!psf7 || psf7.length !== 7) return;

  const pad = { top: 20, bottom: 30, left: 40, right: 10 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const maxVal = Math.max(...psf7, 0.01);
  const barW = plotW / 7 * 0.7;
  const gap = plotW / 7 * 0.3;

  // Grid lines
  ctx.strokeStyle = '#2d3140';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  // Bars
  const labels = ['OPD0', 'OPD1', 'OPD2', 'OPD3', 'OPD4', 'OPD5', 'OPD6'];
  const colors = ['#3b82f6', '#6366f1', '#3b82f6', '#22c55e', '#3b82f6', '#6366f1', '#3b82f6'];
  // Ridge(even) = blue, Valley(odd) = purple, Center = green

  psf7.forEach((v, i) => {
    const x = pad.left + i * (plotW / 7) + gap / 2;
    const h = (v / maxVal) * plotH;
    const y = pad.top + plotH - h;

    const isRidge = i % 2 === 0;
    const isCenter = i === 3;
    ctx.fillStyle = isCenter ? '#22c55e' : isRidge ? '#3b82f6' : '#8b5cf6';
    ctx.fillRect(x, y, barW, h);

    // Value label
    ctx.fillStyle = '#e1e4ea';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(3), x + barW / 2, y - 4);

    // X label
    ctx.fillStyle = '#8b90a0';
    ctx.font = '9px sans-serif';
    ctx.fillText(i === 3 ? 'R' : (i % 2 === 0 ? 'R' : 'V'), x + barW / 2, H - pad.bottom + 12);
    ctx.fillText(i.toString(), x + barW / 2, H - pad.bottom + 22);
  });

  // Y axis label
  ctx.save();
  ctx.fillStyle = '#8b90a0';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Intensity', 0, 0);
  ctx.restore();
}
