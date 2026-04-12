/**
 * drawPareto.js - Pareto front 시각화
 * 가이드 12.4절 규칙:
 *   x축: MTF@ridge, y축: 광량 T, 원 크기: 1/skewness
 *   노란점=현재양산, 파란점=Pareto, 초록점=선택안
 */
function drawPareto(canvasId, candidates, baseline, spec, selectedId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const pad = { top: 20, bottom: 40, left: 50, right: 20 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Axis ranges
  const xMin = 0, xMax = 1;
  const yMin = 0, yMax = 1;

  function toX(mtf) { return pad.left + ((mtf - xMin) / (xMax - xMin)) * plotW; }
  function toY(T) { return pad.top + plotH - ((T - yMin) / (yMax - yMin)) * plotH; }

  // Grid
  ctx.strokeStyle = '#2d3140';
  ctx.lineWidth = 0.5;
  for (let v = 0; v <= 1; v += 0.2) {
    ctx.beginPath(); ctx.moveTo(toX(v), pad.top); ctx.lineTo(toX(v), pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, toY(v)); ctx.lineTo(pad.left + plotW, toY(v)); ctx.stroke();
    ctx.fillStyle = '#8b90a0'; ctx.font = '10px sans-serif';
    ctx.textAlign = 'center'; ctx.fillText((v * 100).toFixed(0) + '%', toX(v), H - pad.bottom + 14);
    ctx.textAlign = 'right'; ctx.fillText((v * 100).toFixed(0) + '%', pad.left - 6, toY(v) + 3);
  }

  // Target zone (green dashed)
  if (spec) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#22c55e55';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(toX(spec.mtf_min), pad.top); ctx.lineTo(toX(spec.mtf_min), pad.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, toY(spec.T_min)); ctx.lineTo(pad.left + plotW, toY(spec.T_min)); ctx.stroke();
    ctx.setLineDash([]);

    // Target region fill
    ctx.fillStyle = '#22c55e08';
    ctx.fillRect(toX(spec.mtf_min), pad.top, pad.left + plotW - toX(spec.mtf_min), toY(spec.T_min) - pad.top);
  }

  // Baseline point (yellow)
  if (baseline) {
    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.arc(toX(baseline.mtf), toY(baseline.T), 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#eab308';
    ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('Current', toX(baseline.mtf) + 12, toY(baseline.T) + 4);
  }

  // Pareto candidates (blue/green)
  if (candidates && candidates.length > 0) {
    candidates.forEach(c => {
      const isSelected = c.id === selectedId;
      const r = Math.max(4, Math.min(16, 4 / (Math.abs(c.skewness) + 0.01)));

      ctx.fillStyle = isSelected ? '#22c55e' : '#4f8ff7';
      ctx.globalAlpha = isSelected ? 1.0 : 0.7;
      ctx.beginPath();
      ctx.arc(toX(c.mtf_ridge), toY(c.throughput), r, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = 1.0;
      ctx.fillStyle = isSelected ? '#22c55e' : '#4f8ff7';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(c.label, toX(c.mtf_ridge), toY(c.throughput) - r - 4);
    });

    // Arrow from baseline to best
    if (baseline && selectedId) {
      const sel = candidates.find(c => c.id === selectedId);
      if (sel) {
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(toX(baseline.mtf), toY(baseline.T));
        ctx.lineTo(toX(sel.mtf_ridge), toY(sel.throughput));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // Axis labels
  ctx.fillStyle = '#8b90a0';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('MTF@ridge', pad.left + plotW / 2, H - 4);
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Throughput T', 0, 0);
  ctx.restore();
}
