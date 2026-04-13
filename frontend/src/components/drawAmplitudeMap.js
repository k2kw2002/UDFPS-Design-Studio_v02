/**
 * drawAmplitudeMap.js - 2D |A| heatmap (Canvas)
 * BM region: dark (|A|~0), Slit: bright
 */
function drawAmplitudeMap(canvas, ampData, options) {
  const ctx = canvas.getContext('2d');
  const nz = ampData.length;
  const nx = ampData[0].length;
  const vmin = (options && options.vmin) || 0;
  const vmax = (options && options.vmax) || 0.8;
  const W = canvas.width;
  const H = canvas.height;
  const pw = W / nx;
  const ph = H / nz;

  function colormap(v) {
    var t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
    var r = Math.round(68 + 187 * t);
    var g = Math.round(1 + 179 * Math.pow(t, 0.8));
    var b = Math.round(84 * (1 - t * 0.5));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }

  for (var j = 0; j < nz; j++) {
    for (var i = 0; i < nx; i++) {
      ctx.fillStyle = colormap(ampData[j][i]);
      ctx.fillRect(i * pw, (nz - 1 - j) * ph, pw + 1, ph + 1);
    }
  }

  // Labels
  ctx.fillStyle = '#555';
  ctx.font = '11px sans-serif';
  ctx.fillText('x (um)', W / 2 - 20, H - 2);
  ctx.fillText('z=40', 4, 14);
  ctx.fillText('z=0', 4, H - 6);

  // BM lines
  ctx.strokeStyle = 'rgba(255,255,255,0.4)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  var yBM1 = 0;
  var yBM2 = H / 2;
  ctx.moveTo(0, yBM1); ctx.lineTo(W, yBM1);
  ctx.moveTo(0, yBM2); ctx.lineTo(W, yBM2);
  ctx.stroke();
  ctx.setLineDash([]);
}
