/**
 * drawAmplitudeProfile.js - 1D |A| or |A|^2 profile
 * Shows slit positions as shaded bands.
 */
function drawAmplitudeProfile(canvas, values, xCoords, options) {
  var ctx = canvas.getContext('2d');
  var color = (options && options.color) || '#1428A0';
  var fill = (options && options.fill) || 'rgba(20,40,160,0.12)';
  var label = (options && options.label) || '|A|';
  var d = (options && options.d) || 0;
  var w = (options && options.w) || 10;
  var pitch = 72;

  var W = canvas.width, H = canvas.height;
  var m = { top: 22, right: 16, bottom: 26, left: 44 };
  var pW = W - m.left - m.right;
  var pH = H - m.top - m.bottom;
  ctx.clearRect(0, 0, W, H);

  var xmin = xCoords[0], xmax = xCoords[xCoords.length - 1];
  var vmax = 0;
  for (var i = 0; i < values.length; i++) { if (values[i] > vmax) vmax = values[i]; }
  vmax = (options && options.vmax) ? options.vmax : vmax * 1.15;
  if (vmax < 0.01) vmax = 1;

  function xPx(x) { return m.left + (x - xmin) / (xmax - xmin) * pW; }
  function yPx(y) { return m.top + pH - (y / vmax) * pH; }

  // Slit bands
  ctx.fillStyle = 'rgba(255,220,100,0.25)';
  for (var p = 0; p < 7; p++) {
    var cx = p * pitch + pitch / 2 + d;
    ctx.fillRect(xPx(cx - w / 2), m.top, xPx(cx + w / 2) - xPx(cx - w / 2), pH);
  }

  // Line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (var i = 0; i < values.length; i++) {
    var px = xPx(xCoords[i]), py = yPx(values[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Fill
  ctx.fillStyle = fill;
  ctx.lineTo(xPx(xCoords[xCoords.length - 1]), yPx(0));
  ctx.lineTo(xPx(xCoords[0]), yPx(0));
  ctx.closePath();
  ctx.fill();

  // Axes
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH);
  ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH);
  ctx.stroke();

  ctx.fillStyle = '#555';
  ctx.font = '11px sans-serif';
  ctx.fillText(label, m.left + 4, m.top + 14);
  ctx.fillText('x (um)', m.left + pW - 40, H - 4);
  ctx.fillText('0', m.left - 10, m.top + pH + 4);
  ctx.fillText(vmax.toFixed(2), m.left - 38, m.top + 6);
}
