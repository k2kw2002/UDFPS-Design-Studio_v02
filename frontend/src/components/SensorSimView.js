/**
 * SensorSimView.js - 원본 지문 이미지 + PSF 기반 센서 출력 시뮬레이션
 *
 * 원본 fingerprint_sample.png의 ridge/valley를 기반으로
 * PSF MTF에 따른 센서 출력을 시뮬레이션.
 */

// 전역 이미지 (init 시 로드)
let _sensorFpImg = null;
let _sensorFpReady = false;

/**
 * 페이지 로드 시 호출 — 이미지를 미리 로드
 */
function initSensorImage() {
  return new Promise((resolve) => {
    if (_sensorFpReady) { resolve(_sensorFpImg); return; }
    const img = new Image();
    img.onload = () => { _sensorFpImg = img; _sensorFpReady = true; resolve(img); };
    img.onerror = () => { _sensorFpImg = null; _sensorFpReady = true; resolve(null); };
    img.src = '/src/assets/fingerprint_sample.png';
  });
}

// ridge 마스크 캐시
const _ridgeMaskCache = {};

function _getRidgeMask(img, W, H, key) {
  if (_ridgeMaskCache[key]) return _ridgeMaskCache[key];

  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const ctx = off.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  if (img) {
    const scale = Math.min(W / img.width, H / img.height);
    const sw = img.width * scale, sh = img.height * scale;
    ctx.drawImage(img, (W - sw) / 2, (H - sh) / 2, sw, sh);
  }

  const data = ctx.getImageData(0, 0, W, H).data;
  const mask = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const lum = (data[i*4] * 0.299 + data[i*4+1] * 0.587 + data[i*4+2] * 0.114) / 255;
    mask[i] = 1.0 - lum; // ridge=1, valley=0
  }

  _ridgeMaskCache[key] = mask;
  return mask;
}

/**
 * 센서 시뮬레이션 렌더링 (동기 — 이미지는 이미 로드됨)
 */
function drawSensorSim(canvasId, psf7, options) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  options = options || {};
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  if (!psf7 || psf7.length !== 7) {
    psf7 = [0.17, 0.06, 0.20, 0.12, 0.20, 0.06, 0.17];
  }

  // MTF (절대값)
  const psfMax = Math.max(...psf7, 1e-8);
  const psfN = psf7.map(v => v / psfMax);
  const rMean = (psfN[0] + psfN[2] + psfN[4] + psfN[6]) / 4;
  const vMean = (psfN[1] + psfN[3] + psfN[5]) / 3;
  const mtf = Math.abs((rMean - vMean) / (rMean + vMean + 1e-8));

  // MTF → 밝기
  const baseBright = 0.40;
  const contrast = mtf * 0.55;
  const ridgeBright = Math.min(1.0, baseBright + contrast);
  const valleyBright = Math.max(0.03, baseBright - contrast);

  // 이미지가 로드되었으면 원본 지문 사용, 아니면 fallback 패턴
  const cacheKey = canvasId + '_' + W + 'x' + H;

  if (_sensorFpReady && _sensorFpImg) {
    // 원본 지문에서 ridge 마스크
    const mask = _getRidgeMask(_sensorFpImg, W, H, cacheKey);
    _renderWithMask(ctx, W, H, mask, ridgeBright, valleyBright);
  } else {
    // fallback: 생성 패턴
    _renderFallbackPattern(ctx, W, H, ridgeBright, valleyBright);
  }

  // 오버레이
  if (options.showPsfValues) {
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(W - 78, 3, 74, 108);
    ctx.font = '9px monospace';
    const labels = ['R0','V1','R2','C3','R4','V5','R6'];
    psf7.forEach((v, j) => {
      ctx.fillStyle = (j === 3) ? '#22c55e' : (j % 2 === 0) ? '#3b82f6' : '#8b5cf6';
      ctx.fillText(labels[j] + ' ' + v.toFixed(3), W - 74, 15 + j * 13);
    });
    ctx.fillStyle = mtf >= 0.30 ? '#22c55e' : mtf >= 0.20 ? '#eab308' : '#ef4444';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('MTF ' + (mtf * 100).toFixed(0) + '%', W - 74, 15 + 7 * 13);
  }

  if (options.label) {
    ctx.font = 'bold 11px sans-serif';
    const tw = ctx.measureText(options.label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(4, H - 24, tw + 14, 22);
    ctx.fillStyle = options.labelColor || '#e1e4ea';
    ctx.fillText(options.label, 11, H - 9);
  }
}

function _renderWithMask(ctx, W, H, mask, ridgeBright, valleyBright) {
  const imgData = ctx.createImageData(W, H);
  const px = imgData.data;
  for (let i = 0; i < W * H; i++) {
    const r = mask[i];
    let intensity = r * ridgeBright + (1 - r) * valleyBright;
    const x = i % W, y = Math.floor(i / W);
    intensity += (Math.sin(x * 127.1 + y * 311.7) * 0.5 + 0.5) * 0.01;
    intensity = Math.max(0, Math.min(1, intensity));
    px[i*4]   = Math.floor(intensity * 50 + 3);
    px[i*4+1] = Math.floor(intensity * 210 + 10);
    px[i*4+2] = Math.floor(intensity * 40 + 3);
    px[i*4+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

function _renderFallbackPattern(ctx, W, H, ridgeBright, valleyBright) {
  const imgData = ctx.createImageData(W, H);
  const px = imgData.data;
  const cx = W * 0.48, cy = H * 0.52;
  const period = Math.max(10, W / 16);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const dx = x - cx, dy = y - cy;
      const r = Math.sqrt(dx*dx + dy*dy);
      const theta = Math.atan2(dy, dx);
      const warp = Math.sin(theta*2.3)*4 + Math.cos(theta*1.7)*3;
      const phase = (r + warp) / period;
      const ridge = Math.cos(phase * Math.PI * 2) > 0 ? 1 : 0;
      const ex = dx/(W*0.38), ey = dy/(H*0.44);
      const ell = ex*ex + ey*ey;
      let mask = ell > 1 ? 0 : ell > 0.82 ? (1-ell)/0.18 : 1;
      let intensity = (ridge * ridgeBright + (1-ridge) * valleyBright) * mask;
      intensity = Math.max(0, Math.min(1, intensity));
      px[i*4]   = Math.floor(intensity * 50 + 3);
      px[i*4+1] = Math.floor(intensity * 210 + 10);
      px[i*4+2] = Math.floor(intensity * 40 + 3);
      px[i*4+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}
