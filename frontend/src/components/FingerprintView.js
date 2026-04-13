/**
 * FingerprintView.js - 원본 지문 이미지 + PSF 광학 효과 시뮬레이션
 *
 * 원본 지문: fingerprint_sample.png (실제 지문 이미지)
 * PSF 효과: BM 설계에 따른 MTF/skewness/crosstalk 반영
 *
 * Current Design = 현재 양산 기준 설계(baseline)에서 OPD 센서로 촬영한 결과
 * Optimized     = 역설계 후 최적 BM 구조에서 OPD 센서로 촬영한 결과
 * 둘 다 동일한 원본 지문, BM 설계에 따라 품질이 달라짐
 */

// ============================================================
// 원본 지문 이미지 로드 (한 번만)
// ============================================================
const _fpState = {
  img: null,
  loaded: false,
  loading: false,
  callbacks: [],
};

function _loadFingerprintImage(callback) {
  if (_fpState.loaded && _fpState.img) {
    callback(_fpState.img);
    return;
  }
  _fpState.callbacks.push(callback);
  if (_fpState.loading) return;
  _fpState.loading = true;

  const img = new Image();
  img.onload = () => {
    _fpState.img = img;
    _fpState.loaded = true;
    _fpState.loading = false;
    _fpState.callbacks.forEach(cb => cb(img));
    _fpState.callbacks = [];
  };
  img.onerror = () => {
    console.warn('Fingerprint image not found, using generated pattern');
    _fpState.loading = false;
    _fpState.loaded = true;
    _fpState.img = null;
    _fpState.callbacks.forEach(cb => cb(null));
    _fpState.callbacks = [];
  };
  img.src = '/src/assets/fingerprint_sample.png';
}

// ============================================================
// 원본 지문을 grayscale로 추출
// ============================================================
function _extractGrayscale(img, W, H) {
  const offscreen = document.createElement('canvas');
  offscreen.width = W;
  offscreen.height = H;
  const ctx = offscreen.getContext('2d');

  // 원본 비율 유지하며 캔버스에 맞춤
  const scale = Math.min(W / img.width, H / img.height);
  const sw = img.width * scale;
  const sh = img.height * scale;
  const ox = (W - sw) / 2;
  const oy = (H - sh) / 2;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(img, ox, oy, sw, sh);

  const imgData = ctx.getImageData(0, 0, W, H);
  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = imgData.data[i * 4];
    const g = imgData.data[i * 4 + 1];
    const b = imgData.data[i * 4 + 2];
    // 원본: ridge=검정(0), valley=흰색(255)
    // 정규화: ridge=1, valley=0 (센서에서 ridge가 밝음)
    gray[i] = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
  }
  return gray;
}

// ============================================================
// fallback: 생성 패턴 (이미지 로드 실패 시)
// ============================================================
function _generateFallbackFingerprint(W, H) {
  const data = new Float32Array(W * H);
  const cx = W * 0.48, cy = H * 0.52;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - cx, dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      const theta = Math.atan2(dy, dx);
      const stretch = 1.0 + 0.15 * Math.cos(theta * 2 - 0.5);
      const period = 7.5 + r * stretch * 0.012;
      const phase = r * stretch / period + theta * 1.8;
      let ridge = Math.cos(phase * Math.PI * 2) > 0 ? 1.0 : 0.0;
      const ex = dx / (W * 0.38), ey = dy / (H * 0.44);
      const ell = ex * ex + ey * ey;
      if (ell > 1.0) ridge = 0;
      else if (ell > 0.85) ridge *= (1.0 - ell) / 0.15;
      data[y * W + x] = ridge;
    }
  }
  return data;
}

// ============================================================
// PSF 효과: Gaussian blur
// ============================================================
function _gaussianBlur(data, W, H, radius) {
  if (radius <= 0) return new Float32Array(data);
  const kernel = [];
  let sum = 0;
  const sigma = radius * 0.5 + 0.5;
  for (let i = -radius; i <= radius; i++) {
    const w = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(w);
    sum += w;
  }
  kernel.forEach((_, i, a) => a[i] /= sum);

  // Horizontal
  const temp = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.max(0, Math.min(W - 1, x + k));
        v += data[y * W + xx] * kernel[k + radius];
      }
      temp[y * W + x] = v;
    }
  }
  // Vertical
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.max(0, Math.min(H - 1, y + k));
        v += temp[yy * W + x] * kernel[k + radius];
      }
      out[y * W + x] = v;
    }
  }
  return out;
}

// ============================================================
// PSF 광학 효과 적용
// ============================================================
function _applyPsfEffect(ideal, W, H, metrics) {
  const mtf = metrics ? (metrics.mtf_ridge != null ? metrics.mtf_ridge : 0.42) : 0.42;
  const skew = metrics ? Math.abs(metrics.skewness || 0.28) : 0.28;
  const xtalk = metrics ? (metrics.crosstalk_ratio != null ? metrics.crosstalk_ratio : 0.15) : 0.15;

  // MTF → blur+contrast 매핑
  // 캔버스가 150~200px로 작으므로 blur는 최대 2까지만
  // ridge가 2~3px 폭이라 blur>2이면 완전히 사라짐
  // 대신 contrast로 MTF 차이를 표현
  const mtfNorm = Math.max(0, Math.min(1, (mtf - 0.20) / 0.60)); // 0.20→0, 0.50→0.5, 0.80→1
  // blur: 매우 약하게 (0~2)
  const blurR = Math.max(0, Math.round((1.0 - mtfNorm) * 2));
  let out = _gaussianBlur(ideal, W, H, blurR);

  // contrast가 주된 시각 효과
  // MTF=0.20 → 40% (흐릿), MTF=0.42 → 60%, MTF=0.60 → 76%, MTF=0.80 → 92%
  const contrast = 0.40 + 0.55 * mtfNorm;

  for (let i = 0; i < W * H; i++) {
    let v = out[i];

    // contrast (ridge/valley 대비 조절 — MTF가 주 효과)
    v = 0.5 + (v - 0.5) * contrast;

    // crosstalk: valley에 약간의 빛 누출 (너무 강하면 ridge 안 보임)
    if (v < 0.3) v += xtalk * 0.15;

    // skewness: 좌우 비대칭 (약하게)
    if (skew > 0.05) {
      const x = i % W;
      const xn = (x / W - 0.5) * 2;
      v += skew * xn * 0.03;
    }

    out[i] = Math.max(0, Math.min(1, v));
  }
  return out;
}

// ============================================================
// 메인 렌더링
// ============================================================
function drawFingerprint(canvasId, psf7, metrics, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  _loadFingerprintImage((img) => {
    _renderFingerprint(canvas, img, psf7, metrics, options);
  });
}

function _renderFingerprint(canvas, fpImg, psf7, metrics, options) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // 1) 원본 지문 grayscale 추출 (캐시)
  const cacheKey = canvas.id + '_gray_' + W + 'x' + H;
  if (!_renderFingerprint._cache) _renderFingerprint._cache = {};
  if (!_renderFingerprint._cache[cacheKey]) {
    if (fpImg) {
      _renderFingerprint._cache[cacheKey] = _extractGrayscale(fpImg, W, H);
    } else {
      _renderFingerprint._cache[cacheKey] = _generateFallbackFingerprint(W, H);
    }
  }
  const ideal = _renderFingerprint._cache[cacheKey];

  // 2) PSF 효과
  const rendered = _applyPsfEffect(ideal, W, H, metrics);

  // 3) 캔버스에 그리기
  const imgData = ctx.createImageData(W, H);
  const px = imgData.data;

  for (let i = 0; i < W * H; i++) {
    const v = rendered[i];

    // Grayscale (ridge=밝음, valley=어두움)
    {
      const lum = Math.floor(v * 255);
      px[i * 4] = lum;
      px[i * 4 + 1] = lum;
      px[i * 4 + 2] = lum;
    }
    px[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  // 4) OPD 그리드
  if (options.showGrid) {
    ctx.strokeStyle = 'rgba(79, 143, 247, 0.15)';
    ctx.lineWidth = 0.5;
    const step = W / 7;
    for (let i = 1; i < 7; i++) {
      ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, H); ctx.stroke();
    }
  }

  // 5) 정보 오버레이
  if (options.showInfo !== false && metrics) {
    const mtf = metrics.mtf_ridge || 0;
    const skew = Math.abs(metrics.skewness || 0);
    const xtalk = metrics.crosstalk_ratio || 0;

    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(4, H - 48, 145, 44);
    ctx.font = '11px monospace';

    ctx.fillStyle = mtf >= 0.60 ? '#22c55e' : mtf >= 0.45 ? '#eab308' : '#ef4444';
    ctx.fillText(`MTF  ${(mtf * 100).toFixed(0)}%`, 10, H - 32);
    ctx.fillStyle = skew <= 0.10 ? '#22c55e' : skew <= 0.20 ? '#eab308' : '#ef4444';
    ctx.fillText(`Skew ${skew.toFixed(3)}`, 10, H - 18);
    ctx.fillStyle = '#8b90a0';
    ctx.fillText(`XT ${(xtalk * 100).toFixed(0)}%`, 90, H - 32);
  }

  // 6) 라벨
  if (options.label) {
    ctx.font = 'bold 12px sans-serif';
    const tw = ctx.measureText(options.label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(4, 4, tw + 16, 24);
    ctx.fillStyle = options.labelColor || '#e1e4ea';
    ctx.fillText(options.label, 12, 20);
  }
}

/**
 * 원본 지문 이미지 그대로 표시 (PSF 효과 없음)
 */
function drawFingerprintOriginal(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  _loadFingerprintImage((img) => {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);

    if (img) {
      // 원본 비율 유지
      const scale = Math.min(W / img.width, H / img.height);
      const sw = img.width * scale;
      const sh = img.height * scale;
      const ox = (W - sw) / 2;
      const oy = (H - sh) / 2;
      ctx.drawImage(img, ox, oy, sw, sh);
    }

    // 라벨
    ctx.font = 'bold 11px sans-serif';
    const tw = ctx.measureText('Original').width;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(4, 4, tw + 14, 22);
    ctx.fillStyle = '#e1e4ea';
    ctx.fillText('Original', 11, 18);
  });
}
