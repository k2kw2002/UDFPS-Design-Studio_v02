/**
 * client.js - Backend API client
 * 가이드 12.9절 FNO 연동 구현.
 */
const ApiClient = (() => {
  const BASE = '';  // 상대경로 — localhost, HF Spaces 등 어디서든 동작

  async function post(path, body) {
    const res = await fetch(BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function get(path) {
    const res = await fetch(BASE + path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // ---- Public API ----

  /** PSF 추론 (Explore 탭) */
  async function inferPsf(params) {
    return post('/api/inference/psf', { params });
  }

  /** 역설계 실행 */
  async function runDesign(spec, weights) {
    return post('/api/design/run', { spec, weights });
  }

  /** 작업 상태 조회 */
  async function getStatus(jobId) {
    return get(`/api/design/status/${jobId}`);
  }

  /** 후보 목록 */
  async function getCandidates(jobId) {
    return get(`/api/design/candidates/${jobId}`);
  }

  /** LT 내보내기 */
  async function exportCandidate(candidateId) {
    return post(`/api/design/export/${candidateId}`, {});
  }

  /** 서버 상태 */
  async function health() {
    return get('/api/health');
  }

  /**
   * approxModel 삭제됨 — 모든 PSF 계산은 ASM API 사용
   */

  /**
   * 통합 추론 API — 서버가 최상위 모델 자동 선택
   * Parametric PINN (~1ms) > ASM (~1s)
   */
  async function predict(params) {
    const q = new URLSearchParams(params).toString();
    return get('/api/inference/predict?' + q);
  }

  /** ASM 기반 (fallback) */
  async function getMetrics(params) {
    return predict(params);
  }

  /** ASM 파동광학 PSF */
  async function getAsmPsf(params) {
    const q = new URLSearchParams(params).toString();
    return get('/api/inference/asm-psf?' + q);
  }

  /** ASM 비교 (CG only / AR baseline / AR optimized) */
  async function getAsmComparison(w1, w2) {
    return get('/api/inference/asm-comparison?w1=' + w1 + '&w2=' + w2);
  }

  /** PINN 학습 결과 조회 */
  async function getPinnResult() {
    return get('/api/inference/pinn-result');
  }

  /** BoTorch 최적화 결과 조회 */
  async function getBotorchResult() {
    return get('/api/inference/botorch-result');
  }

  return {
    inferPsf, runDesign, getStatus, getCandidates,
    exportCandidate, health, predict, getMetrics,
    getPinnResult, getBotorchResult,
    getAsmPsf, getAsmComparison,
  };
})();
