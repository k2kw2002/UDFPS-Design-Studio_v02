/**
 * designStore.js - Zustand-style global state (vanilla JS)
 * 가이드 12.8절 Zustand 전역 상태 구현.
 * Node.js 없이 순수 JS로 pub/sub 패턴 구현.
 */
const DesignStore = (() => {
  // ---- State ----
  const state = {
    // 역설계 목표
    spec: { mtf_min: 0.60, skew_max: 0.10, T_min: 0.60 },
    weights: { mtf: 0.4, T: 0.3, skew: 0.3 },

    // 현재 양산 기준 (ASM AR baseline, CG=550um, w=10)
    baseline: { mtf: 0.283, skew: 0.0, T: 0.15 },

    // Explore 탭 설계변수
    explore_params: { delta_bm1: 0, delta_bm2: 0, w1: 10, w2: 10, d: 20 },

    // 역설계 결과
    candidates: [],
    pareto_points: [],
    selected_id: null,
    hypervolume: null,

    // 학습/최적화 상태
    pinn_loss: { helm: null, phase: null, I: null, BC: null },
    iter_current: 0,
    iter_total: 50,
    fno_ready: false,
    backend_mode: 'approx', // 'approx' | 'fno'
    lt_count: 0,

    // Explore 실시간 결과
    explore_psf: [0, 0, 0, 0, 0, 0, 0],
    explore_metrics: { mtf_ridge: 0, skewness: 0, throughput: 0, crosstalk_ratio: 0 },

    // 작업 상태
    job_id: null,
    job_status: 'idle', // idle, running, completed, error
  };

  // ---- Subscribers ----
  const listeners = [];

  function subscribe(fn) {
    listeners.push(fn);
    return () => { const i = listeners.indexOf(fn); if (i >= 0) listeners.splice(i, 1); };
  }

  function notify() {
    listeners.forEach(fn => fn(state));
  }

  // ---- Actions ----
  function set(partial) {
    Object.assign(state, partial);
    notify();
  }

  function setSpec(s) {
    state.spec = { ...state.spec, ...s };
    notify();
  }

  function setWeights(w) {
    state.weights = { ...state.weights, ...w };
    notify();
  }

  function setExploreParams(p) {
    state.explore_params = { ...state.explore_params, ...p };
    notify();
  }

  function selectCandidate(id) {
    state.selected_id = id;
    notify();
  }

  function setBackendMode(m) {
    state.backend_mode = m;
    notify();
  }

  function setCandidates(candidates) {
    state.candidates = candidates;
    notify();
  }

  function setJobStatus(status) {
    state.job_status = status;
    notify();
  }

  function setExploreResult(psf7, metrics) {
    state.explore_psf = psf7;
    state.explore_metrics = metrics;
    notify();
  }

  function setPinnLoss(loss) {
    state.pinn_loss = { ...state.pinn_loss, ...loss };
    notify();
  }

  function setProgress(iter, total, hv) {
    state.iter_current = iter;
    state.iter_total = total;
    if (hv !== undefined) state.hypervolume = hv;
    notify();
  }

  return {
    state, subscribe, set, setSpec, setWeights,
    setExploreParams, selectCandidate, setBackendMode,
    setCandidates, setJobStatus, setExploreResult,
    setPinnLoss, setProgress,
  };
})();
