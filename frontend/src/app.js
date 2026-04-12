const StudioApp = (() => {
  const state = {
    mode: "reverse",
    selectedId: null,
    candidates: [],
    current: {
      id: "manual",
      name: "Manual",
      tag: "Forward design",
      params: { delta_bm1: 0, delta_bm2: 0, w1: 10, w2: 10, d: 20 },
      psf7: [],
      metrics: {},
      reason: "슬라이더를 움직여 BM aperture와 offset의 영향을 확인합니다.",
      source: "Fast estimate",
    },
  };

  // ASM 파동광학 기반 설계 후보 (검증된 물리 결과)
  const presets = [
    { name: "Baseline", tag: "현재 양산 설계", params: { delta_bm1: 0, delta_bm2: 0, w1: 10, w2: 10, d: 20 } },
    { name: "Offset -3", tag: "BM1 오프셋 (최적)", params: { delta_bm1: -3, delta_bm2: 0, w1: 10, w2: 10, d: 20 } },
    { name: "Offset +3", tag: "BM1 오프셋 (대칭)", params: { delta_bm1: 3, delta_bm2: 0, w1: 10, w2: 10, d: 20 } },
    { name: "Narrow w=6", tag: "좁은 아퍼처 (누화 차단)", params: { delta_bm1: 0, delta_bm2: 0, w1: 6, w2: 6, d: 20 } },
    { name: "Offset+Wide", tag: "오프셋+넓힘", params: { delta_bm1: 5, delta_bm2: 0, w1: 15, w2: 15, d: 20 } },
    { name: "Asym w1=10,w2=18", tag: "비대칭 아퍼처", params: { delta_bm1: 0, delta_bm2: 0, w1: 10, w2: 18, d: 20 } },
    { name: "Wide w=18", tag: "넓은 아퍼처 (광량 우선)", params: { delta_bm1: 0, delta_bm2: 0, w1: 18, w2: 18, d: 20 } },
  ];

  function $(id) {
    return document.getElementById(id);
  }

  function pct(v) {
    return `${Math.round((v || 0) * 100)}%`;
  }

  function fixed(v, digits = 3) {
    return Number.isFinite(v) ? v.toFixed(digits) : "-";
  }

  async function evaluateFast(params) {
    // 통합 추론 API (PINN > ASM 자동 선택)
    try {
      const r = await ApiClient.predict({
        delta_bm1: params.delta_bm1, delta_bm2: params.delta_bm2,
        w1: params.w1, w2: params.w2, use_ar: true,
      });
      return {
        psf7: r.psf7,
        metrics: { ...r.metrics, mtf_ridge: r.contrast },
        source: r.source,
      };
    } catch (e) {
      console.warn('API failed:', e);
      return {
        psf7: [0.16, 0.15, 0.09, 0.09, 0.15, 0.16, 0.15],
        metrics: { mtf_ridge: 0.28, skewness: 0, throughput: 0.15, crosstalk_ratio: 0 },
        source: "Fallback",
      };
    }
  }

  // evaluateLocal은 삭제됨 — approxModel 사용 금지
  // sensitivity에서는 ASM API를 직접 호출

  function scoreCandidate(metrics) {
    const spec = DesignStore.state.spec;
    const mtfScore = Math.min(1.3, (metrics.mtf_ridge || 0) / spec.mtf_min);
    const tScore = Math.min(1.3, (metrics.throughput || 0) / spec.T_min);
    const skewScore = Math.min(1.3, spec.skew_max / Math.max(0.001, Math.abs(metrics.skewness || 0.001)));
    const xtPenalty = Math.max(0, (metrics.crosstalk_ratio || 0) - 0.18) * 0.8;
    return Math.max(0, (mtfScore * 42 + tScore * 28 + skewScore * 25 - xtPenalty * 100));
  }

  function reasonFor(candidate) {
    const m = candidate.metrics;
    const bits = [];
    if ((m.mtf_ridge || 0) >= DesignStore.state.spec.mtf_min) bits.push("MTF 목표 통과");
    if ((m.throughput || 0) >= DesignStore.state.spec.T_min) bits.push("광량 목표 통과");
    if (Math.abs(m.skewness || 0) <= DesignStore.state.spec.skew_max) bits.push("skew 여유");
    if ((m.crosstalk_ratio || 0) <= 0.18) bits.push("crosstalk 안정");
    return bits.length ? bits.join(", ") : "목표에 가까운 탐색 후보입니다.";
  }

  function constraintStatus(params) {
    const deltaOk = Math.abs(params.delta_bm1) <= params.w1 / 2 && Math.abs(params.delta_bm2) <= params.w2 / 2;
    const widthOk = params.w1 >= 5 && params.w1 <= 20 && params.w2 >= 5 && params.w2 <= 20;
    const theta = Math.atan(params.w1 / (2 * params.d)) * 180 / Math.PI;
    return {
      ok: deltaOk && widthOk && theta <= 41.1,
      theta,
    };
  }

  function badgesFor(candidate) {
    const m = candidate.metrics;
    const constraints = constraintStatus(candidate.params);
    const badges = [];
    badges.push({ label: candidate.source || "Fast estimate", warn: false });
    badges.push({ label: constraints.ok ? "Constraint OK" : "Constraint warning", warn: !constraints.ok });
    if ((m.crosstalk_ratio || 0) <= 0.18) badges.push({ label: "Low XT", warn: false });
    if ((m.throughput || 0) < DesignStore.state.spec.T_min) badges.push({ label: "T risk", warn: true });
    return badges;
  }

  function selectedReadiness(design) {
    const m = design.metrics || {};
    const spec = DesignStore.state.spec;
    const constraints = constraintStatus(design.params);
    const misses = [];
    if (!constraints.ok) misses.push("constraint");
    if ((m.mtf_ridge || 0) < spec.mtf_min) misses.push("MTF");
    if ((m.throughput || 0) < spec.T_min) misses.push("throughput");
    if (Math.abs(m.skewness || 0) > spec.skew_max) misses.push("skew");
    return {
      ok: misses.length === 0,
      misses,
      constraints,
    };
  }

  async function evaluateCandidate(base, idx) {
    const fast = await evaluateFast(base.params);
    const candidate = {
      id: `candidate-${idx}`,
      name: base.name,
      tag: base.tag,
      params: base.params,
      psf7: fast.psf7,
      metrics: fast.metrics,
      source: fast.source,
    };
    candidate.score = scoreCandidate(candidate.metrics);
    candidate.reason = reasonFor(candidate);
    return candidate;
  }

  async function runReverseDesign() {
    const btn = $("btn-run");
    btn.disabled = true;
    $("status-line").textContent = "후보를 평가하는 중입니다. 빠른 surrogate로 먼저 정렬합니다.";

    const evaluated = [];
    for (let i = 0; i < presets.length; i += 1) {
      evaluated.push(await evaluateCandidate(presets[i], i + 1));
    }

    evaluated.sort((a, b) => b.score - a.score);
    evaluated.forEach((candidate, index) => {
      candidate.rank = index + 1;
    });

    state.candidates = evaluated;
    state.selectedId = evaluated[0]?.id || null;
    $("candidate-count").textContent = String(evaluated.length);
    $("status-line").textContent = `추천 후보 ${evaluated.length}개를 만들었습니다. 상위 후보는 ${evaluated[0]?.name || "-"}입니다.`;
    btn.disabled = false;
    render();
  }

  function getSelectedDesign() {
    if (state.mode === "forward") return state.current;
    return state.candidates.find((candidate) => candidate.id === state.selectedId) || state.current;
  }

  function renderCandidates() {
    const list = $("candidate-list");
    if (!state.candidates.length) {
      list.innerHTML = `<div class="candidate-card"><p class="candidate-reason">후보 생성 버튼을 누르면 목표에 맞는 BM 구조를 비교할 수 있습니다.</p></div>`;
      return;
    }

    list.innerHTML = state.candidates.map((candidate) => `
      <article class="candidate-card ${candidate.id === state.selectedId ? "selected" : ""}" data-id="${candidate.id}">
        <div class="candidate-top">
          <div>
            <div class="candidate-name">${candidate.rank}. ${candidate.name}</div>
            <div class="candidate-tag">${candidate.tag}</div>
          </div>
          <div class="score-pill">${Math.round(candidate.score)}</div>
        </div>
        <div class="candidate-metrics">
          <span>MTF <strong>${pct(candidate.metrics.mtf_ridge)}</strong></span>
          <span>Skew <strong>${fixed(candidate.metrics.skewness)}</strong></span>
          <span>T <strong>${pct(candidate.metrics.throughput)}</strong></span>
          <span>XT <strong>${pct(candidate.metrics.crosstalk_ratio)}</strong></span>
        </div>
        <p class="candidate-reason">${candidate.reason}</p>
        <div class="candidate-badges">
          ${badgesFor(candidate).map((badge) => `<span class="candidate-badge ${badge.warn ? "warn" : ""}">${badge.label}</span>`).join("")}
        </div>
        <div class="candidate-params">d1=${candidate.params.delta_bm1.toFixed(1)} d2=${candidate.params.delta_bm2.toFixed(1)} w1=${candidate.params.w1.toFixed(1)} w2=${candidate.params.w2.toFixed(1)}</div>
      </article>
    `).join("");

    list.querySelectorAll(".candidate-card[data-id]").forEach((card) => {
      card.addEventListener("click", () => {
        state.selectedId = card.dataset.id;
        render();
      });
    });
  }

  function renderSelected() {
    const design = getSelectedDesign();
    const metrics = design.metrics || {};
    const readiness = selectedReadiness(design);
    const constraints = readiness.constraints;

    $("metric-mtf").textContent = pct(metrics.mtf_ridge);
    $("metric-skew").textContent = fixed(metrics.skewness);
    $("metric-t").textContent = pct(metrics.throughput);
    $("metric-xt").textContent = pct(metrics.crosstalk_ratio);
    $("psf-source").textContent = design.source || "Fast estimate";
    $("constraint-label").textContent = constraints.ok ? `Constraint OK, theta ${constraints.theta.toFixed(1)} deg` : "Constraint warning";
    $("model-status-label").textContent = design.source || "Fast estimate";
    $("current-fp-label").textContent = state.mode === "forward" ? "순방향 설계" : "기준 설계";
    $("best-fp-label").textContent = design.name || "추천 후보";
    renderDecisionSummary(design, readiness);

    drawPsf("psf-chart", design.psf7 || []);
    drawStack("coe-chart", design.params || {});
    drawTradeoff("tradeoff-chart");
    renderSensitivity(design);
    drawFingerprint("fp-original", null, { mtf_ridge: 0.75, skewness: 0, crosstalk_ratio: 0.02 });
    drawFingerprint("fp-current", null, state.current.metrics);
    drawFingerprint("fp-best", null, metrics);
  }

  function renderDecisionSummary(design, readiness) {
    $("summary-title").textContent = readiness.ok ? `${design.name} is ready for ASM verification` : `${design.name} needs review`;
    $("summary-copy").textContent = readiness.ok
      ? "현재 목표와 기본 제약을 통과했습니다. 외부 전달 전에는 ASM wave optics와 PINN/FNO cross-check를 실행하는 흐름이 적합합니다."
      : `검토 항목: ${readiness.misses.join(", ")}. 목표 슬라이더나 BM 파라미터를 조정한 뒤 다시 비교하세요.`;
    const chips = [
      { label: `MTF ${pct(design.metrics.mtf_ridge)}`, warn: (design.metrics.mtf_ridge || 0) < DesignStore.state.spec.mtf_min },
      { label: `T ${pct(design.metrics.throughput)}`, warn: (design.metrics.throughput || 0) < DesignStore.state.spec.T_min },
      { label: `Skew ${fixed(design.metrics.skewness)}`, warn: Math.abs(design.metrics.skewness || 0) > DesignStore.state.spec.skew_max },
      { label: readiness.constraints.ok ? "Constraint OK" : "Constraint warning", warn: !readiness.constraints.ok },
    ];
    $("summary-chips").innerHTML = chips.map((chip) => `<span class="summary-chip ${chip.warn ? "warn" : ""}">${chip.label}</span>`).join("");
    $("verify-asm").classList.toggle("ready", readiness.ok);
  }

  function exportReport() {
    const design = getSelectedDesign();
    const payload = {
      generated_at: new Date().toISOString(),
      mode: state.mode,
      spec: DesignStore.state.spec,
      selected: {
        name: design.name,
        params: design.params,
        metrics: design.metrics,
        source: design.source,
        readiness: selectedReadiness(design),
      },
      candidates: state.candidates.map((candidate) => ({
        rank: candidate.rank,
        name: candidate.name,
        tag: candidate.tag,
        params: candidate.params,
        metrics: candidate.metrics,
        score: candidate.score,
        reason: candidate.reason,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bm_design_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    $("status-line").textContent = "선택 후보와 비교 결과를 JSON report로 내보냈습니다.";
  }

  function drawTradeoff(canvasId) {
    const canvas = $(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { left: 42, right: 18, top: 18, bottom: 34 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fbfffd";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#cfe0dc";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const x = pad.left + (i / 4) * plotW;
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    ctx.fillStyle = "#66746f";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("MTF", pad.left + plotW / 2, H - 10);
    ctx.save();
    ctx.translate(13, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Throughput", 0, 0);
    ctx.restore();

    const points = state.candidates.length ? state.candidates : [state.current];
    points.forEach((candidate) => {
      const m = candidate.metrics || {};
      const x = pad.left + Math.max(0, Math.min(1, m.mtf_ridge || 0)) * plotW;
      const y = pad.top + plotH - Math.max(0, Math.min(1, m.throughput || 0)) * plotH;
      const selected = candidate.id === getSelectedDesign().id;
      ctx.beginPath();
      ctx.arc(x, y, selected ? 8 : 6, 0, Math.PI * 2);
      ctx.fillStyle = selected ? "#0f8b8d" : "#3f8f45";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#17211d";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(candidate.name || "Manual", x + 10, y + 4);
    });
  }

  async function renderSensitivity(design) {
    const el = $("sensitivity-list");
    if (!el) return;
    const base = design.params || state.current.params;
    const baseResult = await evaluateFast(base);
    const baseMtf = baseResult.metrics.mtf_ridge || 0;
    const params = ["delta_bm1", "delta_bm2", "w1", "w2"];
    const rows = [];
    for (const key of params) {
      const step = key.startsWith("w") ? 1.0 : 0.5;
      const next = { ...base, [key]: Math.max(key.startsWith("w") ? 5 : -10, Math.min(key.startsWith("w") ? 20 : 10, base[key] + step)) };
      const nextResult = await evaluateFast(next);
      const mtf = nextResult.metrics.mtf_ridge || 0;
      rows.push({ key, delta: mtf - baseMtf });
    }
    const max = Math.max(...rows.map((r) => Math.abs(r.delta)), 0.001);
    el.innerHTML = rows.map((row) => {
      const width = Math.max(4, Math.abs(row.delta) / max * 100);
      const klass = row.delta < 0 ? "negative" : "";
      return `
        <div class="sensitivity-row">
          <span>${row.key.replace("delta_", "d_")}</span>
          <div class="sensitivity-track"><div class="sensitivity-fill ${klass}" style="width:${width}%"></div></div>
          <span>${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(3)}</span>
        </div>
      `;
    }).join("");
  }

  function renderMode() {
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === state.mode);
    });
    $("reverse-controls").classList.toggle("hidden", state.mode !== "reverse");
    $("forward-controls").classList.toggle("hidden", state.mode !== "forward");
    $("stage-title").textContent = state.mode === "reverse" ? "역설계" : "순방향 설계";
    $("stage-kicker").textContent = state.mode === "reverse" ? "목표 성능에서 BM 후보 찾기" : "BM 파라미터를 움직이며 즉시 평가";
    $("btn-run").textContent = state.mode === "reverse" ? "후보 생성" : "현재 설계 평가";
    $("stage-copy").textContent = state.mode === "reverse"
      ? "목표 성능과 공정 제약을 기준으로 BM 후보를 비교합니다. 수치는 um 단위입니다."
      : "BM aperture 폭과 offset을 직접 조정하고 PSF, 지문 응답, local sensitivity를 즉시 확인합니다.";
  }

  function render() {
    renderMode();
    renderCandidates();
    renderSelected();
  }

  function setSpecFromControls() {
    const mtf = Number($("spec-mtf").value) / 100;
    const skew = Number($("spec-skew").value) / 100;
    const throughput = Number($("spec-t").value) / 100;
    DesignStore.setSpec({ mtf_min: mtf, skew_max: skew, T_min: throughput });
    $("spec-mtf-val").textContent = `${Math.round(mtf * 100)}%`;
    $("spec-skew-val").textContent = skew.toFixed(2);
    $("spec-t-val").textContent = `${Math.round(throughput * 100)}%`;
  }

  async function setForwardFromControls() {
    const params = {
      delta_bm1: Number($("param-d1").value) / 10,
      delta_bm2: Number($("param-d2").value) / 10,
      w1: Number($("param-w1").value) / 10,
      w2: Number($("param-w2").value) / 10,
      d: 20,
    };
    const result = await evaluateFast(params);
    state.current = { ...state.current, params, psf7: result.psf7, metrics: result.metrics, source: result.source };
    $("param-d1-val").textContent = `${params.delta_bm1.toFixed(1)} um`;
    $("param-d2-val").textContent = `${params.delta_bm2.toFixed(1)} um`;
    $("param-w1-val").textContent = `${params.w1.toFixed(1)} um`;
    $("param-w2-val").textContent = `${params.w2.toFixed(1)} um`;
    DesignStore.setExploreParams(params);
  }

  function bind() {
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.mode = btn.dataset.mode;
        render();
      });
    });

    ["spec-mtf", "spec-skew", "spec-t"].forEach((id) => {
      $(id).addEventListener("input", () => {
        setSpecFromControls();
        if (state.candidates.length) runReverseDesign();
      });
    });

    ["param-d1", "param-d2", "param-w1", "param-w2"].forEach((id) => {
      $(id).addEventListener("input", async () => {
        await setForwardFromControls();
        if (state.mode !== "forward") state.mode = "forward";
        render();
      });
    });

    $("btn-run").addEventListener("click", async () => {
      if (state.mode === "reverse") {
        runReverseDesign();
      } else {
        await setForwardFromControls();
        $("status-line").textContent = "현재 BM 파라미터를 빠른 순방향 모델로 평가했습니다.";
        render();
      }
    });

    $("btn-export-report").addEventListener("click", exportReport);
  }

  async function init() {
    setSpecFromControls();
    await setForwardFromControls();
    bind();
    runReverseDesign();
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", StudioApp.init);
