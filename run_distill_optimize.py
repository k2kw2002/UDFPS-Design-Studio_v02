"""
run_distill_optimize.py - PINN → FNO 증류 + BoTorch 역설계
=============================================================
Step 1: 학습된 Parametric PINN에서 teacher data 생성
Step 2: FNO Surrogate 증류 (0.1ms 추론)
Step 3: BoTorch qNEHVI 3-objective 역설계
Step 4: Top-5 최적 설계 저장

Usage:
  python run_distill_optimize.py --teacher_samples 5000 --fno_epochs 3000 --bo_iters 50 --device cuda
"""

import argparse, time, logging, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.core.parametric_pinn import ParametricHelmholtzPINN
from backend.core.fno_model import FNOSurrogate
from backend.physics.psf_metrics import PSFMetrics
from backend.data.lhs_sampler import generate_lhs_samples, samples_to_numpy
from backend.harness.physical_validator import BMPhysicalValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_samples', type=int, default=5000)
parser.add_argument('--fno_epochs', type=int, default=3000)
parser.add_argument('--bo_iters', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


def step1_generate_teacher():
    """Parametric PINN에서 teacher data 생성."""
    log.info("Step 1: Generating teacher data from Parametric PINN...")

    ckpt = torch.load("parametric_pinn_ckpt.pt", map_location=device, weights_only=False)
    model = ParametricHelmholtzPINN(hidden_dim=128, num_layers=4, num_freqs=48)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    samples = generate_lhs_samples(n_samples=args.teacher_samples, validate=True)
    X = samples_to_numpy(samples)  # (N, 4)

    Y = np.zeros((len(X), 7))
    t0 = time.time()
    for i in range(len(X)):
        d1, d2, w1, w2 = X[i]
        psf7 = model.predict_psf7(d1, d2, w1, w2, device=str(device))
        Y[i] = psf7

        if (i + 1) % 500 == 0:
            log.info(f"  {i+1}/{len(X)} ({time.time()-t0:.0f}s)")

    log.info(f"  Teacher data: {X.shape} -> {Y.shape} ({time.time()-t0:.0f}s)")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def step2_distill_fno(X, Y):
    """FNO 증류."""
    log.info(f"Step 2: FNO distillation ({args.fno_epochs} epochs)...")

    fno = FNOSurrogate(in_dim=4, out_dim=7, width=32, modes=4, num_layers=3)
    fno.to(device)

    optimizer = optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fno_epochs)

    X_d, Y_d = X.to(device), Y.to(device)
    dataset = torch.utils.data.TensorDataset(X_d, Y_d)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    t0 = time.time()
    for epoch in range(args.fno_epochs):
        fno.train()
        total_loss = 0
        for xb, yb in loader:
            pred = fno(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % (args.fno_epochs // 10) == 0 or epoch == args.fno_epochs - 1:
            avg = total_loss / len(loader)
            log.info(f"  [{epoch:4d}/{args.fno_epochs}] loss={avg:.6f} | {time.time()-t0:.0f}s")

    # 저장
    torch.save({"model_state": fno.state_dict()}, "fno_surrogate_ckpt.pt")
    log.info(f"  Saved: fno_surrogate_ckpt.pt")
    return fno


def step3_botorch_optimize(fno):
    """BoTorch qNEHVI 3-objective."""
    log.info(f"Step 3: BoTorch optimization ({args.bo_iters} iterations)...")

    validator = BMPhysicalValidator()
    metrics_calc = PSFMetrics()

    bounds = torch.tensor([[-10, -10, 5, 5], [10, 10, 20, 20]], dtype=torch.float64)
    ref_point = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)

    def evaluate(X_batch):
        results = []
        fno.eval()
        with torch.no_grad():
            psf_batch = fno(X_batch.float().to(device)).cpu().numpy()
        for i in range(X_batch.shape[0]):
            m = metrics_calc.compute(psf_batch[i])
            results.append([m['mtf_ridge'], m['throughput'], -abs(m['skewness'])])
        return torch.tensor(results, dtype=torch.float64)

    # 초기 LHS
    samples = generate_lhs_samples(n_samples=20, seed=77)
    X = torch.tensor(samples_to_numpy(samples), dtype=torch.float64)
    Y = evaluate(X)

    try:
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        from botorch.optim import optimize_acqf
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

        hv_calc = Hypervolume(ref_point=ref_point)

        for iteration in range(args.bo_iters):
            models = [SingleTaskGP(X, Y[:, j:j+1]) for j in range(3)]
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                continue

            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref_point.tolist(), X_baseline=X)
            candidates, _ = optimize_acqf(
                acqf, bounds=bounds, q=4, num_restarts=5, raw_samples=256)

            Y_new = evaluate(candidates)
            X = torch.cat([X, candidates])
            Y = torch.cat([Y, Y_new])

            if iteration % 10 == 0:
                mask = is_non_dominated(Y)
                hv = hv_calc.compute(Y[mask])
                log.info(f"  [BO {iteration:3d}/{args.bo_iters}] HV={hv:.4f} pareto={mask.sum()}")

        # Top-5
        mask = is_non_dominated(Y)
        pareto_X, pareto_Y = X[mask], Y[mask]
        scores = pareto_Y[:, 0] * 0.4 + pareto_Y[:, 1] * 0.3 + pareto_Y[:, 2] * 0.3
        top5_idx = scores.topk(min(5, len(scores))).indices

        labels = ['A', 'B', 'C', 'D', 'E']
        top5 = []
        for rank, idx in enumerate(top5_idx):
            p, y = pareto_X[idx], pareto_Y[idx]
            top5.append({
                'label': labels[rank],
                'params': {'delta_bm1': float(p[0]), 'delta_bm2': float(p[1]),
                           'w1': float(p[2]), 'w2': float(p[3])},
                'mtf': float(y[0]), 'throughput': float(y[1]), 'skewness': float(-y[2]),
            })
            log.info(f"  {labels[rank]}: d1={p[0]:.1f} w1={p[2]:.1f} MTF={y[0]:.3f}")

        torch.save({"top5": top5}, "botorch_top5.pt")
        log.info(f"  Saved: botorch_top5.pt")

    except ImportError:
        log.warning("  BoTorch not installed, skipping")


def main():
    t0 = time.time()
    X, Y = step1_generate_teacher()
    fno = step2_distill_fno(X, Y)
    step3_botorch_optimize(fno)
    log.info(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
