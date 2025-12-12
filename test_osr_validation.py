import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def log_sinkhorn(cost, epsilon=0.05, iterations=5):
    """
    안정형 로그-싱크혼.
    cost: [B, E]
    반환: log_P, where P is approximately doubly stochastic with
           row sums = 1, col sums = B / E.
    """
    B, E = cost.shape
    log_K = -cost / epsilon  # [B, E]

    # 초기 로그 스케일
    log_u = torch.zeros(B, device=cost.device)
    log_v = torch.zeros(E, device=cost.device)

    # 목표 열 합: B / E
    log_col_mass = math.log(B / E)

    for _ in range(iterations):
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)  # row normalize to 1
        log_v = log_col_mass - torch.logsumexp(
            log_K.transpose(0, 1) + log_u.unsqueeze(0), dim=1
        )  # col normalize to B/E

    log_P = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return log_P


class OSRRouter(nn.Module):
    def __init__(self, dim, num_experts, beta=0.5, epsilon=0.05, sinkhorn_iter=5):
        super().__init__()
        self.num_experts = num_experts
        self.beta = beta
        self.epsilon = epsilon
        self.sinkhorn_iter = sinkhorn_iter

        self.experts = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.orthogonal_(self.experts)

    def forward(self, x):
        # Normalize inputs and experts
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, D]
        e_norm = F.normalize(self.experts, p=2, dim=-1)  # [E, D]

        # Similarity
        sim = torch.matmul(x_norm, e_norm.t())  # [B, E]

        # Expert Gram and repulsion term
        gram = torch.matmul(e_norm, e_norm.t())  # [E, E]
        gram_off = torch.relu(gram - torch.eye(self.num_experts, device=x.device))

        # Repulsive cost: similarity mass onto correlated experts
        repulsion = torch.matmul(sim, gram_off)  # [B, E]

        # Unified cost: prefer high sim, penalize correlated experts
        score = sim - self.beta * repulsion
        cost = -score

        # Log-domain Sinkhorn
        log_P = log_sinkhorn(cost, epsilon=self.epsilon, iterations=self.sinkhorn_iter)
        probs = torch.exp(log_P)

        return probs, sim, gram, gram_off


def run_osr_validation():
    torch.manual_seed(0)

    BATCH = 128
    DIM = 128
    EXPERTS = 8
    STEPS = 400
    LR = 1e-3
    ORTHO_PENALTY = 0.2
    BAL_PENALTY = 0.05
    BETA = 0.6
    EPSILON = 0.05
    SINKHORN_ITER = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = OSRRouter(DIM, EXPERTS, beta=BETA, epsilon=EPSILON, sinkhorn_iter=SINKHORN_ITER).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=LR)

    history = {"loss": [], "cv": [], "ortho": [], "repulsion": []}

    print(f"Running OSR validation on {device}...")
    print(
        f"{'step':^6} | {'task':^9} | {'cv':^10} | {'ortho':^10} | {'repulsion':^10}"
    )
    print("-" * 60)

    for step in range(STEPS):
        x = torch.randn(BATCH, DIM, device=device)
        probs, sim, gram, gram_off = router(x)

        # Targets: greedy from similarity (specialization)
        targets = sim.argmax(dim=1)
        log_probs = torch.log(probs + 1e-9)
        task_loss = F.nll_loss(log_probs, targets)

        # Orthogonality penalty on experts
        ortho_loss = (gram - torch.eye(EXPERTS, device=device)).pow(2).mean()

        # Column-wise usage balance penalty (guard against numerical drift)
        usage = probs.mean(dim=0)
        balance_loss = (usage - 1.0 / EXPERTS).pow(2).mean()

        total_loss = task_loss + ORTHO_PENALTY * ortho_loss + BAL_PENALTY * balance_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Re-orthogonalize experts (maintenance step)
        with torch.no_grad():
            q, _ = torch.linalg.qr(router.experts.t(), mode="reduced")
            router.experts.copy_(q.t())

        with torch.no_grad():
            cv = usage.std() / (usage.mean() + 1e-9)
            ortho_err = (gram - torch.eye(EXPERTS, device=device)).norm()
            repulsion_mean = gram_off.mean()

            history["loss"].append(task_loss.item())
            history["cv"].append(cv.item())
            history["ortho"].append(ortho_err.item())
            history["repulsion"].append(repulsion_mean.item())

        if step % 50 == 0:
            print(
                f"{step:^6} | {task_loss.item():^9.4f} | {cv.item():^10.4f} | "
                f"{ortho_err.item():^10.6f} | {repulsion_mean.item():^10.6f}"
            )

    return history


if __name__ == "__main__":
    hist = run_osr_validation()

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].plot(hist["loss"]); ax[0].set_title("Task Loss")
        ax[1].plot(hist["cv"]); ax[1].set_title("Expert CV")
        ax[2].plot(hist["ortho"]); ax[2].set_title("Ortho Error")
        ax[3].plot(hist["repulsion"]); ax[3].set_title("Mean Repulsion")
        fig.tight_layout()
        out_path = os.path.join(os.getcwd(), "osr_validation.png")
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    else:
        out_path = os.path.join(os.getcwd(), "osr_validation.pt")
        torch.save(hist, out_path)
        print(f"Saved metrics to {out_path}")

