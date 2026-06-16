import torch
import torch.nn.functional as F
from src.dto import get_loss_mode_config

def similarity_to_uncertainty(
    sim: torch.Tensor,
    prior_weight: float = 2,
) -> torch.Tensor:
    """
    Convert similarity scores into evidential uncertainty.

    Args:
        sim: [N, N] similarity matrix
        prior_weight: prior uncertainty mass

    Returns:
        uncertainty: [N, N]
    """

    # Positive evidence
    e_pos = torch.exp(
        F.softmax(sim, dim=1)
    )

    # Negative evidence
    e_neg = torch.exp(
        F.softmax(1.0 - sim, dim=1)
    )

    # Total evidence mass
    total_evidence = (
        e_pos
        + e_neg
        + prior_weight
    )

    # Evidential uncertainty
    uncertainty = (
        prior_weight
        / total_evidence
    )

    return uncertainty


def co_cluster_opinion_batch(
    z: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Build intra-class uncertainty graph.

    Args:
        z   : [B, V, D]
            B = batch size
            V = number of views/patches
            D = embedding dimension

        labels: [B]

    Returns:
        uncertainty: [B*V, B*V]
    """

    B, V, D = z.shape

    # Flatten embeddings
    z = z.reshape(B * V, D)


    labels = labels.repeat_interleave(V)

    
    # Pairwise similarity
    sim = torch.matmul(z, z.T)

    # Remove self-similarity
    sim.fill_diagonal_(0.0)

    # Convert similarity to uncertainty
    uncertainty = similarity_to_uncertainty(sim)

    # Keep only intra-class relations
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    uncertainty[~mask] = 0.0
    

    return uncertainty

def compute_lambda(uncertainty, epoch, T):

    N, M = uncertainty.shape

    progress = epoch / T

    idx = torch.argsort(
        uncertainty,
        dim=1,
        descending=True
    )

    ranks = torch.zeros_like(
        uncertainty,
        dtype=torch.float32
    )

    rank_values = torch.arange(
        1,
        M + 1,
        device=uncertainty.device,
        dtype=torch.float32
    )

    ranks.scatter_(
        1,
        idx,
        rank_values.unsqueeze(0).expand(N, -1)
    )
      # normalized rank in [0,1]
    hardness = (ranks-1) / max(M-1 , 1)

    # paper-style curriculum weight
    Lambda = 1.0 + torch.tanh(
        progress * hardness
    )
    

    return Lambda

 



def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    img_ids: torch.Tensor,
    u: torch.Tensor,
    epoch: int,
    temperature: float = 0.1,
    same_img_weight: float = 1,
    eps: float = 1e-8,
    T: int = 50,
    mode: str = "full",
    return_stats: bool = True,
):
    B, V, D = z.shape
    device = z.device

    z = z.reshape(B * V, D)
    z = F.normalize(z, dim=-1)   # important

    labels = labels.repeat_interleave(V)
    img_ids = img_ids.repeat_interleave(V)

    N = z.size(0)

    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    eye = torch.eye(N, device=device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    exp_sim = torch.exp(sim)

    same_class = (labels[:, None] == labels[None, :]) & (~eye)
    same_img = img_ids[:, None] == img_ids[None, :]

    strong_pos = same_class & same_img
    weak_pos = same_class & (~same_img)

    pos_mask = same_class
    neg_mask = ~same_class & (~eye)

    cfg = get_loss_mode_config(mode)

    if cfg.use_only_uncertainty and u is not None:
        diff_img_weight = u.detach()
    elif cfg.use_uncertainty and u is not None:
        diff_img_weight = compute_lambda(u.detach(), epoch=epoch, T=T)
    else:
        diff_img_weight = torch.ones_like(exp_sim)

    if cfg.use_pos_weighting:
        pos_weights = (
            same_img_weight * strong_pos.float()
            + diff_img_weight * weak_pos.float()
        )
    else:
        pos_weights = pos_mask.float()

    if cfg.use_neg_weight:
        num_neg = neg_mask.sum(dim=1, keepdim=True)
        neg_logits = sim.masked_fill(~neg_mask, float("-inf"))
        neg_weights = torch.softmax(neg_logits, dim=1)
        neg_weights = neg_weights * torch.sqrt(num_neg)
        neg_weights = torch.nan_to_num(neg_weights, nan=0.0)
        neg_term = exp_sim * neg_mask.float() * neg_weights
    else:
        neg_term = exp_sim * neg_mask.float()
        neg_weights = None

    pos_term = exp_sim * pos_weights

    if cfg.use_neg_weight:
      denominator = pos_term + neg_term.sum(dim=1, keepdim=True) + eps
    else: 
      denominator = exp_sim.sum(dim=1, keepdim=True) + eps

    # only positives contribute to numerator
    log_prob = torch.log((pos_term + eps) / denominator)

    loss_matrix = -(log_prob * pos_mask.float())
    num_pos = pos_mask.sum(dim=1)
    valid = num_pos > 0

    loss = (loss_matrix.sum(dim=1)[valid] / (num_pos[valid] + eps)).mean()

    if return_stats:

        def safe_mean(x):
            return (
                x.mean().detach().cpu().item()
                if x.numel() > 0
                else 0.0
            )

        def safe_std(x):
            return (
                x.std().detach().cpu().item()
                if x.numel() > 1
                else 0.0
            )

        stats = {
            "epoch": epoch,
            "mode": mode,
            "loss": loss.item(),
            "num_pos": num_pos.float().mean().item(),
            "pos_ratio": pos_mask.float().mean().item(),
            "neg_ratio": neg_mask.float().mean().item(),
        }

        if cfg.use_uncertainty and u is not None:

            weak_vals = diff_img_weight[weak_pos]

            stats["diff_weight_mean"] = safe_mean(weak_vals)
            stats["diff_weight_std"] = safe_std(weak_vals)

        else:

            stats["diff_weight_mean"] = 0.0
            stats["diff_weight_std"] = 0.0

        if cfg.use_neg_weight and neg_weights is not None:

            neg_vals = neg_weights[neg_mask]

            stats["neg_weight_mean"] = safe_mean(neg_vals)
            stats["neg_weight_std"] = safe_std(neg_vals)

        else:

            stats["neg_weight_mean"] = 0.0
            stats["neg_weight_std"] = 0.0

        return loss, stats

    return loss