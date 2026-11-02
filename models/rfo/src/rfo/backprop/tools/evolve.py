import torch
import torch.nn.functional as F


def logits_update(logits, step=0, total_steps=50, temp=1, prot_dim=20):
    """
    Transition phase from raw logits to a smoothed representation.
    Linearly interpolates between the raw logits and their softmax.
    """
    protein_logits = logits[..., :prot_dim]
    other_zeros = torch.zeros_like(logits[..., prot_dim:])
    coeff = (step + 1) / total_steps
    seq_soft = F.softmax(protein_logits / temp, dim=-1)
    protein_repre = (1 - coeff) * protein_logits + coeff * seq_soft
    new_logits = torch.cat([protein_repre, other_zeros], dim=-1)
    return new_logits


def soft_update(
    logits, step=0, total_steps=50, temp_start=1, temp_end=0.01, prot_dim=20
):
    """
    Soft phase: compute a temperature-controlled softmax.
    The temperature is annealed quadratically from ``temp_start`` to ``temp_end`` over ``total_steps``.
    """
    temp = temp_start + (temp_end - temp_start) * (1 - (step + 1) / total_steps) ** 2
    protein_logits = logits[..., :prot_dim]
    other_zeros = torch.zeros_like(logits[..., prot_dim:])
    seq_soft = F.softmax(protein_logits / temp, dim=-1)
    seq_soft = torch.cat([seq_soft, other_zeros], dim=-1)
    return seq_soft


def hard_update(
    logits, step=0, total_steps=50, temp_start=1, temp_end=0.01, prot_dim=20
):
    """
    Hard phase: obtain a one-hot representation using a straight-through estimator.
    """
    temp = temp_start + (temp_end - temp_start) * (1 - (step + 1) / total_steps) ** 2
    protein_logits = logits[..., :prot_dim]
    other_zeros = torch.zeros_like(logits[..., prot_dim:])
    seq_soft = F.softmax(protein_logits / temp, dim=-1)
    seq_hard_ = F.one_hot(seq_soft.argmax(-1), num_classes=seq_soft.size(-1)).float()
    seq_hard = (seq_hard_ - seq_soft).detach() + seq_soft
    seq_hard = torch.cat([seq_hard, other_zeros], dim=-1)
    return seq_hard


def recover_non_interest_restype(seq, init_seq, seq_of_interest):
    """
    Recover the restype sequence by replacing positions that are not of interest
    with the initial sequence.
    """
    if seq_of_interest.dim() == 1:
        seq_of_interest = seq_of_interest.unsqueeze(-1)

    mask = seq_of_interest.float()
    recovered_seq = mask * seq + (1 - mask) * init_seq
    return recovered_seq
