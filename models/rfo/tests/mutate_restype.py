import torch
from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from rfo.backprop.tools.mutate import (
    compare_seq_mutation,
    mutation_constraint,
    replace_mutate,
)
from torch.nn import functional as F


def mutate_reslogits(
    old_seq_str: str,
    res_logits: torch.Tensor,
    encoding=None,
    random_mut: bool = False,
    allowed_mutate_mask: torch.Tensor = None,
    max_mut: int = None,
    pos_mut_prob: torch.Tensor = None,
    prot_dim: int = 20,
    full_dim: int = 32,
    force_mutate: bool = False,
):
    """
    Mutate a sequence based on the predicted residue types.
    """
    if res_logits is None:
        L = len(old_seq_str)
        res_logits_ = torch.ones(L, full_dim)
    else:
        L = res_logits.size(0)
        res_logits_ = res_logits.clone()

    if force_mutate:
        old_seq_digit = torch.tensor(encoding.encode(old_seq_str), dtype=torch.int64)
        old_seq_one_hot = F.one_hot(old_seq_digit, num_classes=full_dim).float()
        res_logits_ = res_logits_ - old_seq_one_hot * 1e6
        print("res_logits_", res_logits_)

    if max_mut is None:
        max_mut = L

    if allowed_mutate_mask is None:
        allowed_mutate_mask = torch.ones(L, dtype=torch.bool, device=res_logits.device)

    if random_mut:
        res_prob = F.softmax(res_logits_[:, :prot_dim], dim=-1)
        sample_residx = torch.multinomial(res_prob, num_samples=1).squeeze(-1)
    else:
        sample_residx = torch.argmax(res_logits_[:, :prot_dim], dim=-1)

    sample_residx_allowed = sample_residx[allowed_mutate_mask].cpu().numpy()
    sample_mut_seq_allowed = encoding.decode(sample_residx_allowed)
    sample_mut_seq_allowed = "".join(
        [DICT_THREE_TO_ONE.get(res, "X") for res in sample_mut_seq_allowed]
    )

    new_seq_str = replace_mutate(
        old_seq_str, sample_mut_seq_allowed, allowed_mutate_mask
    )

    mutate_num = len(compare_seq_mutation(old_seq_str, new_seq_str, print_num=0))

    if mutate_num == 0:
        return res_logits, old_seq_str
    print(f"Mutated {mutate_num} positions.")
    constrained_seq, actual_mut_mask = mutation_constraint(
        old_seq_str, new_seq_str, max_mut, pos_mut_prob
    )
    sample_residx_one_hot = (
        F.one_hot(sample_residx, num_classes=full_dim).float().to(res_logits.device)
    )
    mut_replaced_res_logits = torch.where(
        actual_mut_mask.unsqueeze(-1).to(res_logits.device),
        sample_residx_one_hot,
        res_logits,
    )

    return mut_replaced_res_logits, constrained_seq


if __name__ == "__main__":
    encoding = AF3SequenceEncoding()

    old_seq_str = "AAAAAAAA"

    L = len(old_seq_str)
    full_dim = 32
    logits_prot = torch.randn(L, 20)
    logits_other = torch.zeros(L, full_dim - 20)
    res_logits = torch.cat([logits_prot, logits_other], dim=-1)

    allowed_mutate_mask = torch.tensor(
        [False, False, True, False, False, True, False, True], dtype=torch.bool
    )

    max_mut = 2

    pos_mut_prob = None

    mutated_logits, mutated_seq = mutate_reslogits(
        old_seq_str=old_seq_str,
        res_logits=res_logits,
        encoding=encoding,
        random_mut=False,
        allowed_mutate_mask=allowed_mutate_mask,
        max_mut=max_mut,
        pos_mut_prob=pos_mut_prob,
        prot_dim=20,
        full_dim=full_dim,
        force_mutate=True,
    )

    compare_seq_mutation(old_seq_str, mutated_seq, print_num=5)

    print("Original sequence:", old_seq_str)
    print("Mutated sequence:", mutated_seq)
    print("Mutated logits:")
    print(mutated_logits)
