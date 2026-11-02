from collections import Counter
from math import ceil

import torch
import torch.nn.functional as F
from atomworks.constants import DICT_THREE_TO_ONE

from rfo.backprop.tools.utils import ensure_probability, index2mask

DICT_ONE_TO_THREE = {v: k for k, v in DICT_THREE_TO_ONE.items()}


def apply_mutations(old_seq_str, actions, encoding):
    """
    Apply mutations to a sequence given (position, aa_index) list.

    Args:
        old_seq_str: Original sequence string (1-letter).
        actions: List of (position, aa_index).
        encoding: Encoder object with ``.decode`` (3-letter) and DICT_THREE_TO_ONE.

    Returns:
        Mutated sequence string.
    """
    aa_codes = []
    for pos, aa_idx in actions:
        aa_3 = encoding.decode([aa_idx])[0]
        aa_1 = DICT_THREE_TO_ONE.get(aa_3, "X")
        aa_codes.append((pos, aa_1))

    seq_list = list(old_seq_str)
    for pos, aa in aa_codes:
        seq_list[pos] = aa
    return "".join(seq_list)


def replace_mutate(
    old_full_seq: str, mutations: str, allowed_mutate_mask: torch.Tensor
):
    """
    Mutate ``old_full_seq`` with the mutations at the ``allowed_mutate_mask`` positions.

    The length of ``mutations`` must equal the number of True in ``allowed_mutate_mask``.
    """
    mut_indices = [i for i, m in enumerate(allowed_mutate_mask) if m]
    seq_list = list(old_full_seq)
    for i, mut in zip(mut_indices, mutations):
        seq_list[i] = mut
    return "".join(seq_list)


def compare_seq_mutation(old_seq_str, current_seq_str, print_num=5):
    """
    Compare two sequences and print the mutations.

    Returns:
        List of mutation positions.
    """
    if len(old_seq_str) != len(current_seq_str):
        raise ValueError("Sequences must have the same length.")

    mutation_positions = [
        i for i, (a, b) in enumerate(zip(old_seq_str, current_seq_str)) if a != b
    ]
    num_mutations = len(mutation_positions)

    if print_num > 0:
        print(f"Number of mutations: {num_mutations}")

    print_count = 0
    for idx in mutation_positions:
        if print_count < print_num:
            print(f"Mutated at {idx}: {old_seq_str[idx]} -> {current_seq_str[idx]}")
            print_count += 1
        else:
            break

    return mutation_positions


def mutation_constraint(
    before_seq: str,
    after_seq: str,
    max_mut: int = None,
    pos_mut_prob: torch.Tensor = None,
) -> str:
    """
    Compare two sequences and apply a mutation limit based on ``max_mut``.

    Returns:
        Tuple of (constrained sequence, boolean mask of selected mutation positions).
    """
    L = len(before_seq)

    mutation_positions = compare_seq_mutation(before_seq, after_seq, print_num=0)
    num_mutations = len(mutation_positions)
    actual_mut_mask = index2mask(mutation_positions, L)

    if max_mut is None:
        return after_seq, actual_mut_mask
    if num_mutations <= max_mut:
        return after_seq, actual_mut_mask

    if pos_mut_prob is None:
        pos_mut_prob = torch.ones(L) / L
    else:
        pos_mut_prob = ensure_probability(pos_mut_prob, method="sum", dim=-1)

    pos_mut_prob = pos_mut_prob[actual_mut_mask]

    if max_mut < num_mutations:
        selected_indices = torch.multinomial(
            pos_mut_prob, num_samples=max_mut, replacement=False
        )
        selected_positions = [mutation_positions[i] for i in selected_indices.tolist()]
    else:
        selected_positions = mutation_positions

    constrained_mut_list = list(before_seq)
    for pos in selected_positions:
        constrained_mut_list[pos] = after_seq[pos]

    constrained_seq = "".join(constrained_mut_list)
    select_mut_mask = index2mask(selected_positions, L)

    return constrained_seq, select_mut_mask


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

    Args:
        old_seq_str: Original sequence string.
        res_logits: Tensor of shape (L, F) containing residue type logits.
        encoding: An encoding object with ``.encode`` and ``.decode`` methods.
        random_mut: Whether to sample randomly.
        allowed_mutate_mask: Boolean mask of allowed mutation positions.
        max_mut: Maximum number of mutations allowed.
        pos_mut_prob: Probability of mutating each position.
        prot_dim: Number of protein dimensions.
        full_dim: Total number of dimensions.
        force_mutate: If True, penalize positions that match the original.

    Returns:
        Tuple of (mutated logits tensor, mutated sequence string).
    """
    if res_logits is None:
        L = len(old_seq_str)
        res_logits_ = torch.ones(L, full_dim)
    else:
        L = res_logits.size(0)
        res_logits_ = res_logits.clone()

    if force_mutate:
        old_seq_digit = torch.tensor(encoding.encode(old_seq_str), dtype=torch.int64)
        old_seq_one_hot = (
            F.one_hot(old_seq_digit, num_classes=full_dim)
            .float()
            .to(res_logits_.device)
        )
        res_logits_ = res_logits_ - old_seq_one_hot * 1e6

    if max_mut is None:
        max_mut = L

    if allowed_mutate_mask is None:
        allowed_mutate_mask = torch.ones(L, dtype=torch.bool)

    if random_mut:
        res_prob = F.softmax(res_logits_[:, :prot_dim], dim=-1)
        sample_residx = torch.multinomial(res_prob, num_samples=1).squeeze(-1)
    else:
        sample_residx = torch.argmax(res_logits_[:, :prot_dim], dim=-1)

    sample_residx_allowed = sample_residx[allowed_mutate_mask.cpu()].cpu().numpy()
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

    constrained_seq, actual_mut_mask = mutation_constraint(
        old_seq_str, new_seq_str, max_mut, pos_mut_prob
    )
    sample_residx_one_hot = (
        F.one_hot(sample_residx, num_classes=full_dim).float().to(res_logits_.device)
    )
    mut_replaced_res_logits = torch.where(
        actual_mut_mask.unsqueeze(-1).to(res_logits_.device),
        sample_residx_one_hot,
        res_logits_,
    )

    return mut_replaced_res_logits, constrained_seq


def majority_vote_mutations(
    old_seq_str: str,
    gradients: list[torch.Tensor],
    encoding,
    allowed_mutate_mask: torch.Tensor = None,
    max_mut: int = None,
    random_mut: bool = False,
    prot_dim: int = 20,
    full_dim: int = 32,
    force_mutate: bool = False,
):
    """
    Propose mutations from multiple model gradients and keep only those
    where a strict majority of models agree on the same position AND amino acid.

    Args:
        old_seq_str: Current sequence string.
        gradients: List of gradient tensors, one per model (each shape [L, D]).
        encoding: Sequence encoding object.
        allowed_mutate_mask: Boolean mask of mutable positions.
        max_mut: Maximum mutations per individual model proposal.
        random_mut: Whether to sample randomly from gradient-derived logits.
        prot_dim: Number of protein residue types.
        full_dim: Total feature dimension.
        force_mutate: If True, penalize positions matching the original sequence.

    Returns:
        Consensus sequence string with only majority-agreed mutations applied.
    """
    n_models = len(gradients)
    threshold = ceil(n_models / 2)

    proposals = []
    for grad in gradients:
        _, proposed_seq = mutate_reslogits(
            old_seq_str=old_seq_str,
            res_logits=-grad,
            encoding=encoding,
            random_mut=random_mut,
            allowed_mutate_mask=allowed_mutate_mask,
            max_mut=max_mut,
            prot_dim=prot_dim,
            full_dim=full_dim,
            force_mutate=force_mutate,
        )
        proposals.append(proposed_seq)

    consensus_seq = list(old_seq_str)
    for pos in range(len(old_seq_str)):
        votes = Counter(p[pos] for p in proposals)
        best_aa, count = votes.most_common(1)[0]
        if best_aa != old_seq_str[pos] and count >= threshold:
            consensus_seq[pos] = best_aa

    return "".join(consensus_seq)
