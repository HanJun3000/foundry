import os
import os.path as osp
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rf3.utils.io import dump_structures
from torch import Tensor

from rfo.backprop.tools.af3 import decode_restype
from rfo.backprop.tools.mutate import (
    apply_mutations,
)
from rfo.backprop.tools.utils import (
    compute_saliency_pos_aa,
    norm_seq_grad,
)


def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # e.g., np.bool_ -> bool, np.float32 -> float
    else:
        return obj


class GradProposal:
    """ """

    def __init__(self, algorithm_config):
        self.algorithm_config = algorithm_config

    def _eval_sequence(
        self,
        folding_model,
        my_objective,
        seq_str: str,
        protein_mask: Tensor,
        cif_add_info=None,
    ):
        trunk, conf, pipe, components = folding_model.project_with_cif_reconstruction(
            seq=seq_str, cif_add_info=cif_add_info, protein_mask=protein_mask
        )
        restype = (
            trunk["f"]["restype"].clone().detach().float().to(folding_model.device)
        )
        restype.requires_grad_(True)
        trunk["f"]["restype"] = restype

        loss, add_info = my_objective(folding_model, trunk, conf, pipe)
        loss.backward()

        grad = -restype.grad[:, :20].detach()
        grad = norm_seq_grad(grad)
        restype.grad.zero_()

        return loss.item(), grad, add_info, components

    def optimize(
        self,
        folding_model,
        my_objective,
        init_restype: Tensor,
        trunk_input: Dict[str, Any],
        opt_steps: Dict[str, Any],
        seq_of_interest_mask: Optional[Tensor] = None,
        cif_add_info: Optional[Dict[str, Any]] = None,
        output_path: str = "./output",
        **kwargs,
    ):
        # set up the post analysis directories
        modelhub_pred_dir = osp.join(output_path, "modelhub_pred")
        af3_pred_dir = osp.join(output_path, "af3_pred")
        os.makedirs(modelhub_pred_dir, exist_ok=True)
        os.makedirs(af3_pred_dir, exist_ok=True)

        encoding = folding_model.encoding
        protein_mask = trunk_input["f"]["is_protein"]

        # step 0: evaluate candidate sequence
        curr_seq = decode_restype(init_restype, encoding, protein_mask)
        loss, grad_token, add_info, components = self._eval_sequence(
            folding_model, my_objective, curr_seq, protein_mask, cif_add_info
        )
        best_iPAE = add_info.get("ipae_min", float("inf"))
        atom_array_stack = add_info.get("atom_array_stack")
        if atom_array_stack is not None:
            dump_structures(
                atom_array_stack,
                base_path=osp.join(modelhub_pred_dir, "0.cif"),
                one_model_per_file=False,
                file_type="cif",
            )
        else:
            print("Warning: No structure available for step 0, skipping CIF save")

        history: List[Dict[str, Any]] = []
        accept_traj: List[Dict[str, Any]] = []
        history.append(
            dict(
                name="init_seq",
                step=0,
                seq=curr_seq,
                ipae_mean=add_info.get("ipae_mean", float("inf")),
                ipae_min=add_info.get("ipae_min", float("inf")),
                loss=loss,
                components=components,
            )
        )

        seen = {curr_seq}
        nsteps = opt_steps["total_steps"]
        k_mut_each_round = opt_steps["k_mut_each_round"]

        for step in range(1, nsteps + 1):
            # generate a proposal mutation pool
            keep_mask, grad_neg = compute_saliency_pos_aa(
                grad_token,
                seq_of_interest_mask,
                threshold=self.algorithm_config["pool_threshold_grad"],
                min_num_candidates=self.algorithm_config["pool_min_cands"],
            )
            pos_aa_pairs = torch.nonzero(keep_mask, as_tuple=False)
            if len(pos_aa_pairs) == 0:
                break

            # select k_mut_each_round positions according to the gradient amp
            amp_vals = grad_neg[pos_aa_pairs[:, 0], pos_aa_pairs[:, 1]]
            k = min(k_mut_each_round, len(amp_vals))
            topk_idx = torch.topk(amp_vals, k).indices
            actions_batch = [tuple(map(int, pos_aa_pairs[i])) for i in topk_idx]
            best_cand = None

            for inner_cnt, (pos, aa_idx) in enumerate(actions_batch, start=1):
                cand_name = f"outer_{step}_inner_{inner_cnt}"
                new_seq = apply_mutations(curr_seq, [(pos, aa_idx)], encoding)
                if new_seq in seen:
                    continue
                cand_loss, cand_grad, cand_info, cand_comps = self._eval_sequence(
                    folding_model, my_objective, new_seq, protein_mask, cif_add_info
                )
                cand_atom_array_stack = cand_info.get("atom_array_stack")
                if cand_atom_array_stack is not None:
                    dump_structures(
                        cand_atom_array_stack,
                        base_path=osp.join(modelhub_pred_dir, f"{cand_name}.cif"),
                        one_model_per_file=False,
                        file_type="cif",
                    )
                hist_rec = dict(
                    name=cand_name,
                    step=step,
                    sequence=new_seq,
                    mutation={"position": pos, "aa_index": aa_idx},
                    loss=cand_loss,
                    ipae_mean=cand_info.get("ipae_mean", float("inf")),
                    ipae_min=cand_info.get("ipae_min", float("inf")),
                    components=cand_comps,
                )
                history.append(hist_rec)
                if best_cand is None or cand_info.get(
                    "ipae_min", float("inf")
                ) < best_cand[3].get("ipae_min", float("inf")):
                    best_cand = (
                        new_seq,
                        cand_loss,
                        cand_grad,
                        cand_info,
                        cand_comps,
                        cand_name,
                    )
                    best_hist_idx = len(history) - 1

            if best_cand is None:
                print(f"[step {step}] no unseen candidates, continue.")
                continue

            improved = best_cand[3].get("ipae_min", float("inf")) < best_iPAE
            if improved:
                (curr_seq, loss, grad_token, info, comps, cand_name) = best_cand
                best_iPAE = info.get("ipae_min", float("inf"))
                accept_traj.append(history[best_hist_idx])

        history = [to_python_type(h) for h in history]
        accept_traj = [to_python_type(a) for a in accept_traj]
        return history, accept_traj
