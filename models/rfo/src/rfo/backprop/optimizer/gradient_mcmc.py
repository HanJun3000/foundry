import os
import os.path as osp
import random as pyrandom
from math import ceil
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from rf3.utils.io import dump_structures
from torch import Tensor

from rfo.backprop.tools.af3 import decode_restype
from rfo.backprop.tools.mutate import mutate_reslogits
from rfo.backprop.tools.utils import norm_seq_grad


def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class GradMCMC:
    """Gradient-guided MCMC with multi-checkpoint consensus support.

    Two orthogonal configuration knobs control multi-model behaviour:

    ``gradient_strategy`` -- how to compute the gradient for mutation proposals:
        - ``single``:  always use the first checkpoint (original behaviour).
        - ``random``:  randomly pick one checkpoint per step (BindCraft-style).
        - ``average``: forward+backward through all checkpoints, average gradients.

    ``acceptance_strategy`` -- how to decide whether to accept a proposed mutation:
        - ``mcmc_majority``:   each model independently runs Metropolis-Hastings on
          its own loss delta; accept only if a strict majority votes accept.
        - ``simple_majority``: accept if a strict majority of models show
          ``prop_loss < curr_loss`` (no temperature, no stochastic acceptance).

    When only a single checkpoint is provided both strategies degenerate to
    standard single-model GradMCMC regardless of the setting.
    """

    def __init__(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.gradient_strategy = getattr(
            algorithm_config, "gradient_strategy", "single"
        )
        self.acceptance_strategy = getattr(
            algorithm_config, "acceptance_strategy", "mcmc_majority"
        )

    # ------------------------------------------------------------------
    # Primitive: evaluate one sequence with one pipeline
    # ------------------------------------------------------------------
    def _eval_single(
        self,
        folding_model,
        pipeline,
        my_objective,
        seq_str,
        protein_mask,
        cif_add_info=None,
    ):
        """Forward + backward through a single pipeline.

        Returns:
            ``(loss, grad, trunk_input, conf_input, pipeline_output,
            add_info, components)``
        """
        trunk_input, conf_input, pipeline_output, components = (
            folding_model.project_with_cif_reconstruction(
                seq=seq_str,
                cif_add_info=cif_add_info,
                protein_mask=protein_mask,
            )
        )
        restype = (
            trunk_input["f"]["restype"]
            .clone()
            .detach()
            .float()
            .to(folding_model.device)
        )
        restype.requires_grad_(True)
        trunk_input["f"]["restype"] = restype

        loss, add_info = my_objective(
            folding_model,
            trunk_input,
            conf_input,
            pipeline_output,
            inference_pipeline=pipeline,
        )
        loss.backward()
        grad = norm_seq_grad(restype.grad).detach()
        restype.grad.zero_()

        return (
            loss.detach().cpu().item(),
            grad,
            trunk_input,
            conf_input,
            pipeline_output,
            add_info,
            components,
        )

    # ------------------------------------------------------------------
    # Gradient strategy: produce a single gradient for proposal
    # ------------------------------------------------------------------
    def _get_proposal_gradient(
        self, folding_model, my_objective, seq_str, protein_mask, cif_add_info
    ):
        """Compute a proposal gradient according to ``self.gradient_strategy``.

        Returns:
            ``(grad, loss_from_proposal_model, add_info, components)``
            where *grad* is the single gradient tensor used for mutation proposal.
        """
        pipelines = folding_model.inference_pipelines
        strategy = self.gradient_strategy

        if strategy == "single" or len(pipelines) == 1:
            loss, grad, _, _, _, info, comps = self._eval_single(
                folding_model,
                pipelines[0],
                my_objective,
                seq_str,
                protein_mask,
                cif_add_info,
            )
            return grad, loss, info, comps

        if strategy == "random":
            idx = pyrandom.randrange(len(pipelines))
            loss, grad, _, _, _, info, comps = self._eval_single(
                folding_model,
                pipelines[idx],
                my_objective,
                seq_str,
                protein_mask,
                cif_add_info,
            )
            return grad, loss, info, comps

        if strategy == "average":
            grads: List[Tensor] = []
            first_info = first_comps = None
            total_loss = 0.0
            for i, pipe in enumerate(pipelines):
                loss_i, grad_i, _, _, _, info_i, comps_i = self._eval_single(
                    folding_model,
                    pipe,
                    my_objective,
                    seq_str,
                    protein_mask,
                    cif_add_info,
                )
                grads.append(grad_i)
                total_loss += loss_i
                if i == 0:
                    first_info, first_comps = info_i, comps_i
            avg_grad = torch.stack(grads).mean(dim=0)
            return avg_grad, total_loss / len(pipelines), first_info, first_comps

        raise ValueError(f"Unknown gradient_strategy: {strategy}")

    # ------------------------------------------------------------------
    # Acceptance strategy: evaluate proposal with ALL models
    # ------------------------------------------------------------------
    def _evaluate_and_accept(
        self,
        folding_model,
        my_objective,
        prop_seq,
        curr_losses,
        protein_mask,
        cif_add_info,
        T,
    ) -> Tuple[bool, List[float], Any, Any]:
        """Evaluate the proposed sequence with every model and decide acceptance.

        Args:
            curr_losses: Per-model losses for the current (accepted) sequence.
            T: Current temperature (only used by ``mcmc_majority``).

        Returns:
            ``(accepted, prop_losses, add_info, components)``
        """
        pipelines = folding_model.inference_pipelines
        n = len(pipelines)
        prop_losses: List[float] = []
        first_info = first_comps = None

        for i, pipe in enumerate(pipelines):
            loss_i, _, _, _, _, info_i, comps_i = self._eval_single(
                folding_model,
                pipe,
                my_objective,
                prop_seq,
                protein_mask,
                cif_add_info,
            )
            prop_losses.append(loss_i)
            if i == 0:
                first_info, first_comps = info_i, comps_i

        # Single-model fast path
        if n == 1:
            delta = prop_losses[0] - curr_losses[0]
            accepted = delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10))
            return accepted, prop_losses, first_info, first_comps

        threshold = ceil(n / 2)

        if self.acceptance_strategy == "mcmc_majority":
            votes = 0
            for i in range(n):
                delta = prop_losses[i] - curr_losses[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    votes += 1
            accepted = votes >= threshold

        elif self.acceptance_strategy == "simple_majority":
            votes = sum(1 for i in range(n) if prop_losses[i] < curr_losses[i])
            accepted = votes >= threshold

        else:
            raise ValueError(f"Unknown acceptance_strategy: {self.acceptance_strategy}")

        return accepted, prop_losses, first_info, first_comps

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    def optimize(
        self,
        folding_model,
        my_objective,
        init_restype: Tensor,
        trunk_input: Dict[str, Any],
        confidence_input: Dict[str, Any],
        pipeline_output: Dict[str, Any],
        opt_steps: Dict[str, Any],
        lr: float = 1e-2,
        seq_of_interest_mask=None,
        cif_add_info=None,
        output_path="./output",
        random_mut=False,
        mutate_each=1,
        save_accept=False,
        **kwargs,
    ):
        modelhub_pred_dir = osp.join(output_path, "modelhub_pred")
        af3_pred_dir = osp.join(output_path, "af3_pred")
        os.makedirs(modelhub_pred_dir, exist_ok=True)
        os.makedirs(af3_pred_dir, exist_ok=True)

        encoding = folding_model.encoding
        protein_mask = trunk_input["f"]["is_protein"]
        n_models = folding_model.n_models
        T_init = opt_steps["T_init"]
        half_lf = opt_steps["half_life"]
        nsteps = opt_steps["total_steps"]

        print(
            f"GradMCMC: {n_models} model(s), "
            f"gradient={self.gradient_strategy}, "
            f"acceptance={self.acceptance_strategy}"
        )

        # ---- step 0: evaluate initial sequence with ALL models ----
        init_seq = decode_restype(init_restype, encoding, protein_mask)
        init_losses: List[float] = []
        first_info = first_comps = None
        for i, pipe in enumerate(folding_model.inference_pipelines):
            loss_i, _, _, _, _, info_i, comps_i = self._eval_single(
                folding_model,
                pipe,
                my_objective,
                init_seq,
                protein_mask,
                cif_add_info,
            )
            init_losses.append(loss_i)
            if i == 0:
                first_info, first_comps = info_i, comps_i

        # Get the proposal gradient for the initial sequence
        init_grad, _, _, _ = self._get_proposal_gradient(
            folding_model,
            my_objective,
            init_seq,
            protein_mask,
            cif_add_info,
        )

        curr = dict(
            seq=init_seq,
            losses=init_losses,
            avg_loss=float(np.mean(init_losses)),
            grad=init_grad,
        )

        # First proposal
        _, prop_seq = mutate_reslogits(
            old_seq_str=curr["seq"],
            res_logits=-curr["grad"],
            encoding=encoding,
            random_mut=random_mut,
            allowed_mutate_mask=seq_of_interest_mask,
            max_mut=mutate_each,
            prot_dim=20,
            full_dim=32,
            force_mutate=True,
        )

        history = [
            {
                "step": 0,
                "name": "init",
                "loss_curr": curr["avg_loss"],
                "loss_prop": curr["avg_loss"],
                "per_model_losses": curr["losses"],
                "accepted": None,
                "n_models": n_models,
                "ipae_mean": first_info.get("ipae_mean", float("inf")),
                "ipae_min": first_info.get("ipae_min", float("inf")),
                "geometric_distances": first_info.get("geometric_distances", {}),
                "substrate_plddt": first_info.get("substrate_plddt", 0.0),
                "components": first_comps,
            }
        ]

        atom_array_stack = first_info.get("atom_array_stack")
        if atom_array_stack is not None:
            dump_structures(
                atom_arrays=atom_array_stack,
                base_path=osp.join(modelhub_pred_dir, "0.cif"),
                one_model_per_file=False,
                file_type="cif",
            )

        accept_traj: List[Dict[str, Any]] = []

        # ---- main loop ----
        for step in range(1, nsteps + 1):
            T = T_init * (np.exp(np.log(0.5) / half_lf) ** (step - 1))

            # 1. Evaluate proposal with ALL models and decide acceptance
            accepted, prop_losses, prop_info, prop_comps = self._evaluate_and_accept(
                folding_model,
                my_objective,
                prop_seq,
                curr["losses"],
                protein_mask,
                cif_add_info,
                T,
            )
            prop_avg_loss = float(np.mean(prop_losses))

            # Save structure
            atom_array_stack = prop_info.get("atom_array_stack")
            if atom_array_stack is not None:
                dump_structures(
                    atom_arrays=atom_array_stack,
                    base_path=osp.join(modelhub_pred_dir, f"{step}.cif"),
                    one_model_per_file=False,
                    file_type="cif",
                )

            # 2. Accept or reject
            if accepted:
                # Get new proposal gradient from the accepted sequence
                new_grad, _, _, _ = self._get_proposal_gradient(
                    folding_model,
                    my_objective,
                    prop_seq,
                    protein_mask,
                    cif_add_info,
                )
                curr.update(
                    seq=prop_seq,
                    losses=prop_losses,
                    avg_loss=prop_avg_loss,
                    grad=new_grad,
                )
                print(
                    f"ACCEPTED [Step {step}] Avg Loss: {prop_avg_loss:.4f}  "
                    f"Per-model: {prop_losses}"
                )
                accept_traj.append(
                    {
                        "step": step,
                        "name": f"{step}",
                        "loss_curr": curr["avg_loss"],
                        "loss_prop": prop_avg_loss,
                        "per_model_losses": prop_losses,
                        "accepted": True,
                        "ipae_mean": prop_info.get("ipae_mean", float("inf")),
                        "ipae_min": prop_info.get("ipae_min", float("inf")),
                        "geometric_distances": prop_info.get("geometric_distances", {}),
                        "substrate_plddt": prop_info.get("substrate_plddt", 0.0),
                        "components": prop_comps,
                    }
                )
            else:
                print(
                    f"REJECTED [Step {step}] Avg Loss: {prop_avg_loss:.4f}  "
                    f"(Current: {curr['avg_loss']:.4f})  Per-model: {prop_losses}"
                )

            # 3. Propose next mutation from the current gradient
            _, prop_seq = mutate_reslogits(
                old_seq_str=curr["seq"],
                res_logits=-curr["grad"],
                encoding=encoding,
                random_mut=random_mut,
                allowed_mutate_mask=seq_of_interest_mask,
                max_mut=mutate_each,
                prot_dim=20,
                full_dim=32,
            )

            history.append(
                {
                    "step": step,
                    "name": f"{step}",
                    "loss_curr": curr["avg_loss"],
                    "loss_prop": prop_avg_loss,
                    "per_model_losses": prop_losses,
                    "accepted": accepted,
                    "ipae_mean": prop_info.get("ipae_mean", float("inf")),
                    "ipae_min": prop_info.get("ipae_min", float("inf")),
                    "geometric_distances": prop_info.get("geometric_distances", {}),
                    "substrate_plddt": prop_info.get("substrate_plddt", 0.0),
                    "components": prop_comps,
                }
            )

        history = [to_python_type(h) for h in history]
        accept_traj = [to_python_type(a) for a in accept_traj]
        return history, accept_traj
