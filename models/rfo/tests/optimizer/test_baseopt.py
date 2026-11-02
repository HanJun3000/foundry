import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.utils.io_utils import to_cif_file
from rfo.backprop.optimizer.folding_model.modelhub_grad import ModelhubGradient
from rfo.backprop.tools.af3 import cif2input
from rfo.backprop.tools.utils import ensure_probability
from torch.optim.lr_scheduler import ExponentialLR


class Optimizer(ModelhubGradient):
    def __init__(self, seqopt_config):
        super().__init__(seqopt_config)

    def sequence_gradient_opt(
        self,
        init_restype: torch.Tensor,
        network_input_template: dict,
        confidence_input: dict,
        nsteps: int = 100,
        lr: float = 1e-2,
        prob_norm_step=10,
        prob_norm_method="softmax",
        seq_of_interest_mask=None,
    ):
        """Observe the impact of gradient and normalization strategies on sequence optimization."""
        restype = torch.nn.Parameter(
            init_restype.clone().detach().float().to(self.device)
        )
        restype.requires_grad = True

        if seq_of_interest_mask is not None:
            seq_of_interest_mask = seq_of_interest_mask.to(self.device)

        optimizer = torch.optim.Adam([restype], lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        net_in_template = dict(network_input_template)
        history = []

        for step in range(nsteps):
            optimizer.zero_grad()

            net_in = dict(net_in_template)
            net_in["f"] = dict(net_in["f"])
            net_in["f"]["restype"] = restype
            loss = self.my_objective(net_in, confidence_input)
            loss.backward()

            if seq_of_interest_mask is not None:
                restype.grad *= seq_of_interest_mask.unsqueeze(-1)

            optimizer.step()
            scheduler.step()
            print(f"Step {step + 1}/{nsteps}: Loss = {loss.item():.4f}")
            print("Gradient w.r.t. restype:", restype.grad[1, :])
            print("Restype:", restype[1, :])
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "restype": restype.detach().cpu().numpy(),
                    "grad": restype.grad.detach().cpu().numpy(),
                }
            )
            if (prob_norm_step > 0) and (step % prob_norm_step == 0):
                with torch.no_grad():
                    restype.data = ensure_probability(
                        restype.data,
                        dim=-1,
                        method=prob_norm_method,
                        mask=seq_of_interest_mask,
                        temperature=0.1,
                    )
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AF3 using specified paths.")
    parser.add_argument(
        "input", nargs="+", default="/projects/ppi/dl_bad_pae_examples/v1/bcov_small"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/projects/ml/RF2_allatom/weights/af3_repro_with_confidence_20250124.pt",
    )
    parser.add_argument(
        "--n_recycles", type=int, default=1, help="Number of recycles for AF3"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--n_optimization_steps",
        type=int,
        default=100,
        help="Number of optimization steps",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "checkpoint_path": args.checkpoint_path,
            "n_recycles": args.n_recycles,
        }
    )

    input_files = []
    for path in args.input:
        path = Path(path)
        if path.is_dir():
            input_files.extend(path.glob("*.cif"))
            input_files.extend(path.glob("*.pdb"))
        elif path.suffix in {".cif", ".pdb", ".json"}:
            input_files.append(path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for input_file in input_files:
            if "pdb2cw2" not in str(input_file):
                continue
            else:
                print("input_file", input_file)
            if input_file.suffix == ".json":
                with open(input_file, "r") as json_file:
                    inputs = json.load(json_file)
                    atom_array, components = components_to_atom_array(
                        inputs, return_components=True
                    )
                    msa_paths_by_chain_id = (
                        build_msa_paths_by_chain_id_from_component_list(components)
                    )

                    input_cif = temp_dir / f"{input_file.stem}.cif"
                    save_path = to_cif_file(
                        atom_array,
                        input_cif,
                        extra_categories={
                            "msa_paths_by_chain_id": msa_paths_by_chain_id
                        }
                        if msa_paths_by_chain_id
                        else None,
                    )
                    input_cif = Path(save_path)
            else:
                input_cif = input_file

            print(f"Processing: {input_cif}")

            af3_seq_optimizer = Optimizer(seqopt_config=config)

            network_input, confidence_input, pipeline_output = cif2input(
                input_cif,
                pipeline=af3_seq_optimizer.pipeline,
                device=af3_seq_optimizer.device,
            )
            print(network_input.keys())

            f = network_input["f"]
            init_restype = f["restype"]

            interface_mask, binder_mask = af3_seq_optimizer.pipeline_output2mask(
                pipeline_output
            )
            seq_of_interest_mask = binder_mask & interface_mask
            seq_of_interest_mask = seq_of_interest_mask.to(torch.bool)

            history = af3_seq_optimizer.sequence_gradient_opt(
                init_restype=init_restype,
                network_input_template=network_input,
                confidence_input=confidence_input,
                nsteps=args.n_optimization_steps,
                lr=1e-1,
                prob_norm_step=10,
                prob_norm_method="sum",
                seq_of_interest_mask=seq_of_interest_mask,
            )

            output_file = os.path.join(
                args.output_path, "sum_every10_normalize_history.pkl"
            )
            import pickle

            with open(output_file, "wb") as f_out:
                pickle.dump(history, f_out)
