"""
Protein-Protein Interaction (PPI) optimizer for sequence optimization.

This optimizer supports:
- Both GradMCMC and GradProposal algorithms
- Random sequence initialization with variable length support
- Template fixing and MSA handling
- Fully config-driven execution
- Interface and binder optimization
- Easy loss function switching
"""

import argparse
import csv
import os
import os.path as osp
import pathlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch

from rfo.backprop.optimizer.folding_model.modelhub_grad import ModelhubGradient
from rfo.backprop.optimizer.gradient_mcmc import GradMCMC
from rfo.backprop.optimizer.gradient_proposal import GradProposal
from rfo.backprop.tools.af3 import (
    cif2input_with_template,
    display_mutation_mapping,
    input_cif_prep,
)
from rfo.backprop.tools.utils import index2mask, modelhub2af3, write_json


class PPIDesigner:
    """
    Designer for protein-protein interaction optimization.

    Features:
    - Interface residue detection
    - Binder chain identification
    - Built-in contact-based objective (inter + intra chain contacts)
    - MSA handling with CSV lookup support
    - Chain-separated sequence output
    - Random sequence initialization with variable length support
    - Template fixing support
    - Support for both GradMCMC and GradProposal algorithms
    - Easy loss function switching
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PPI optimizer from config."""
        self.config = config

        # MSA configuration
        msa_paths = config.get("msa_paths", None)
        if isinstance(msa_paths, str) and msa_paths.endswith(".csv"):
            csv_path = pathlib.Path(msa_paths)
            if not csv_path.exists():
                raise FileNotFoundError(f"{csv_path} not found")

            self._msa_lookup = {}
            with open(csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    pdb_stem = pathlib.Path(row[0]).stem
                    msa_file = row[3]
                    self._msa_lookup[pdb_stem] = msa_file

            self.msa_paths = None
            print(f"Loaded {len(self._msa_lookup)} MSA entries from {csv_path}")
        else:
            self._msa_lookup = None
            self.msa_paths = msa_paths

        # Template handling
        self.self_template = config.get("self_template", True)

        # Algorithm configuration
        self.algorithm_type = config.get("algorithm", "gradient_mcmc")
        self.algorithm_config = config.get("algorithm_config", {})

        # Random sequence configuration
        self.random_seq = config.get("random_seq", False)
        self.random_seq_length = config.get("random_seq_length", None)

        # Optimization configuration
        self.optimize_binder = config.get("optimize_binder", True)
        self.interface_only = config.get("interface_only", False)
        self.allowed_mut_pos = config.get("allowed_mut_pos", "")

        # Template configuration
        self.fix_template_dict = config.get("fix_template_dict", None)
        self.self_template_selection_syntax = config.get(
            "self_template_selection_syntax", None
        )

        # Loss configuration
        self.loss_type = config.get(
            "loss_type", "sum_contact_loss"
        )  # Default to contact loss

    @staticmethod
    def split_sequence_by_chain(seq_str, chain_ids):
        """
        Given a single-letter sequence (seq_str) and a 1D array/tensor of chain IDs (chain_ids),
        return a single string with each chain's subsequence separated by a space.
        """
        if isinstance(chain_ids, torch.Tensor):
            chain_ids = chain_ids.detach().cpu().numpy()
        unique_chain_ids = np.unique(chain_ids)

        chainwise_subseqs = []
        for c_id in unique_chain_ids:
            chain_residues = [
                seq_str[i] for i in range(len(seq_str)) if chain_ids[i] == c_id
            ]
            chainwise_subseqs.append("".join(chain_residues))

        return " ".join(chainwise_subseqs)

    def pipeline_output2mask(self, pipeline_output):
        """
        Special for binder optimization, we compute the interface mask and binder mask.
        """
        interface_mask = self.create_interface_residue_mask(
            pipeline_output["ground_truth"]["chain_iid_token_lvl"],
            pipeline_output["ground_truth"]["coord_atom_lvl"],
            pipeline_output["feats"]["atom_to_token_map"],
        ).to(self.device)
        binder_mask = self.create_binder_mask(
            pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        ).to(self.device)
        return interface_mask, binder_mask

    @staticmethod
    def create_interface_residue_mask(
        chain_iid_token_lvl, xyz, atom_to_token_map, cutoff=5.0
    ):
        """
        Identify interface residues based on inter-chain proximity.

        Args:
            chain_iid_token_lvl: Chain IDs per token
            xyz: Atom coordinates
            atom_to_token_map: Mapping from atoms to tokens
            cutoff: Distance cutoff for interface definition

        Returns:
            Boolean mask of interface residues
        """
        xyz_cpu = xyz.detach().cpu()
        if torch.is_tensor(atom_to_token_map):
            atom_to_token = atom_to_token_map.detach().cpu().numpy()
        else:
            atom_to_token = np.asarray(atom_to_token_map)
        if torch.is_tensor(chain_iid_token_lvl):
            chain_ids = chain_iid_token_lvl.detach().cpu().numpy()
        else:
            chain_ids = np.asarray(chain_iid_token_lvl)
        if chain_ids.dtype.kind in {"U", "S", "O"}:
            _, chain_ids = np.unique(chain_ids, return_inverse=True)

        dist = torch.cdist(xyz_cpu, xyz_cpu)
        chain_per_atom = torch.as_tensor(chain_ids[atom_to_token], dtype=torch.long)
        diff_chain = chain_per_atom[:, None] != chain_per_atom[None, :]
        interface_atom_pairs = (dist < cutoff) & diff_chain
        interface_atoms = interface_atom_pairs.any(dim=1)
        mask_cpu = torch.zeros(len(chain_ids), dtype=torch.bool)
        mask_cpu[atom_to_token[interface_atoms.cpu().numpy()]] = True

        return mask_cpu.to(xyz.device)

    @staticmethod
    def create_binder_mask(chain_iid_token_lvl):
        """
        Create mask for the binder chain (first chain).
        """
        chain_ids = np.asarray(chain_iid_token_lvl)
        first = chain_ids[0]
        return torch.as_tensor(chain_ids == first, dtype=torch.bool)

    def my_objective(
        self,
        folding_model,
        trunk_input,
        confidence_input,
        pipeline_output=None,
        inference_pipeline=None,
    ):
        """
        Objective function.

        Runs the forward pass through a single inference pipeline, computes
        the requested loss, and extracts additional info (structures, iPAE, etc.).

        Args:
            folding_model: The folding model instance.
            trunk_input: Network input dictionary.
            confidence_input: Confidence input dictionary.
            pipeline_output: Full pipeline output.
            inference_pipeline: Specific inference pipeline to use.  When
                ``None``, falls back to ``folding_model.inference_pipeline``
                (first / only model).
        """
        if inference_pipeline is None:
            inference_pipeline = folding_model.inference_pipeline

        confidence_based_losses = [
            "pae_interface_mean",
            "pae_interface_min",
            "pae_mean",
            "pde_mean",
            "plddt_mean",
            "iptm",
        ]
        skip_diffusion = self.loss_type not in confidence_based_losses

        model_output = inference_pipeline.forward(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            n_cycle=inference_pipeline.n_recycles,
            compute_gradient=True,
            skip_diffusion=skip_diffusion,
        )

        loss = inference_pipeline.compute_loss(
            self.loss_type, trunk_input, model_output, confidence_input, pipeline_output
        )

        additional_info = folding_model.extract_info_from_model_output(
            inference_pipeline,
            model_output,
            pipeline_output,
            expect_full_outputs=not skip_diffusion,
        )

        return loss, additional_info

    def get_objective_function(self):
        """Get the unified objective function that works for all algorithm types."""
        return self.my_objective

    def apply_random_sequence_initialization(self, init_restype, pipeline_output, f):
        """
        Apply random sequence initialization with support for variable length.
        """
        if not self.random_seq:
            return init_restype

        print(
            "🎲 Random sequence initialization enabled - hallucinating chain A from scratch"
        )

        # Get chain information
        chain_ids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        if isinstance(chain_ids, torch.Tensor):
            chain_ids = chain_ids.detach().cpu().numpy()

        # Find chain A positions (assume first unique chain is chain A)
        unique_chains = np.unique(chain_ids)
        chain_A_id = unique_chains[0]  # First chain
        chain_A_mask = chain_ids == chain_A_id
        chain_A_positions = np.where(chain_A_mask)[0]
        original_chain_A_length = len(chain_A_positions)

        # Determine target length for random sequence
        if self.random_seq_length is not None:
            target_length = self.random_seq_length
            print(
                f"🎯 Creating new chain A with fixed length: {target_length} (original: {original_chain_A_length})"
            )

            if target_length > original_chain_A_length:
                print(
                    f"⚠️  WARNING: Target length ({target_length}) > original ({original_chain_A_length})"
                )
                print(
                    "   This may cause issues since we're extending beyond the original structure"
                )
                print(
                    "   Consider using a target length ≤ original length for better results"
                )
                # Use original length as maximum to avoid structural issues
                target_length = min(target_length, original_chain_A_length)
                print(f"   Capping at original length: {target_length}")

            # Apply complex tensor resizing logic (from random_seq_18 version)
            init_restype = self._apply_variable_length_random_seq(
                init_restype,
                chain_ids,
                chain_A_positions,
                target_length,
                pipeline_output,
                f,
            )
        else:
            # Use all original chain A positions (original behavior)
            target_length = original_chain_A_length
            random_aa_indices = torch.randint(
                0, 20, (target_length,), device=init_restype.device
            )

            # Create one-hot encoded random sequence
            random_restype = init_restype.clone()
            random_restype[chain_A_positions, :] = 0.0  # Clear old chain A
            random_restype[chain_A_positions, random_aa_indices] = (
                1.0  # Set new random sequence
            )

            init_restype = random_restype

        # Decode and print the new chain A sequence
        self._print_random_sequence_info(
            init_restype, pipeline_output, f, target_length
        )

        return init_restype

    def _apply_variable_length_random_seq(
        self,
        init_restype,
        chain_ids,
        chain_A_positions,
        target_length,
        pipeline_output,
        f,
    ):
        """
        Apply variable length random sequence initialization (complex tensor resizing).
        This is the advanced logic from the random_seq_18 version.
        """
        original_chain_A_length = len(chain_A_positions)
        chain_A_id = chain_ids[chain_A_positions[0]]  # Get chain A ID

        # CRITICAL: We need to resize the entire sequence tensor to accommodate the new chain A length
        # Calculate new total length
        other_chains_length = len(init_restype) - original_chain_A_length
        new_total_length = target_length + other_chains_length

        print(
            f"🔧 Resizing structure: {len(init_restype)} → {new_total_length} residues"
        )

        # Create new tensors with the correct size
        new_restype = torch.zeros(
            (new_total_length, init_restype.shape[1]),
            device=init_restype.device,
            dtype=init_restype.dtype,
        )

        # Generate random sequence for new chain A (one-hot encoding)
        random_aa_indices = torch.randint(
            0, 20, (target_length,), device=init_restype.device
        )
        print(
            f"🎲 Generated random amino acid indices: {random_aa_indices.cpu().numpy()}"
        )
        for i, aa_idx in enumerate(random_aa_indices):
            new_restype[i, aa_idx] = 1.0

        # Copy other chains after the new chain A
        other_chains_mask = ~(chain_ids == chain_A_id)
        other_chains_positions = np.where(other_chains_mask)[0]
        if len(other_chains_positions) > 0:
            # Copy other chains' sequences
            other_chains_start_idx = target_length
            for i, orig_pos in enumerate(other_chains_positions):
                new_restype[other_chains_start_idx + i] = init_restype[orig_pos]

        # Update all related tensors to match new length
        self._resize_feature_tensors(
            f,
            init_restype,
            new_total_length,
            target_length,
            chain_A_positions,
            other_chains_positions,
            new_restype,
        )

        # Update pipeline_output chain information
        self._update_chain_information(
            pipeline_output,
            chain_ids,
            chain_A_id,
            target_length,
            other_chains_positions,
            new_total_length,
        )

        return new_restype

    def _resize_feature_tensors(
        self,
        f,
        init_restype,
        new_total_length,
        target_length,
        chain_A_positions,
        other_chains_positions,
        new_restype,
    ):
        """Resize all feature tensors to match the new sequence length."""
        for key in f.keys():
            if isinstance(f[key], torch.Tensor) and len(f[key]) == len(init_restype):
                if key == "restype":
                    # Use our carefully constructed random sequence
                    f[key] = new_restype
                    print("🎲 Updated f['restype'] with random sequence")
                else:
                    # For other per-residue features, we need to resize accordingly
                    new_tensor = torch.zeros(
                        (new_total_length,) + f[key].shape[1:],
                        device=f[key].device,
                        dtype=f[key].dtype,
                    )

                    # Handle chain A: for most features, we can just use the first target_length entries
                    if key in ["is_protein", "atom_mask", "coord"]:
                        # Copy first target_length entries for chain A
                        if len(chain_A_positions) >= target_length:
                            new_tensor[:target_length] = f[key][
                                chain_A_positions[:target_length]
                            ]
                        else:
                            # If target_length > original, pad with the last entry
                            new_tensor[: len(chain_A_positions)] = f[key][
                                chain_A_positions
                            ]
                            if target_length > len(chain_A_positions):
                                new_tensor[len(chain_A_positions) : target_length] = f[
                                    key
                                ][chain_A_positions[-1:]]
                    else:
                        # For other features, just copy the pattern
                        if len(chain_A_positions) >= target_length:
                            new_tensor[:target_length] = f[key][
                                chain_A_positions[:target_length]
                            ]
                        else:
                            new_tensor[: len(chain_A_positions)] = f[key][
                                chain_A_positions
                            ]

                    # Copy other chains
                    if len(other_chains_positions) > 0:
                        other_chains_start_idx = target_length
                        for i, orig_pos in enumerate(other_chains_positions):
                            new_tensor[other_chains_start_idx + i] = f[key][orig_pos]

                    f[key] = new_tensor

    def _update_chain_information(
        self,
        pipeline_output,
        chain_ids,
        chain_A_id,
        target_length,
        other_chains_positions,
        new_total_length,
    ):
        """Update chain IDs and atom mappings for the new structure."""
        # Create new chain IDs array
        new_chain_ids = np.empty(new_total_length, dtype=chain_ids.dtype)
        new_chain_ids[:target_length] = chain_A_id  # New chain A

        # Copy other chain IDs
        if len(other_chains_positions) > 0:
            other_chains_start_idx = target_length
            for i, orig_pos in enumerate(other_chains_positions):
                new_chain_ids[other_chains_start_idx + i] = chain_ids[orig_pos]

        # Update pipeline_output chain IDs
        original_chain_tensor = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        if isinstance(original_chain_tensor, torch.Tensor):
            if chain_ids.dtype.kind in {"U", "S", "O"}:  # String types
                unique_chains = np.unique(new_chain_ids)
                chain_to_int = {chain: i for i, chain in enumerate(unique_chains)}
                new_chain_ids_int = np.array(
                    [chain_to_int[x] for x in new_chain_ids], dtype=np.int64
                )
                pipeline_output["ground_truth"]["chain_iid_token_lvl"] = (
                    torch.from_numpy(new_chain_ids_int).to(original_chain_tensor.device)
                )
            else:
                pipeline_output["ground_truth"]["chain_iid_token_lvl"] = (
                    torch.from_numpy(new_chain_ids).to(original_chain_tensor.device)
                )
        else:
            pipeline_output["ground_truth"]["chain_iid_token_lvl"] = new_chain_ids

    def _print_random_sequence_info(
        self, init_restype, pipeline_output, f, target_length
    ):
        """Print information about the generated random sequence."""
        if self.random_seq_length is not None:
            # For fixed length, chain A is now at positions 0:target_length
            chain_A_restype = init_restype[:target_length]
            chain_A_protein_mask = f["is_protein"][:target_length]

            print(f"🔍 chain_A_restype shape: {chain_A_restype.shape}")
            print(
                f"🔍 chain_A_restype argmax: {torch.argmax(chain_A_restype, dim=1).cpu().numpy()}"
            )
            print(f"🔍 chain_A_protein_mask: {chain_A_protein_mask.cpu().numpy()}")
        else:
            # For original length, use the updated chain A positions
            updated_chain_ids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
            if isinstance(updated_chain_ids, torch.Tensor):
                updated_chain_ids = updated_chain_ids.detach().cpu().numpy()

            unique_chains = np.unique(updated_chain_ids)
            chain_A_id = unique_chains[0]
            updated_chain_A_mask = updated_chain_ids == chain_A_id
            chain_A_mask_tensor = torch.from_numpy(updated_chain_A_mask).to(
                f["is_protein"].device
            )
            chain_A_protein_mask = chain_A_mask_tensor & f["is_protein"]

            if chain_A_protein_mask.any():
                chain_A_restype = init_restype[chain_A_protein_mask]
            else:
                print(f"🎲 Generated random sequence for {target_length} positions")
                return

        print(f"🎲 New chain A sequence (length {target_length})")

    def process_ppi_design(
        self,
        folding_model,
        optimizer,
        input_file: Path,
        output_path: str = "./output",
        output_suffix: Optional[str] = None,
    ):
        """
        Process a single PPI structure with all special handling.

        Args:
            folding_model: The folding model instance
            optimizer: The optimization algorithm instance
            input_file: Path to input PDB/CIF file
            output_path: Base output directory
            output_suffix: Suffix for output directory

        Returns:
            Tuple of (history, results)
        """
        self.device = folding_model.device
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_cif = input_cif_prep(input_file, temp_dir)
            print(f"Processing: {input_file}")

            trunk_input, confidence_input, pipeline_output = cif2input_with_template(
                input_cif,
                pipeline=folding_model.pipeline,
                device=folding_model.device,
                self_template_selection_syntax=self.self_template_selection_syntax,
                fix_template_dict=self.fix_template_dict,
                msa_paths_by_chain_id=self.msa_paths,
            )
            print()
            f = trunk_input["f"]

            init_restype = f["restype"]

            # Apply random sequence initialization if configured
            init_restype = self.apply_random_sequence_initialization(
                init_restype, pipeline_output, f
            )

            # Prepare CIF additional info
            cif_add_info = {
                "chain_id": None,
                "smiles": None,
                "msa_paths": self.msa_paths,
                "chain_iid_token_level": pipeline_output["ground_truth"][
                    "chain_iid_token_lvl"
                ],
                "self_template": self.self_template_selection_syntax
                if self.self_template_selection_syntax
                else None,
                "fix_template_dict": self.fix_template_dict
                if self.fix_template_dict
                else None,
            }

            # Create optimization masks
            interface_mask, binder_mask = self.pipeline_output2mask(pipeline_output)
            seq_of_interest_mask = f["is_protein"]
            if self.optimize_binder:
                seq_of_interest_mask = seq_of_interest_mask & binder_mask
                if self.interface_only:
                    seq_of_interest_mask = seq_of_interest_mask & interface_mask

            # Handle fixed random sequence length restrictions
            if self.random_seq and self.random_seq_length is not None:
                # After resizing, chain A is now at positions 0:random_seq_length
                fixed_length_mask = torch.zeros_like(seq_of_interest_mask)
                actual_target_length = min(
                    self.random_seq_length, len(seq_of_interest_mask)
                )
                fixed_length_mask[:actual_target_length] = (
                    True  # Chain A is now at the beginning
                )

                # Combine with protein mask and other constraints
                seq_of_interest_mask = seq_of_interest_mask & fixed_length_mask

                print(
                    f"🎯 Restricting optimization to {actual_target_length} new chain A positions only"
                )

            # Handle allowed mutation positions
            if self.allowed_mut_pos != "":
                seq_of_interest_index = list(self.allowed_mut_pos.split(","))
                seq_of_interest_index = [int(i) for i in seq_of_interest_index]

                # Validate positions for variable length sequences
                if self.random_seq and self.random_seq_length is not None:
                    actual_target_length = min(
                        self.random_seq_length, len(init_restype)
                    )
                    invalid_positions = [
                        pos for pos in seq_of_interest_index if pos >= len(init_restype)
                    ]
                    if invalid_positions:
                        print(
                            f"⚠️  WARNING: Some allowed_mut_pos are beyond new structure length ({len(init_restype)}): {invalid_positions}"
                        )
                        print("   Filtering to valid positions only")
                        seq_of_interest_index = [
                            pos
                            for pos in seq_of_interest_index
                            if pos < len(init_restype)
                        ]

                    outside_chain_a = [
                        pos
                        for pos in seq_of_interest_index
                        if pos >= actual_target_length
                    ]
                    if outside_chain_a:
                        print(
                            f"⚠️  WARNING: Some allowed_mut_pos are outside new chain A (length {actual_target_length}): {outside_chain_a}"
                        )
                        print(
                            "   These positions are in other chains and may not be what you intended"
                        )

                seq_of_interest_mask = index2mask(
                    seq_of_interest_index, length=len(init_restype)
                )

            if seq_of_interest_mask is not None:
                seq_of_interest_index = (
                    torch.where(seq_of_interest_mask)[0].cpu().numpy()
                )
                display_mutation_mapping(seq_of_interest_index, pipeline_output)

            # Setup output directory
            if output_suffix is None:
                output_suffix = ""
                if self.msa_paths is not None:
                    output_suffix += "_with_msa"
                if self.fix_template_dict is not None:
                    output_suffix += "_with_template"
                if self.random_seq:
                    output_suffix += "_random_seq"
                    if self.random_seq_length is not None:
                        output_suffix += f"_{self.random_seq_length}"
                if self.algorithm_type == "gradient_proposal":
                    output_suffix += "_gradproposal"

            output_path = osp.join(output_path, input_file.stem + output_suffix)
            os.makedirs(output_path, exist_ok=True)

            init_save_path = osp.join(output_path, "input.cif")
            shutil.copy(input_file, init_save_path)

            # Get the appropriate objective function
            objective_function = self.get_objective_function()

            # Run optimization with algorithm-specific parameters
            if self.algorithm_type == "gradient_mcmc":
                history, accept_traj = optimizer.optimize(
                    folding_model=folding_model,
                    my_objective=objective_function,
                    opt_steps=optimizer.algorithm_config.opt_steps,
                    init_restype=init_restype,
                    trunk_input=trunk_input,
                    confidence_input=confidence_input,
                    pipeline_output=pipeline_output,
                    seq_of_interest_mask=seq_of_interest_mask,
                    cif_add_info=cif_add_info,
                    output_path=output_path,
                )
            elif self.algorithm_type == "gradient_proposal":
                history, accept_traj = optimizer.optimize(
                    folding_model=folding_model,
                    my_objective=objective_function,
                    init_restype=init_restype,
                    trunk_input=trunk_input,
                    opt_steps=optimizer.algorithm_config.opt_steps,
                    seq_of_interest_mask=seq_of_interest_mask,
                    cif_add_info=cif_add_info,
                    output_path=output_path,
                )
            else:
                raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")

            # Save results
            write_json(accept_traj, osp.join(output_path, "accept_traj.json"))

            # Create modelhub and AF3 input files from trajectory
            modelhub_inputs = []
            for i, data in enumerate(history):
                modelhub_input = {
                    "name": data["name"],
                    "components": data["components"],
                }
                modelhub_inputs.append(modelhub_input)

            print(
                f"Saving {len(modelhub_inputs)} modelhub inputs to {osp.join(output_path, 'modelhub_inputs.json')}"
            )
            write_json(modelhub_inputs, osp.join(output_path, "modelhub_inputs.json"))
            print(
                f"Saving {len(history)} history entries to {osp.join(output_path, 'history.json')}"
            )

            af3_input = osp.join(output_path, "af3_inputs.json")
            modelhub2af3(modelhub_inputs, af3_input)
            print(f"AF3 input saved to {af3_input}")
            write_json(history, osp.join(output_path, "history.json"))
            print(f"Results saved in: {output_path}")

            return history, accept_traj


def main():
    """Main function for config-driven execution."""
    parser = argparse.ArgumentParser(description="Run PPI sequence optimization")
    parser.add_argument(
        "--config_name", type=str, default="cycle_ppi", help="Config file name to use"
    )
    parser.add_argument(
        "--config_path", type=str, default="configs", help="Path to config directory"
    )
    args = parser.parse_args()

    # Load configuration
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(Path(args.config_path).resolve()),
        job_name="ppi_optimization",
    ):
        config = hydra.compose(config_name=args.config_name)

    # Initialize components from config
    folding_model = ModelhubGradient(seqopt_config=config)

    # Create optimizer based on algorithm type
    algorithm_type = config.get("algorithm", "gradient_mcmc")
    algorithm_config = config.get("algorithm_config", {})

    if algorithm_type == "gradient_mcmc":
        optimizer = GradMCMC(algorithm_config=algorithm_config)
    elif algorithm_type == "gradient_proposal":
        optimizer = GradProposal(algorithm_config=algorithm_config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    # Initialize PPI designer
    ppi_designer = PPIDesigner(config)

    # Get input files from config
    input_path = Path(config.input)
    output_path = config.output_path

    input_files = []
    if input_path.is_dir():
        input_files.extend(input_path.glob("*.cif"))
        input_files.extend(input_path.glob("*.pdb"))
        input_files.extend(input_path.glob("*.json"))
    elif input_path.suffix in {".cif", ".pdb", ".json"}:
        input_files.append(input_path)

    print(f"Found {len(input_files)} input files in {input_path}")

    if not input_files:
        raise ValueError("No input files found.")

    # Process each input file
    for input_file in input_files:
        print(f"Processing input file: {input_file}")
        # Process the PPI design
        history, results = ppi_designer.process_ppi_design(
            folding_model=folding_model,
            optimizer=optimizer,
            input_file=input_file,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
