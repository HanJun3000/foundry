import logging
import tempfile

import torch
import tree
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from rf3.metrics.metric_utils import create_interface_masks_2d
from rf3.utils.io import build_stack_from_atom_array_and_batched_coords
from torch.optim.lr_scheduler import ExponentialLR

from rfo.backprop.inference_pipeline.af3_inference_pipeline import AF3InferencePipeline
from rfo.backprop.tools.af3 import cif2input_with_template, decode_restype, seq2cif
from rfo.backprop.tools.loss import unbin_logits
from rfo.backprop.tools.mutate import compare_seq_mutation, mutate_reslogits
from rfo.backprop.tools.utils import norm_seq_grad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelhubGradient:
    """
    AF3 gradient-based sequence optimizer using inference pipeline for all model interactions.
    Supports single or multiple checkpoints for consensus-based optimization.
    """

    def __init__(self, seqopt_config=None):
        """Initialize the optimizer with one or more inference pipelines.

        Accepts either ``checkpoint_path`` (str) or ``checkpoint_paths`` (list of str)
        in the config.  When multiple paths are provided every pipeline is stored in
        ``self.inference_pipelines`` and the first one is aliased as
        ``self.inference_pipeline`` for backward compatibility.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seqopt_config = seqopt_config

        # Resolve checkpoint path(s) -- accepts any of:
        #   checkpoint_paths: [path1, path2]   (preferred for multi-checkpoint)
        #   checkpoint_path:  [path1, path2]   (also works)
        #   checkpoint_path:  single_path      (original single-checkpoint)
        from omegaconf import ListConfig

        ckpt_paths = getattr(seqopt_config, "checkpoint_paths", None)
        if ckpt_paths is None:
            ckpt_paths = seqopt_config.checkpoint_path
        if isinstance(ckpt_paths, (list, ListConfig)):
            ckpt_paths = [str(p) for p in ckpt_paths]
        else:
            ckpt_paths = [str(ckpt_paths)]

        common_kwargs = dict(
            n_recycles=getattr(seqopt_config, "n_recycles", 10),
            diffusion_batch_size=getattr(seqopt_config, "diffusion_batch_size", 1),
            num_steps=getattr(seqopt_config, "num_steps", 50),
            device=str(self.device),
            load_ema_weights=False,
        )

        self.inference_pipelines: list[AF3InferencePipeline] = []
        for ckpt_path in ckpt_paths:
            logger.info(f"Loading checkpoint: {ckpt_path}")
            self.inference_pipelines.append(
                AF3InferencePipeline(ckpt_path=ckpt_path, **common_kwargs)
            )

        # Backward-compatible alias (first pipeline)
        self.inference_pipeline = self.inference_pipelines[0]
        self.pipeline = self.inference_pipeline.pipeline
        self.encoding = AF3SequenceEncoding()
        self.n_models = len(self.inference_pipelines)

        # Configuration options
        self.self_template = getattr(seqopt_config, "self_template", False)
        self.msa_paths = getattr(seqopt_config, "msa_paths", None)
        self.cif_add_info = None

        logger.info(f"Optimizer initialized with {self.n_models} model(s)")

    def project_with_cif_reconstruction(
        self, seq, cif_add_info, protein_mask, out_file=None
    ):
        """
        Takes a protein sequence as input, writes it to a CIF file, and reads back
        via pipeline transformation for consistency in model features.
        """
        with tempfile.NamedTemporaryFile(delete=True, suffix=".cif.gz") as tmp_file:
            if out_file is None:
                out_file = tmp_file.name
            cif_path, inputs = seq2cif(
                seq, add_info=cif_add_info, protein_mask=protein_mask, out_file=out_file
            )
            new_net_input, new_confidence_feats, new_pipeline_output = (
                cif2input_with_template(
                    cif_path,
                    self.pipeline,
                    device=self.device,
                    self_template_selection_syntax=cif_add_info["self_template"],
                    fix_template_dict=cif_add_info["fix_template_dict"],
                    msa_paths_by_chain_id=cif_add_info["msa_paths"],
                )
            )

        new_net_input = tree.map_structure(
            lambda x: x.to(self.device) if hasattr(x, "to") else x, new_net_input
        )
        new_confidence_feats = tree.map_structure(
            lambda x: x.to(self.device) if hasattr(x, "to") else x, new_confidence_feats
        )
        new_pipeline_output = tree.map_structure(
            lambda x: x.to(self.device) if hasattr(x, "to") else x, new_pipeline_output
        )

        return new_net_input, new_confidence_feats, new_pipeline_output, inputs

    def my_objective(self, trunk_input, confidence_input, pipeline_output=None):
        """
        Compute objective using inference pipeline.

        Default: distogram CCE loss. Override in subclasses for different objectives.

        Available objective types:
        - 'distogram_cce': Distogram cross-entropy loss (default)
        - 'inter_contact': Inter-chain contact loss
        - 'intra_contact': Intra-chain contact loss
        - 'sum_contact_loss': Sum of inter and intra contact losses
        - 'interface_entropy': Interface entropy loss
        - 'pae_interface_mean': Mean PAE at interfaces (confidence-based)
        - 'weighted_interface_entropy': Interface entropy weighted by proximity
        - 'pae_interface_min': Min PAE at interfaces (confidence-based)
        - 'pae_mean': Mean PAE (confidence-based)
        - 'pde_mean': Mean PDE (confidence-based)
        - 'plddt_mean': Mean pLDDT (confidence-based, negative for maximization)
        """
        # Get objective type from config, default to distogram_cce
        objective_type = getattr(
            self.seqopt_config, "objective_type", "pae_interface_mean"
        )

        return self.inference_pipeline.optimization_objective(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            objective_type=objective_type,
            n_cycle=10,
        )

    @staticmethod
    def extract_info_from_model_output(
        inference_pipeline, model_output, pipeline_output, expect_full_outputs=True
    ):
        """
        Extract relevant information from the pipeline output.

        Args:
            inference_pipeline: The inference pipeline instance
            model_output: Output from the model forward pass
            pipeline_output: Pipeline output dictionary
            expect_full_outputs: Whether to expect full outputs (False when skip_diffusion=True)

        Returns:
            Dictionary with extracted information.
        """
        results = {}

        # Handle trunk outputs (available unless skip_diffusion=True)
        if model_output.get("S_I") is not None:
            results["S_I"] = model_output.get("S_I").detach()
        elif expect_full_outputs:
            results["S_I"] = None

        if model_output.get("Z_II") is not None:
            results["Z_II"] = model_output.get("Z_II").detach()
        elif expect_full_outputs:
            results["Z_II"] = None

        # Handle structure outputs (None when skip_diffusion=True)
        structure_coords = model_output.get("X_pred_rollout_L")
        if structure_coords is None:
            structure_coords = model_output.get("X_L")
        if structure_coords is not None:
            results["X_L"] = structure_coords.detach()
            # Build atom array stack only if we have coordinates
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                results["X_L"], pipeline_output["atom_array"]
            )
            results["atom_array_stack"] = atom_array_stack
        elif expect_full_outputs:
            results["X_L"] = None
            results["atom_array_stack"] = None

        # Handle confidence outputs (None when skip_diffusion=True or no confidence head)
        pae_logits = model_output.get("pae")
        if pae_logits is not None and inference_pipeline.confidence_config is not None:
            pae = unbin_logits(
                pae_logits.permute(0, 3, 1, 2).to(torch.float),
                inference_pipeline.confidence_config.pae.max_value,
                inference_pipeline.confidence_config.pae.n_bins,
            )
            ch_label = pipeline_output.get("ground_truth", {}).get(
                "chain_iid_token_lvl"
            )

            pairs_to_score = create_interface_masks_2d(ch_label, device=pae.device)

            if len(pairs_to_score) > 0:
                interface_mask = next(iter(pairs_to_score.values()))
                interface_pae_values = pae[interface_mask.unsqueeze(0).expand_as(pae)]
                if len(interface_pae_values) > 0:
                    results["ipae_mean"] = interface_pae_values.mean().item()
                    results["ipae_min"] = interface_pae_values.min().item()
                else:
                    results["ipae_mean"] = float("inf")
                    results["ipae_min"] = float("inf")
            else:
                results["ipae_mean"] = float("inf")
                results["ipae_min"] = float("inf")
        elif expect_full_outputs:
            # No PAE available (e.g., when skip_diffusion=True or no confidence)
            results["ipae_mean"] = float("inf")
            results["ipae_min"] = float("inf")

        return results

    def inference_without_gradient(
        self, trunk_input, confidence_input, pipeline_output=None
    ):
        trunk_output = self.inference_pipeline.forward(
            trunk_input=trunk_input,
            confidence_input=confidence_input,
            pipeline_output=pipeline_output,
            n_cycle=10,
            compute_gradient=False,
            skip_diffusion=False,
        )

        additional_info = self.extract_info_from_model_output(
            self.inference_pipeline,
            trunk_output,
            pipeline_output,
            expect_full_outputs=True,
        )

        return additional_info

    def save_structure(self, atom_array_stack, output_path, one_model_per_file=False):
        """
        Save atom array stack as CIF file.

        Args:
            atom_array_stack: AtomArrayStack to save
            output_path: Path for output file (without extension)
            one_model_per_file: If True, save each model separately
        """
        from rf3.utils.io import dump_structures

        dump_structures(
            atom_arrays=atom_array_stack,
            base_path=output_path,
            one_model_per_file=one_model_per_file,
            file_type="cif",
        )

    def grad_descent(self, verbose=True):
        """Perform gradient descent on the restype tensor."""
        self.optimizer.zero_grad()
        self.trunk_input["f"]["restype"] = self.restype

        # get loss with gradients from inference pipeline
        loss = self.my_objective(
            self.trunk_input, self.confidence_input, self.pipeline_output
        )
        loss.backward()

        # normalize the gradient
        with torch.no_grad():
            new_grad = norm_seq_grad(self.restype.grad)
            new_grad = new_grad * self.seq_of_interest_mask.unsqueeze(-1).float()
            self.restype.grad.copy_(new_grad)

        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        self.history.append(
            {
                "step": self.step,
                "loss": float(loss.cpu().detach().numpy()),
                "sequence": decode_restype(
                    self.restype.cpu().detach(), self.encoding, self.protein_mask
                ),
            }
        )

        if verbose:
            print(f"[Step {self.step:3d}] loss={loss.item():.4f}")

    def seq2loss(self, seq, out_file=None, protein_mask=None, cif_add_info=None):
        """Evaluate sequence loss using inference pipeline."""
        # Use provided values or fall back to instance attributes
        protein_mask = (
            protein_mask
            if protein_mask is not None
            else getattr(self, "protein_mask", None)
        )
        cif_add_info = (
            cif_add_info
            if cif_add_info is not None
            else getattr(self, "cif_add_info", None)
        )

        if protein_mask is None:
            raise ValueError(
                "protein_mask must be provided or set as an instance attribute"
            )

        with torch.no_grad():
            new_trunk_input, new_confidence_input, new_pipeline_output, _ = (
                self.project_with_cif_reconstruction(
                    seq=seq,
                    cif_add_info=cif_add_info,
                    protein_mask=protein_mask,
                    out_file=out_file,
                )
            )

            loss = self.my_objective(
                new_trunk_input, new_confidence_input, new_pipeline_output
            )

        return loss

    def sequence_gradient_opt(
        self,
        init_restype: torch.Tensor,
        trunk_input: dict,
        confidence_input: dict,
        pipeline_output: dict,
        mutate_sample_step: int = 1,
        opt_steps: int = 100,
        lr: float = 1e-2,
        round_trip: bool = True,
        seq_of_interest_mask=None,
        random_mut: bool = False,
        max_mut=None,
        pos_mut_prob=None,
        cif_add_info=None,
    ):
        """Perform gradient-based sequence optimization."""
        # initialize trainable restype parameter
        self.init_restype = init_restype
        self.restype = torch.nn.Parameter(
            init_restype.clone().detach().float().to(self.device)
        )
        self.restype.requires_grad = True

        self.optimizer = torch.optim.Adam([self.restype], lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)
        self.trunk_input = trunk_input
        self.confidence_input = confidence_input
        self.pipeline_output = pipeline_output
        self.protein_mask = trunk_input["f"]["is_protein"]
        self.cif_add_info = cif_add_info
        self.history = []

        results = {"evolve_traj": []}

        # Initialize sequence tracking
        self.initial_seq_str = decode_restype(
            init_restype, self.encoding, self.protein_mask
        )
        previous_seq_str = self.initial_seq_str
        print("Initial sequence:", previous_seq_str)

        self.seq_of_interest_mask = seq_of_interest_mask
        if seq_of_interest_mask is not None:
            seq_of_interest_index = torch.where(seq_of_interest_mask)[0].cpu().numpy()
            print("################################################################")
            print(f"Positions allowed to mutate: {seq_of_interest_index}.")
            print("################################################################")

        self.step = 0
        for step in range(opt_steps):
            self.grad_descent()

            # apply mutations periodically
            if (
                (mutate_sample_step > 0)
                and (step % mutate_sample_step == 0)
                and (step > 0)
            ):
                mutated_restype, mutated_seq = mutate_reslogits(
                    old_seq_str=previous_seq_str,
                    res_logits=self.restype,
                    encoding=self.encoding,
                    random_mut=random_mut,
                    allowed_mutate_mask=seq_of_interest_mask,
                    max_mut=max_mut,
                    pos_mut_prob=pos_mut_prob,
                    prot_dim=20,
                    full_dim=32,
                    force_mutate=False,
                )

                mutation_positions = compare_seq_mutation(
                    previous_seq_str, mutated_seq, print_num=5
                )

                # update restype
                with torch.no_grad():
                    self.restype.copy_(mutated_restype)

                if round_trip and (len(mutation_positions) > 0):
                    new_trunk_input, new_confidence_input, new_pipeline_output, _ = (
                        self.project_with_cif_reconstruction(
                            seq=mutated_seq,
                            cif_add_info=cif_add_info,
                            protein_mask=self.protein_mask,
                        )
                    )
                    self.trunk_input = new_trunk_input
                    self.confidence_input = new_confidence_input
                    self.pipeline_output = new_pipeline_output

                    print("Performing round trip transformation...")

                    with torch.no_grad():
                        self.restype.copy_(new_trunk_input["f"]["restype"])
                        real_loss = self.my_objective(
                            new_trunk_input, new_confidence_input, self.pipeline_output
                        )

                        results["evolve_traj"].append(
                            {
                                "step": step,
                                "loss": real_loss.item(),
                                "sequence": mutated_seq,
                            }
                        )

                    print(f"Real Loss: [Step {step:3d}] loss={real_loss.item():.4f}")

                previous_seq_str = mutated_seq

        print("\nOptimization complete!")
        return self.history, results
