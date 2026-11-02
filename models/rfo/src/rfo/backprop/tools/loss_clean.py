import hydra
import numpy as np
import torch
from rf3.metrics.metric_utils import find_bin_midpoints
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, verbose=False, **losses):
        super().__init__()
        self.to_compute = []
        for loss_name, loss in losses.items():
            loss_fn = hydra.utils.instantiate(loss)
            loss_fn.set_name(loss_name)
            if verbose:
                print(f"Adding loss {loss_name} to the loss function")
            self.to_compute.append(loss_fn)

    def forward(
        self,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        loss_dict = {}
        loss = 0
        for loss_fn in self.to_compute:
            loss_, loss_dict_ = loss_fn(
                loss_dict,
                trunk_input,
                trunk_output,
                confidence_input,
                confidence_data,
                pipeline_output,
            )
            loss += loss_
            loss_dict.update(loss_dict_)
        loss_dict["total_loss"] = loss.detach()
        return loss, loss_dict


class Loss_ABC(nn.Module):
    """Abstract class for loss function"""

    def __init__(self):
        super().__init__()

    def set_name(self, name):
        self._name = name

    def loss_calc(self, target, min, max):
        """
        To calculate the loss given min and max threshold.
            Args:
                target : torch.tensor, 1D tensor [I]
                min : float, `target` smaller than `min` will have penalty.
                max : float, `target` higher than `max` will have penalty.
            Returns:
                torch.tensor, 1D tensor with single numerical value
        """
        loss_min_mask = torch.where(
            target < min, torch.ones_like(target), torch.zeros_like(target)
        )
        loss_max_mask = torch.where(
            target > max, torch.ones_like(target), torch.zeros_like(target)
        )
        # print("target, min, max: ", target, min, max)
        # print(f"loss_min_mask : {loss_min_mask}")
        # print(f"loss_max_mask : {loss_max_mask}")
        # print(f"loss_min_mask+loss_max_mask : {loss_min_mask+loss_max_mask}")
        return torch.min((target - self.min) ** 2, (target - self.max) ** 2) * (
            loss_min_mask + loss_max_mask
        )

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        raise NotImplementedError


class Distance(Loss_ABC):
    """Distogram based loss"""

    def __init__(
        self, _name=None, atom_1=None, atom_2=None, min=None, max=None, weight=1
    ):
        super().__init__()
        self._name = _name
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.min = float(min)
        self.max = float(max)
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        pred_distogram = trunk_output["distogram_unbinned"]

        # to pinpoint the token
        entityID, resn, resi, atomName = self.atom_1.split(":")
        entity_mask = trunk_input["f"]["entity_id"] == int(entityID)
        resi_mask = trunk_input["f"]["residue_index"] == int(resi) - 1
        atom_1_tok_loc = torch.where(torch.multiply(entity_mask, resi_mask))

        # to pinpoint the token
        entityID, resn, resi, atomName = self.atom_2.split(":")
        entity_mask = trunk_input["f"]["entity_id"] == int(entityID)
        resi_mask = trunk_input["f"]["residue_index"] == int(resi) - 1
        atom_2_tok_loc = torch.where(torch.multiply(entity_mask, resi_mask))

        # to calculate loss if (plddt_chain < min) or (plddt_chain > max)
        target_distoram = torch.flatten(pred_distogram[atom_1_tok_loc, atom_2_tok_loc])
        loss = self.loss_calc(target_distoram, self.min, self.max)
        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss,
                    "weight": self.weight,
                }
            }
        )
        return loss * self.weight, loss_dict


class pLDDT_chain(Loss_ABC):
    def __init__(self, _name=None, chainID=None, min=None, max=None, weight=1):
        super().__init__()
        self._name = _name
        self.chainID = chainID
        self.min = float(min)
        self.max = float(max)
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        plddt_chain = confidence_data["chain_wise_mean_plddt"][self.chainID]
        loss = self.loss_calc(plddt_chain, self.min, self.max)
        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss,
                    "weight": self.weight,
                }
            }
        )
        return loss * self.weight, loss_dict


class pLDDT_motif(Loss_ABC):
    def __init__(self, _name=None, atom=None, min=None, max=None, weight=1):
        super().__init__()
        self.atom = atom
        self.min = float(min)
        self.max = float(max)
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        entityID, resn, resi, atomName = self.atom.split(":")
        entity_mask = trunk_input["f"]["entity_id"] == int(entityID)
        resi_mask = trunk_input["f"]["residue_index"] == int(resi) - 1
        tok_loc = torch.where(torch.multiply(entity_mask, resi_mask))

        target_pLDDT = torch.flatten(
            confidence_data["plddt_per_token"][:, tok_loc]
        )  # [B, I] -> [I]
        loss = self.loss_calc(target_pLDDT, self.min, self.max)
        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss,
                    "weight": self.weight,
                }
            }
        )
        return loss * self.weight, loss_dict


class DistogramCCELoss(Loss_ABC):
    def __init__(self, _name=None, atom=None, min=None, max=None, weight=1):
        super().__init__()
        self.atom = atom
        self.min = float(min)
        self.max = float(max)
        self.weight = float(weight)

    @staticmethod
    def distogram_cce_loss(
        pred_distogram,
        X_rep_atoms_I,
        crd_mask_rep_atoms_I,
        min_distance=2,
        max_distance=22,
        bins=64,
    ):
        """
        computes distogram cross-entropy loss
        """
        device = pred_distogram.device
        X_rep_atoms_I = X_rep_atoms_I.to(device)
        crd_mask_rep_atoms_I = crd_mask_rep_atoms_I.to(device)

        N1, N2, pred_bins = pred_distogram.shape
        # print(f"Predicted distogram shape: [{N1}, {N2}, {pred_bins}]")
        if N1 != N2:
            print(f"Warning: Asymmetric distogram dimensions: {N1} x {N2}")
        if pred_bins != bins + 1:
            print(
                f"Warning: Mismatch in number of bins. Predicted distogram has {pred_bins} bins while expected {bins + 1} bins."
            )

        gt_seq_len = X_rep_atoms_I.shape[0]
        if N1 != gt_seq_len:
            print(
                f"Warning: Sequence length mismatch. Predicted distogram has length {N1} while ground truth has length {gt_seq_len}"
            )
        mask_len = crd_mask_rep_atoms_I.shape[0]
        if mask_len != gt_seq_len:
            print(
                f"Warning: Mask length mismatch. Mask has length {mask_len} while ground truth has length {gt_seq_len}"
            )

        distance_map = torch.cdist(X_rep_atoms_I, X_rep_atoms_I)

        # map the NaN values to a large number
        distance_map[distance_map.isnan()] = 9999.0
        # discretize the distance
        bins = torch.linspace(min_distance, max_distance, bins, device=device)
        binned_distances = torch.bucketize(distance_map, bins)

        crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(
            -1
        ) * crd_mask_rep_atoms_I.unsqueeze(-2)
        distogram_cce = nn.CrossEntropyLoss(reduction="none")(
            pred_distogram.permute(-1, -2, -3)[None], binned_distances[None]
        )
        return distogram_cce[..., crd_mask_rep_atom_II].sum() / (
            crd_mask_rep_atom_II.sum() + 1e-4
        )

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        # distogram cross-entropy loss
        X_rep_atoms_I = pipeline_output["ground_truth"]["coord_token_lvl"]
        crd_mask_rep_atoms_I = pipeline_output["ground_truth"]["mask_token_lvl"]
        pred_num_bins = self.distogram_config.bins - 1

        loss = self.distogram_cce_loss(
            pred_distogram=trunk_output["distogram"],
            X_rep_atoms_I=X_rep_atoms_I,
            crd_mask_rep_atoms_I=crd_mask_rep_atoms_I,
            bins=pred_num_bins,
        )

        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss,
                    "weight": self.weight,
                }
            }
        )
        return loss * self.weight, loss_dict


def unbin_logits(logits, max_distance, num_bins):
    """
    Unbin the logits to get the matrix. Differs from metric utils version in that it keeps the logits attached to the graph.
    Args:
        logits: [B, num_bins, L, X], binned logits  where X is 23 for plddt and L for pae and pde
        max_distance: float, maximum distance
        num_bins: int, number of bins
    Returns:
        unbinned: [B, L, L], unbinned matrix
    """
    midpoints = find_bin_midpoints(max_distance, num_bins, device=logits.device)
    probabilities = torch.nn.Softmax(dim=1)(logits).float()
    unbinned = (probabilities * midpoints[None, :, None, None]).sum(dim=1)
    return unbinned


class ContactLoss(Loss_ABC):
    """
    Implements both inter-contact and intra-contact losses based on the distogram output.
    """

    def __init__(
        self,
        _name=None,
        contact_type="inter",
        cutoff=22.0,
        k=1,
        l=None,
        min_seq_sep=9,
        weight=1.0,
    ):
        """
        Args:
            _name: Name of the loss
            contact_type: "inter" for target-binder contacts, "intra" for within-binder contacts
            cutoff: Distance cutoff for considering a contact (22Å for inter, 14Å for intra)
            k: Number of contacts per binder residue to optimize (1 for inter, 2 for intra)
            l: Number of binder residues to consider (defaults to all)
            min_seq_sep: Minimum sequence separation for intra-chain contacts (default 9)
            weight: Weight for this loss term
        """
        super().__init__()
        self._name = _name
        self.contact_type = contact_type
        self.cutoff = float(cutoff)
        self.k = int(k)
        self.l = l
        self.min_seq_sep = min_seq_sep
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        distogram = trunk_output["distogram"]  # Shape [I, I, bins]
        chain_ids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        device = distogram.device

        if not isinstance(chain_ids, torch.Tensor):
            if hasattr(chain_ids, "dtype") and np.issubdtype(chain_ids.dtype, np.str_):
                unique_chains = np.unique(chain_ids)
                chain_id_map = {chain: i for i, chain in enumerate(unique_chains)}
                numerical_chains = np.array(
                    [chain_id_map[chain] for chain in chain_ids]
                )
                chain_ids = torch.tensor(numerical_chains, device=device)
            else:
                chain_ids = torch.tensor(chain_ids, device=device)

        # binder and target masks
        first_chain_id = chain_ids[0].item()
        binder_mask = chain_ids == first_chain_id
        target_mask = ~binder_mask

        if self.l is None:
            self.l = binder_mask.sum().item()

        device = distogram.device
        num_bins = distogram.shape[-1]
        bin_step = 20.0 / (num_bins - 1)  # assuming 2-22 range with 64 bins
        bins = torch.arange(2.0, 22.0 + bin_step, bin_step, device=device)

        # q_i,j,m (softmax of distogram)
        q = F.softmax(distogram, dim=-1)

        # bins mask
        bins_mask = torch.zeros_like(q)
        for m in range(num_bins):
            bins_mask[..., m] = (bins[m] < self.cutoff).float()

        # q*_i,j,m
        penalized_q = F.softmax(distogram - 1e7 * (1 - bins_mask), dim=-1)

        # p_i,j = -sum(q*_i,j,m * log(q_i,j,m))
        contact_loss = -torch.sum(penalized_q * torch.log(q + 1e-8), dim=-1)

        if self.contact_type == "inter":
            binder_indices = torch.where(binder_mask)[0]
            target_indices = torch.where(target_mask)[0]
            inter_losses = contact_loss[binder_indices][:, target_indices]

            # for each binder residue, get the k lowest contact losses
            binder_losses = []
            for i in range(len(binder_indices)):
                binder_res_losses = inter_losses[i]
                sorted_losses, _ = torch.sort(binder_res_losses)
                # Take the k lowest losses (k=1 for inter-contact)
                binder_losses.append(sorted_losses[: self.k].mean())

            # Get the l lowest residue losses
            binder_losses = torch.stack(binder_losses)
            sorted_losses, _ = torch.sort(binder_losses)
            loss = sorted_losses[: self.l].mean()

        elif self.contact_type == "intra":
            binder_indices = torch.where(binder_mask)[0]
            intra_losses = contact_loss[binder_indices][:, binder_indices]

            # mask for residues distant in sequence (|i-j| >= 9)
            seq_dist_mask = torch.zeros_like(intra_losses, dtype=torch.bool)
            for i in range(len(binder_indices)):
                for j in range(len(binder_indices)):
                    if abs(int(binder_indices[i]) - int(binder_indices[j])) >= 9:
                        seq_dist_mask[i, j] = True

            # sequence distance mask
            masked_intra_losses = torch.where(
                seq_dist_mask, intra_losses, torch.full_like(intra_losses, float("inf"))
            )

            # for each binder residue, get the k lowest contact losses
            binder_losses = []
            for i in range(len(binder_indices)):
                binder_res_losses = masked_intra_losses[i]
                finite_losses = binder_res_losses[binder_res_losses != float("inf")]
                if len(finite_losses) >= self.k:
                    sorted_losses, _ = torch.sort(finite_losses)
                    # take the k lowest losses (k=2 for intra-contact)
                    binder_losses.append(sorted_losses[: self.k].mean())
                else:
                    binder_losses.append(torch.tensor(0.0, device=device))

            # Get the l lowest residue losses
            binder_losses = torch.stack(binder_losses)
            sorted_losses, _ = torch.sort(binder_losses)
            loss = sorted_losses[: self.l].mean()

        else:
            raise ValueError(f"Unknown contact_type: {self.contact_type}")

        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss.detach(),
                    "weight": self.weight,
                }
            }
        )

        return loss * self.weight, loss_dict


class DistogramInterfaceEntropyLoss(Loss_ABC):
    def __init__(self, _name=None, weight=1.0):
        super().__init__()
        self._name = _name
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        """
        calculate the entropy of the distogram at the interface region and return it as a loss.
        """
        distogram = trunk_output["distogram"]  # Shape [I, I, bins]
        chain_ids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        device = distogram.device

        if not isinstance(chain_ids, torch.Tensor):
            if hasattr(chain_ids, "dtype") and np.issubdtype(chain_ids.dtype, np.str_):
                unique_chains = np.unique(chain_ids)
                chain_id_map = {chain: i for i, chain in enumerate(unique_chains)}
                numerical_chains = np.array(
                    [chain_id_map[chain] for chain in chain_ids]
                )
                chain_ids = torch.tensor(numerical_chains, device=device)
            else:
                chain_ids = torch.tensor(chain_ids, device=device)

        prob_dist = F.softmax(distogram, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)

        # mask different chain
        first_chain_id = chain_ids[0].item()
        binder_mask = chain_ids == first_chain_id
        target_mask = ~binder_mask

        binder_indices = torch.where(binder_mask)[0]
        target_indices = torch.where(target_mask)[0]

        # entropy at the interface
        interface_entropy = entropy[binder_indices][:, target_indices]

        loss = interface_entropy.mean()

        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": loss.detach(),
                    "weight": self.weight,
                }
            }
        )

        return loss * self.weight, loss_dict


class WeightedInterfaceEntropyLoss(Loss_ABC):
    def __init__(self, _name=None, weight=1.0):
        super().__init__()
        self._name = _name
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        """
        Compute entropy of distogram at interface, weighted by proximity (importance).
        """
        distogram = trunk_output["distogram"]  # Shape [I, I, bins]
        chain_ids = pipeline_output["ground_truth"]["chain_iid_token_lvl"]
        device = distogram.device

        if not isinstance(chain_ids, torch.Tensor):
            if hasattr(chain_ids, "dtype") and np.issubdtype(chain_ids.dtype, np.str_):
                unique_chains = np.unique(chain_ids)
                chain_id_map = {chain: i for i, chain in enumerate(unique_chains)}
                numerical_chains = np.array(
                    [chain_id_map[chain] for chain in chain_ids]
                )
                chain_ids = torch.tensor(numerical_chains, device=device)
            else:
                chain_ids = torch.tensor(chain_ids, device=device)

        binder_mask = chain_ids == chain_ids[0].item()
        target_mask = ~binder_mask

        binder_indices = torch.where(binder_mask)[0]
        target_indices = torch.where(target_mask)[0]

        prob_dist = F.softmax(distogram, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)  # [I, I]

        num_bins = distogram.shape[-1]
        bin_step = 20.0 / (num_bins - 1)
        bins = torch.arange(2.0, 22.0 + bin_step, bin_step, device=device)
        expected_distance = torch.sum(bins * prob_dist, dim=-1)

        # importance = inverse of expected distance
        importance_weights = 1.0 / (expected_distance + 1.0)

        interface_entropy = entropy[binder_indices][:, target_indices]
        interface_importance = importance_weights[binder_indices][:, target_indices]

        # weighted entropy
        weighted_entropy = (interface_entropy * interface_importance).mean()

        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "value": weighted_entropy.detach(),
                    "weight": self.weight,
                }
            }
        )

        return weighted_entropy * self.weight, loss_dict


class DistanceConstraint(Loss_ABC):
    """
    Distance extraction for enzyme design cases.
    Always returns individual raw predicted distances.
    """

    def __init__(self, _name=None, constraints=None, weight=1.0):
        """
        Args:
            _name: Name of the loss
            constraints: List of constraint dictionaries, each containing:
                - entity1: int, entity ID for first residue/atom
                - res1: int, residue ID for first residue/atom
                - entity2: int, entity ID for second residue/atom
                - res2: int, residue ID for second residue/atom
            weight: Weight for this loss term
        """
        super().__init__()
        self._name = _name
        self.constraints = constraints or []
        self.weight = float(weight)

    def forward(
        self,
        loss_dict,
        trunk_input,
        trunk_output,
        confidence_input,
        confidence_data,
        pipeline_output,
    ):
        if not self.constraints:
            return torch.tensor(0.0, device=trunk_output["distogram"].device), loss_dict

        distogram_unbinned = trunk_output[
            "distogram_unbinned"
        ]  # [I, I] unbinned distances
        device = distogram_unbinned.device

        distances = []
        constraint_info = {}

        for i, constraint in enumerate(self.constraints):
            entity1 = constraint["entity1"]
            res1 = constraint["res1"]
            entity2 = constraint["entity2"]
            res2 = constraint["res2"]

            # Find token locations for both residues
            entity1_mask = trunk_input["f"]["entity_id"] == entity1
            res1_mask = trunk_input["f"]["residue_index"] == (
                res1 - 1
            )  # Convert to 0-indexed
            tok1_loc = torch.where(torch.multiply(entity1_mask, res1_mask))[0]

            entity2_mask = trunk_input["f"]["entity_id"] == entity2
            res2_mask = trunk_input["f"]["residue_index"] == (
                res2 - 1
            )  # Convert to 0-indexed
            tok2_loc = torch.where(torch.multiply(entity2_mask, res2_mask))[0]

            if len(tok1_loc) == 0 or len(tok2_loc) == 0:
                distance = torch.tensor(0.0, device=device)
            else:
                distance = distogram_unbinned[tok1_loc[0], tok2_loc[0]]

            distances.append(distance)
            constraint_info[f"constraint_{i}"] = {
                "predicted_distance": distance.detach().item(),
                "entity1": entity1,
                "res1": res1,
                "entity2": entity2,
                "res2": res2,
            }

        loss_dict.update(
            {
                self._name: {
                    "class": self.__class__.__name__,
                    "distances": distances,
                    "raw_distances": [d.detach().item() for d in distances],
                    "weight": self.weight,
                    "constraints": constraint_info,
                }
            }
        )

        return torch.tensor(0.0, device=device), loss_dict
