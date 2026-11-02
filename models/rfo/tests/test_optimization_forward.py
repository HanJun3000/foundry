"""Check sequence gradients and the diffusion stop-gradient without weights."""

from types import SimpleNamespace

import pytest
import torch
from rfo.backprop.inference_pipeline.optimization_forward import optimization_forward


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distogram_head = torch.nn.Identity()
        self.confidence_head = self.confidence
        self.inference_sampler = SimpleNamespace(sample_diffusion_like_af3=self.sample)
        self.diffusion_module = None
        self.sample_calls = 0
        self.sample_has_grad = None

    def trunk_forward_with_recycling(self, f, n_recycles):
        x = f["restype"] * 2
        for _ in range(n_recycles):
            yield {"S_inputs_I": x, "S_I": x, "Z_II": x}

    def sample(self, **kwargs):
        self.sample_calls += 1
        self.sample_has_grad = torch.is_grad_enabled()
        return dict(
            X_L=kwargs["S_trunk_I"].unsqueeze(0).repeat(2, 1, 1),
            X_noisy_L_traj=[],
            X_denoised_L_traj=[],
            t_hats=[],
        )

    def confidence(
        self, initial, single, pair, coords, seq, representative, frame_atom_idxs
    ):
        value = (initial + single + pair + coords.squeeze(0)).unsqueeze(0)
        return {
            f"{key}_logits": value for key in ("plddt", "pae", "pde", "exp_resolved")
        }


def inputs():
    restype = torch.ones((2, 3), requires_grad=True)
    return restype, dict(
        f={"restype": restype},
        t=torch.zeros(2),
        seq=None,
        rep_atom_idxs=None,
        frame_atom_idxs=None,
    )


def test_confidence_gradients_reach_sequence_but_not_diffusion():
    model = TinyModel().eval()
    restype, features = inputs()
    output = optimization_forward(model, features, 2, torch.zeros(2, 3))
    output["pae"].sum().backward()
    # Three differentiable trunk terms * factor 2 * two diffusion samples.
    torch.testing.assert_close(restype.grad, torch.full_like(restype, 12))
    assert not model.sample_has_grad
    assert not output["X_pred_rollout_L"].requires_grad
    assert output["pae"].shape == (2, 2, 3)


def test_contact_objective_skips_diffusion_and_retains_gradients():
    model = TinyModel().eval()
    restype, features = inputs()
    output = optimization_forward(model, features, 1, None, skip_diffusion=True)
    output["distogram"].sum().backward()
    torch.testing.assert_close(restype.grad, torch.full_like(restype, 2))
    assert model.sample_calls == 0
    assert output["X_L"] is None


def test_evaluation_does_not_build_gradients():
    model = TinyModel().eval()
    _, features = inputs()
    with torch.no_grad():
        output = optimization_forward(model, features, 1, torch.zeros(2, 3))
    assert not output["pae"].requires_grad


def test_training_mode_is_rejected():
    _, features = inputs()
    with pytest.raises(ValueError, match="eval mode"):
        optimization_forward(TinyModel(), features, 1, None)
