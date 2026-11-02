# RFO — backprop

RF3 gradient / MCMC **sequence optimization**. Given a target structure, it
optimizes the binder sequence by backpropagating a structure/interface loss
through RF3 (gradient-guided MCMC or gradient-proposal algorithms).

This is the RF3 branch of the cycling pipeline (`rfo --config ...`). To run the
optimizer on its own (inside the RF3 apptainer / a matching env):

```bash
cd models/rfo/src/rfo/backprop
python optimize.py --config_name my_optimizer --config_path /path/to/configs
```

## Configs (`configs/*.yaml`)

Key fields (see `configs/test_ppi_optimizer.yaml` for a full example):

| Field | Meaning |
|-------|---------|
| `input` | input JSON / CIF / PDB (or a directory of them) |
| `output_path` | where results are written |
| `algorithm` | `gradient_mcmc` or `gradient_proposal` |
| `algorithm_config.opt_steps.total_steps` | optimization steps |
| `checkpoint_path` | RF3 checkpoint(s) |
| `optimize_binder` / `interface_only` | which residues to design |
| `random_seq` / `random_seq_length` | hallucinate the binder from scratch |
| `msa_paths` / `fix_template_dict` | per-chain MSA / fixed template |
| `loss` / `loss_config` | objective (e.g. inter/intra `ContactLoss`) |

## Outputs (in `output_path/<stem><suffix>/`)

| File | Contents |
|------|----------|
| `history.json` | per-step sequences + metrics (lowest `loss` = best) |
| `modelhub_pred/<step>.cif` | predicted structure at each step |
| `modelhub_inputs.json` / `af3_inputs.json` | sequences in modelhub / AF3 format |
| `accept_traj.json` | MCMC acceptance trajectory |

## Code

- `optimize.py` — entrypoint (hydra-config driven).
- `optimizer/` — `gradient_mcmc.py`, `gradient_proposal.py`, `folding_model/modelhub_grad.py`.
- `tools/` — `af3.py` (seq↔cif, featurization), `loss.py`, `mutate.py`, `utils.py`, …
- `inference_pipeline/af3_inference_pipeline.py` — RF3 forward wrapper.

The cycling driver (`../cycling/`) reuses this optimizer as its RF3 branch.

## Portable setup

See the [installation guide](../../../docs/installation.md). A standalone
config must include `input`, `output_path`, and `checkpoint_path`, along with
the optimization settings in `configs/cycle_ppi.yaml`. The cycling driver
fills in input/output automatically. Set `RF3_CKPT` when using the default
cycle config. Absolute config directories are supported.

Other YAMLs are historical research configurations, with internal paths
replaced by `/path/to/` placeholders. Review their data, checkpoint and
algorithm settings before use; they are not ready-to-run examples.
`modelhub_pred` is a retained output name, not a dependency on ModelHub.

RFO's `inference_pipeline/optimization_forward.py` calls the public RF3 model
components with gradients through the final trunk recycle and prediction
heads, while diffusion runs without gradients. Normal RF3 inference is
unchanged. Independent AlphaFold3 evaluation must be configured separately.
