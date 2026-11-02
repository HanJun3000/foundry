# RFO cycling

Each cycle chooses RF3 gradient/MCMC optimization with probability
`backprop_fraction` (default 0.5), or Boltz-2 prediction otherwise. Both
branches then redesign unfixed chain-A residues with an external MPNN
installation and feed the redesigned complex into the next cycle.

```text
RF3 gradient optimization ─┐
                          ├─ structure → external MPNN → next cycle
Boltz-2 prediction ────────┘
```

See the [project README](../../../README.md) and
[installation guide](../../../docs/installation.md) for setup. Each model
runs as a separate local process, or optionally in its own Apptainer image.
The driver requires NumPy, PyYAML and Biopython.

## Inputs

- `input_structure`: initial PDB/mmCIF complex; the designed protein is chain A.
- `af3_json_template`: optional AF3-style full-complex specification for
  explicit ligand identities or embedded MSAs; otherwise derived from the complex.
- `rf3_config`: advanced RF3 optimizer config; defaults to `backprop/configs/cycle_ppi.yaml`.
- `msa_paths`: per-chain MSA files, used by both branches.
- `templates`: per-chain structural templates, used by RF3.

Paths in run configurations are relative to the working directory. Chain A
must be a standard protein with complete backbone atoms for external MPNN.
MPNN preserves all other chains, including ligands. Use a separate chain ID
for each ligand entity. `cyclic` is passed to Boltz for chain A; the RF3 branch
does not currently add explicit cyclic bond constraints.

## Fixed residues and MPNN

Boltz fixes chain-A runs of at least five residues above
`template_plddt_threshold` (default 80). RF3 fixes chain-A residues with any
atom within 8 Å of chain B. Both masks use zero-based sequence positions;
the MPNN adapter translates these to the actual PDB author numbering,
including insertion codes. Partner chains are never redesigned.

`MPNN_MODEL_TYPE=protein_mpnn` selects ProteinMPNN and
`MPNN_MODEL_TYPE=ligand_mpnn` selects LigandMPNN in the upstream checkout.
The first generated sample is carried forward; all samples are retained.
Fully fixed binders bypass sampling. The adapter calls upstream `run.py`
and does not import the bundled Foundry MPNN engine.

## Outputs

```text
inputs/<stem>_<cycle>.pdb    # redesigned complexes for successive cycles
recycle_<n>/
  *.cif / *.pdb              # RF3/Boltz structure
  rf3_config.yaml           # RF3 branch: exact per-cycle config
  mpnn/
    mpnn_design.pdb         # handoff complex
    mpnn_result.json        # chain-A sequence and handoff path
    mpnn_spec.json          # residue mask and sampling settings
    samples/                # upstream FASTA and structures (if sampled)
  <stem>/                   # RF3 history.json and modelhub_pred/<step>.cif
<stem>_records.json
```

`backprop_fraction: 0` needs Boltz and MPNN; `1` needs RF3 and MPNN.
Intermediate fractions need all three. Model-step failures are retried up to
three times the requested cycle count; an incomplete run saves records and
returns a nonzero exit status. External MPNN errors abort immediately.

## RF3 configuration

The default `cycle_ppi.yaml` reads `RF3_CKPT`, uses ten recycles, and optimizes
`pae_interface_mean`. A confidence-based objective is required to write a
predicted structure for MPNN. Contact-only objectives are available in the
standalone optimizer but do not produce the diffusion structure required by
the cycling driver. Config files are written into the output directory, so
installed package directories may be read-only.
