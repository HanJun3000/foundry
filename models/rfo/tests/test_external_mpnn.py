"""Exercise the subprocess contract with a tiny stand-in upstream CLI."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBIO, Atom, Chain, Model, PDBParser, Residue, Structure
from rfo.cycling.mpnn.mpnn_worker import redesign, settings


@pytest.fixture
def complex_pdb(tmp_path):
    structure = Structure.Structure("complex")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, rows in {
        "A": [(10, " ", "ALA"), (12, " ", "GLY"), (12, "B", "SER")],
        "B": [(7, " ", "TYR")],
        "L": [(1, " ", "LIG")],
    }.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for number, icode, name in rows:
            residue = Residue.Residue(
                ("H_LIG" if name == "LIG" else " ", number, icode), name, ""
            )
            chain.add(residue)
            for index, atom_name in enumerate(
                ["C1"] if name == "LIG" else ["N", "CA", "C", "O", "CB"]
            ):
                residue.add(
                    Atom.Atom(
                        atom_name,
                        np.array([index, number, 0.0]),
                        90.0,
                        1.0,
                        " ",
                        f"{atom_name:>4}",
                        index,
                        element=atom_name[0],
                    )
                )
    path = tmp_path / "input complex.pdb"
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))
    return path


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    repo = tmp_path / "external MPNN"
    repo.mkdir()
    (repo / "model_params").mkdir()
    for checkpoint in ("proteinmpnn_v_48_020.pt", "ligandmpnn_v_32_010_25.pt"):
        (repo / "model_params" / checkpoint).touch()
    (repo / "run.py").write_text("""
import json, os, sys
from pathlib import Path
from Bio.PDB import PDBIO, PDBParser
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
assert "PYTHONPATH" not in os.environ
Path("invocation.json").write_text(json.dumps(args))
s = PDBParser(QUIET=True).get_structure("s", args["--pdb_path"])
for chain in s[0]:
    for residue in chain:
        key = f"{chain.id}{residue.id[1]}{residue.id[2].strip()}"
        if key in args["--redesigned_residues"].split():
            residue.resname = "VAL"
out = Path(args["--out_folder"]) / "backbones"
out.mkdir()
w = PDBIO(); w.set_structure(s)
w.save(str(out / (Path(args["--pdb_path"]).stem + "_1.pdb")))
""")
    monkeypatch.setenv("MPNN_REPO", str(repo))
    monkeypatch.setenv("MPNN_PYTHON", sys.executable)
    monkeypatch.delenv("MPNN_SIF", raising=False)
    monkeypatch.delenv("MPNN_CKPT", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/must/not/leak/foundry/models/mpnn")
    return repo


@pytest.mark.parametrize("model", ["protein_mpnn", "ligand_mpnn"])
def test_external_design_preserves_complex_and_maps_residue_ids(
    complex_pdb, upstream, tmp_path, monkeypatch, model
):
    monkeypatch.setenv("MPNN_MODEL_TYPE", model)
    sequence, output = redesign(
        str(complex_pdb), [0], str(tmp_path / "results with spaces"), 3, 0.25, 19
    )
    assert sequence == "AVV"
    args = json.loads((upstream / "invocation.json").read_text())
    assert args["--redesigned_residues"] == "A12 A12B"
    assert args["--chains_to_design"] == "A"
    assert args["--number_of_batches"] == "3"
    assert args["--temperature"] == "0.25"
    assert args["--seed"] == "19"
    assert args["--model_type"] == model
    assert f"--checkpoint_{model}" in args
    result = PDBParser(QUIET=True).get_structure("out", output)[0]
    original = PDBParser(QUIET=True).get_structure("in", str(complex_pdb))[0]
    assert "CB" in result["A"][10]  # fixed side chain retained
    assert "CB" not in result["A"][12]  # mutated side chain removed
    for chain in ("B", "L"):
        for old, new in zip(original[chain].get_atoms(), result[chain].get_atoms()):
            assert old.name == new.name
            assert old.parent.resname == new.parent.resname
            np.testing.assert_array_equal(old.coord, new.coord)
    metadata = json.loads((Path(output).parent / "mpnn_result.json").read_text())
    assert metadata["sequence"] == "AVV"
    assert Path(metadata["structure_pdb"]).is_file()


def test_all_fixed_skips_upstream(complex_pdb, upstream, tmp_path):
    seq, output = redesign(
        str(complex_pdb), [0, 1, 2], str(tmp_path / "fixed"), 1, 0.1, 1
    )
    assert seq == "AGS"
    assert Path(output).read_bytes() == complex_pdb.read_bytes()
    assert not (upstream / "invocation.json").exists()


def test_missing_install_has_setup_message(monkeypatch):
    monkeypatch.delenv("MPNN_REPO", raising=False)
    with pytest.raises(ValueError, match="installation.md"):
        settings()


def test_missing_weights_fail_before_launch(upstream):
    (upstream / "model_params" / "proteinmpnn_v_48_020.pt").unlink()
    with pytest.raises(FileNotFoundError, match="get_model_params"):
        settings()


def test_invalid_mask_rejected(complex_pdb, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        redesign(str(complex_pdb), [3], str(tmp_path / "invalid"), 1, 0.1, 1)


def test_missing_chain_rejected(complex_pdb, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        redesign(str(complex_pdb), [], str(tmp_path / "missing"), 1, 0.1, 1, "Z")


def test_success_without_output_cannot_reuse_stale_result(
    complex_pdb, upstream, tmp_path
):
    out = tmp_path / "stale"
    out.mkdir()
    (out / "mpnn_result.json").write_text('{"sequence":"STALE"}')
    (upstream / "run.py").write_text("# exit successfully without writing a design\n")
    with pytest.raises(RuntimeError, match="no first design"):
        redesign(str(complex_pdb), [], str(out), 1, 0.1, 1)
    assert not (out / "mpnn_result.json").exists()


def test_redesigned_complex_is_used_by_the_next_cycle(
    complex_pdb, upstream, tmp_path, monkeypatch
):
    import argparse

    from rfo.cli import DEFAULTS
    from rfo.cycling import cycle

    seen_sequences = []

    def structure_step(template, structure, *args, **kwargs):
        seen_sequences.append(cycle.get_chain_sequence(structure, "A"))
        return {"method": "Boltz"}, structure, [0]

    monkeypatch.setattr(cycle, "validate_dependencies", lambda args: settings())
    monkeypatch.setattr(cycle, "run_boltz_branch", structure_step)
    output = tmp_path / "cycling"
    params = dict(
        DEFAULTS,
        input_structure=str(complex_pdb),
        out_dir=str(output),
        total_cycles=2,
        backprop_fraction=0.0,
    )
    cycle.run_loop(argparse.Namespace(**params))
    assert seen_sequences == ["AGS", "AVV"]
    records = json.loads((output / "input complex_records.json").read_text())
    assert [r["mpnn_sequence"] for r in records] == ["AVV", "AVV"]
    assert (output / "inputs" / "input complex_2.pdb").is_file()
