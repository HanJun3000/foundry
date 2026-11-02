import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from rfo import cli
from rfo.cycling import cycle
from rfo.cycling.boltz import boltz_runner
from rfo.cycling.mpnn import mpnn_worker


def test_cli_override_precedence_and_asset_settings(tmp_path, monkeypatch):
    source = tmp_path / "source.pdb"
    source.touch()
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                input_structure=str(source),
                out_dir=str(tmp_path / "out"),
                total_cycles=4,
                assets={"mpnn_repo": "/yaml/repo"},
            )
        )
    )
    monkeypatch.setenv("MPNN_REPO", "/env/repo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rfo",
            "--config",
            str(config),
            "--total-cycles",
            "2",
            "--mpnn-repo",
            "/cli/repo",
        ],
    )
    seen = []
    monkeypatch.setattr(cycle, "run_loop", seen.append)
    cli.main()
    assert seen[0].total_cycles == 2
    assert os.environ["MPNN_REPO"] == "/cli/repo"
    assert Path(seen[0].rf3_config).is_file()


def test_cli_help_without_model_imports():
    result = subprocess.run(
        [sys.executable, "-m", "rfo.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--mpnn-repo" in result.stdout
    assert "--boltz-executable" in result.stdout


def test_rf3_config_is_written_under_run_directory(tmp_path, monkeypatch):
    config = tmp_path / "base.yaml"
    config.write_text("loss_type: pae_interface_mean\ncheckpoint_path: /test.ckpt\n")
    source = tmp_path / "complex.pdb"
    source.touch()
    output = tmp_path / "output"
    monkeypatch.setattr(cycle, "RF3_SIF", "")
    commands = []
    monkeypatch.setattr(
        cycle.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd)
    )
    # The stand-in RF3 process doesn't generate structures; inspect its config handoff.
    metrics, structure, fixed = cycle.run_rf3_branch(
        str(config), str(source), 0, str(output), False
    )
    generated = output / "recycle_1" / "rf3_config.yaml"
    assert generated.is_file()
    assert str(generated.parent) == commands[0][-1]
    cfg = yaml.safe_load(generated.read_text())
    assert cfg["input"] == str(source)
    assert cfg["output_path"] == str(generated.parent)
    assert structure is None
    assert metrics["RF3_Status"] == "Failed"


def test_boltz_process_preserves_paths_with_spaces(tmp_path, monkeypatch):
    config = tmp_path / "input complex.yaml"
    config.write_text("sequences: []\n")
    cache = tmp_path / "boltz cache"
    monkeypatch.setattr(boltz_runner, "BOLTZ_SIF", "")
    monkeypatch.setattr(boltz_runner, "BOLTZ_EXECUTABLE", "/env with spaces/bin/boltz")
    monkeypatch.setattr(boltz_runner, "BOLTZ_CACHE", str(cache))
    calls = []
    monkeypatch.setattr(
        boltz_runner.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )
    success, _ = boltz_runner.run_boltz(
        str(config), str(tmp_path / "output"), num_samples=2
    )
    assert success
    command, kwargs = calls[0]
    assert command[:3] == ["/env with spaces/bin/boltz", "predict", str(config)]
    assert command[command.index("--diffusion_samples") + 1] == "2"
    assert not kwargs.get("shell")


def test_boltz_only_does_not_require_rf3(tmp_path, monkeypatch):
    monkeypatch.setattr(mpnn_worker, "settings", lambda: None)
    monkeypatch.setattr(boltz_runner, "BOLTZ_SIF", "")
    monkeypatch.setattr(boltz_runner, "BOLTZ_CKPT", "")
    monkeypatch.setattr(boltz_runner, "BOLTZ_EXECUTABLE", sys.executable)
    cycle.validate_dependencies(argparse.Namespace(backprop_fraction=0.0))


def test_default_rf3_checkpoint_requires_configuration(monkeypatch):
    monkeypatch.setattr(mpnn_worker, "settings", lambda: None)
    monkeypatch.setattr(cycle, "RF3_SIF", "")
    monkeypatch.setattr(cycle, "RF3_PYTHON", sys.executable)
    monkeypatch.delenv("RF3_CKPT", raising=False)
    config = cli.HERE / "backprop" / "configs" / "cycle_ppi.yaml"
    with pytest.raises(ValueError, match="Set RF3_CKPT"):
        cycle.validate_dependencies(
            argparse.Namespace(backprop_fraction=1.0, rf3_config=str(config), loss=None)
        )


def test_failed_cycles_save_records_and_return_failure(tmp_path, monkeypatch):
    source = tmp_path / "complex.pdb"
    source.write_text("END\n")
    monkeypatch.setattr(cycle, "validate_dependencies", lambda args: None)
    monkeypatch.setattr(
        cycle, "run_boltz_branch", lambda *args, **kwargs: ({}, None, [])
    )
    params = dict(
        cli.DEFAULTS,
        input_structure=str(source),
        out_dir=str(tmp_path / "out"),
        total_cycles=1,
        backprop_fraction=0.0,
    )
    with pytest.raises(RuntimeError, match="Only 0/1"):
        cycle.run_loop(argparse.Namespace(**params))
    assert json.loads((tmp_path / "out" / "complex_records.json").read_text()) == []


def test_af3_template_updates_chain_a_even_when_target_is_first(tmp_path, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "sequences": [
                    {"protein": {"id": "B", "sequence": "TARGET"}},
                    {"protein": {"id": "A", "sequence": "OLD"}},
                ]
            }
        )
    )
    source = tmp_path / "complex.pdb"
    source.touch()
    monkeypatch.setattr(cycle, "get_chain_sequence", lambda *args: "NEW")
    monkeypatch.setattr(cycle, "run_boltz", lambda *args, **kwargs: (False, 0.0))
    cycle.run_boltz_branch(str(template), str(source), 0, str(tmp_path), 1, 80, None)
    generated = json.loads(
        (tmp_path / "recycle_1" / "complex_recycle_1.json").read_text()
    )[0]
    assert generated["sequences"][0]["protein"]["sequence"] == "TARGET"
    assert generated["sequences"][1]["protein"]["sequence"] == "NEW"


def test_boltz_conversion_preserves_nucleic_acids_and_ccd_ligands(tmp_path):
    source = tmp_path / "complex.json"
    source.write_text(
        json.dumps(
            {
                "sequences": [
                    {"protein": {"id": "A", "sequence": "AGS"}},
                    {"dna": {"id": "B", "sequence": "ATGC"}},
                    {"rna": {"id": "C", "sequence": "AUGC"}},
                    {"ligand": {"id": "D", "ccdCodes": ["ATP"]}},
                ]
            }
        )
    )
    output = tmp_path / "complex.yaml"
    boltz_runner.af3_json_to_boltz_yaml(str(source), str(output))
    sequences = yaml.safe_load(output.read_text())["sequences"]
    assert sequences[1] == {"dna": {"id": "B", "sequence": "ATGC"}}
    assert sequences[2] == {"rna": {"id": "C", "sequence": "AUGC"}}
    assert sequences[3] == {"ligand": {"id": "D", "ccd": "ATP"}}


def test_missing_explicit_msa_does_not_silently_disable_conditioning(tmp_path):
    source = tmp_path / "complex.json"
    source.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AGS",
                            "unpairedMsaPath": str(tmp_path / "missing.a3m"),
                        }
                    }
                ]
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="MSA"):
        boltz_runner.af3_json_to_boltz_yaml(str(source), str(tmp_path / "complex.yaml"))
