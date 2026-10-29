"""The wrapper must retain the official recipe and frozen study case split."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import prepare_official_nnunet as prepare
from scripts import run_official_nnunet as run
from test_official_nnunet_preparation import make_ucsf_manifest, ucsf_args


def experiment_fixture(tmp_path):
    manifest=make_ucsf_manifest(tmp_path)
    work=tmp_path/"official"
    experiment=prepare.main(ucsf_args(manifest,work))
    return work/"experiment.json",experiment


def test_official_commands_select_one_frozen_fold_and_default_final_checkpoint(tmp_path):
    path,experiment=experiment_fixture(tmp_path)
    args=run.cli().parse_args(["--experiment",str(path),"--allow-nonstandard-cohort"])
    commands=run.commands(args,experiment)
    assert commands["train"][commands["train"].index("3d_fullres")+1]=="0"
    assert "--val_best" not in commands["train"]
    assert "nnUNetTrainer" in commands["train"]
    assert "checkpoint_final.pth" in commands["predict"]
    assert "--disable_tta" not in commands["predict"]
    assert "--save_probabilities" in commands["predict"]
    assert "--verify_dataset_integrity" in commands["plan"]
    args.resume=True;args.disable_tta=True
    changed=run.commands(args,experiment)
    assert "--c" in changed["train"]
    assert "--disable_tta" in changed["predict"]


def test_protocol_refuses_test_labels_or_images_in_training_inventory(tmp_path):
    path,experiment=experiment_fixture(tmp_path)
    assert run.load_protocol(path,True)["split"]==experiment["split"]
    test=experiment["split"]["test"][0]
    directory=Path(experiment["dataset_dir"])
    unexpected=directory/"labelsTr"/f"{test}.nii.gz"
    unexpected.write_bytes(b"not a training case")
    with pytest.raises(ValueError,match="never test labels"):
        run.load_protocol(path,True)
    unexpected.unlink()
    unexpected=directory/"imagesTr"/f"{test}_0000.nii.gz"
    unexpected.write_bytes(b"not a training image")
    with pytest.raises(ValueError,match="four frozen modalities"):
        run.load_protocol(path,True)


def test_split_install_refuses_a_different_existing_fold(tmp_path):
    path,experiment=experiment_fixture(tmp_path)
    target=run.install_split(experiment,path.parent/"splits_final.json")
    assert json.loads(target.read_text())==[{"train":experiment["split"]["train"],"val":experiment["split"]["valid"]}]
    target.write_text("[]")
    with pytest.raises(ValueError,match="refusing to replace"):
        run.install_split(experiment,path.parent/"splits_final.json")


def test_dry_run_does_not_start_training_or_require_a_gpu(tmp_path,capsys):
    path,_=experiment_fixture(tmp_path)
    run.main(["--experiment",str(path),"--allow-nonstandard-cohort","--dry-run"])
    assert not (path.parent/"official_pipeline").exists()
    assert '"default_epoch_ceiling": 1000' in capsys.readouterr().out


@pytest.mark.parametrize("stage,flag",[("plan","-gpu_memory_target"),("train","--c"),("predict","--save_probabilities")])
def test_installed_official_cli_accepts_the_wrappers_entry_point(stage,flag,tmp_path):
    pytest.importorskip("nnunetv2")
    environment=os.environ.copy()
    environment.update({"nnUNet_raw":str(tmp_path/"raw"),"nnUNet_preprocessed":str(tmp_path/"preprocessed"),
                        "nnUNet_results":str(tmp_path/"results"),"NO_ALBUMENTATIONS_UPDATE":"1"})
    result=subprocess.run([sys.executable,str(Path(run.__file__)),"_entry",stage,"55","--help"],
                           env=environment,capture_output=True,text=True,errors="replace",timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert flag in result.stdout


def test_checkpoint_budget_reads_actual_official_epoch_and_plan(tmp_path):
    torch=pytest.importorskip("torch")
    model_dir=tmp_path/"nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
    model_dir.mkdir(parents=True)
    torch.save({"current_epoch":1000,"network_weights":{"conv.weight":torch.ones(3,4,1,1,1)}},
               model_dir/"checkpoint_final.pth")
    torch.save({},model_dir/"checkpoint_best.pth")
    (model_dir.parent/"plans.json").write_text(json.dumps({"configurations":{"3d_fullres":{
        "batch_size":2,"patch_size":[16,16,16]}}}))
    report=run.checkpoint_summary(model_dir)
    assert report["completed_epochs"]==1000
    assert report["completed_optimizer_updates"]==250000
    assert report["parameters"]==12
    assert report["planned_batch_size"]==2


def test_incomplete_or_modified_preprocessing_cannot_be_silently_reused(tmp_path):
    path,experiment=experiment_fixture(tmp_path)
    base=Path(experiment["preprocessed_dir"])/experiment["dataset_name"]
    base.mkdir(parents=True)
    with pytest.raises(FileNotFoundError,match="run --stage plan"):
        run.preprocessing_snapshot(experiment)
    (base/"nnUNetPlans.json").write_text(json.dumps({"configurations":{"3d_fullres":{"data_identifier":"cache"}}}))
    (base/"dataset.json").write_text("{}")
    (base/"dataset_fingerprint.json").write_text("{}")
    data=base/"cache";data.mkdir()
    for case in experiment["split"]["train"]+experiment["split"]["valid"]:
        for suffix in (".b2nd","_seg.b2nd",".pkl"):
            (data/f"{case}{suffix}").write_bytes(b"test fixture")
    first=run.preprocessing_snapshot(experiment)
    (base/"dataset_fingerprint.json").write_text('{"changed":true}')
    assert run.preprocessing_snapshot(experiment)!=first
    (data/f"{experiment['split']['test'][0]}.b2nd").write_bytes(b"test case")
    with pytest.raises(ValueError,match="inventory differs"):
        run.preprocessing_snapshot(experiment)


def test_result_archive_excludes_raw_predictions_and_weights(tmp_path):
    import zipfile
    out=tmp_path/"output";out.mkdir()
    (out/"run_protocol.json").write_text("{}")
    predictions=out/"predictions";predictions.mkdir()
    (predictions/"case.nii.gz").write_bytes(b"raw volume")
    (predictions/"plans.json").write_text("{}")
    (out/"checkpoint_final.pth").write_bytes(b"weights")
    experiment=tmp_path/"experiment.json";experiment.write_text("{}")
    (tmp_path/"splits_final.json").write_text("[]")
    archive=run.package_results(out,experiment,tmp_path/"model/fold_0")
    with zipfile.ZipFile(archive)as saved:
        assert "run_protocol.json"in saved.namelist()
        assert "data_protocol/experiment.json"in saved.namelist()
        assert all("predictions"not in name and not name.endswith(".pth")for name in saved.namelist())
