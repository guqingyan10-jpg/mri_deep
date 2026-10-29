"""Native training data and the historical scoring grid must stay distinct."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts import prepare_official_nnunet as prepare
from scripts.eval_official_nnunet import evaluate_experiment, load_prediction, load_target
from scripts.prepare_ucsf_bmsr import resize_dhw


def make_native_ucsf_manifest(root):
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    shape=(24,20,16)
    # A flipped source axis exercises canonical orientation and world geometry.
    affine=np.diag([-1.,1.5,2.,1.])
    affine[0,3]=23
    for i,phase in enumerate(("train","train","train","valid","test"),start=1):
        case_id=f"{100000+i}A"
        source=root/"raw"/case_id
        source.mkdir(parents=True)
        image=(np.indices(shape)[0]*7+1000).astype(np.float32)
        mask=np.zeros(shape,dtype=np.uint8)
        mask[4:4+2*i,4:12,2:8]=1
        for channel,name in enumerate(prepare.UCSF_MODALITIES):
            nib.save(nib.Nifti1Image(image+channel*100,affine),source/f"{case_id}_{name}.nii.gz")
        label=nib.Nifti1Image(mask,affine)
        nib.save(label,source/f"{case_id}_seg.nii.gz")
        canonical=nib.as_closest_canonical(label)
        target_shape=(10,17,17)
        target=resize_dhw(np.asarray(canonical.dataobj).transpose(2,1,0).astype(np.float32),target_shape,"nearest")[None]
        spacing=nib.affines.voxel_sizes(canonical.affine)[::-1]*np.asarray(shape[::-1])/np.asarray(target_shape)
        cached=root/f"{case_id}.npz"
        np.savez_compressed(cached,image=np.zeros((4,*target_shape),dtype=np.float16),
                            mask=target.astype(np.uint8),spacing_dhw=spacing.astype(np.float32))
        rows.append({"subject_id":case_id,"patient_group":str(100000+i),"split":phase,
                     "prepared_path":str(cached),"source_dir":str(source)})
    manifest=root/"manifest.csv"
    with manifest.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return manifest


def test_native_ucsf_preserves_resolution_intensities_and_original_scoring_gt(tmp_path):
    manifest=make_native_ucsf_manifest(tmp_path/"source")
    work=tmp_path/"official"
    experiment=prepare.main(["--dataset","ucsf","--ucsf-manifest",str(manifest),
                             "--work-dir",str(work),"--allow-nonstandard-cohort"])
    assert experiment["input_space"]=="native"
    case=experiment["cases"][-1]
    native=nib.load(case["image_files"][0])
    assert native.shape==(24,20,16)
    assert nib.aff2axcodes(native.affine)==("R","A","S")
    assert np.asarray(native.dataobj).max()>1000
    target,reference,_=load_target(case,"ucsf")
    assert target.shape==(1,10,17,17)
    with np.load(tmp_path/"source/100005A.npz") as cached:
        np.testing.assert_array_equal(target,cached["mask"])
    assert Path(case["label_file"]).parent==work/"evaluation_targets"
    assert not any("100005A"in p.name for p in (Path(experiment["dataset_dir"])/"labelsTr").iterdir())

    predictions=tmp_path/"predictions"
    predictions.mkdir()
    label=nib.load(case["native_label_file"])
    nib.save(label,predictions/"100005A.nii.gz")
    native_labels=np.asarray(label.dataobj).transpose(2,1,0)
    foreground=native_labels.astype(np.float32)*.9
    np.savez_compressed(predictions/"100005A.npz",probabilities=np.stack([1-foreground,foreground]))
    actual=load_prediction(predictions,case,reference,"ucsf","native-mask",.33,"official")
    np.testing.assert_array_equal(actual,target)
    result=evaluate_experiment(work/"experiment.json",predictions,tmp_path/"evaluation",
                               decision="native-mask",probability_layout="official",allow_nonstandard_cohort=True)
    assert result["dice_mean"]==1
    assert result["lesion_recall"]==1


def test_native_ucsf_cache_from_a_different_label_is_rejected(tmp_path):
    manifest=make_native_ucsf_manifest(tmp_path/"source")
    cached=tmp_path/"source/100001A.npz"
    with np.load(cached)as data:values={key:data[key].copy()for key in data.files}
    values["mask"][0,0,0,0]=1
    np.savez_compressed(cached,**values)
    row=prepare.ucsf_rows(manifest,None,input_space="native")[0]
    with pytest.raises(ValueError,match="does not reproduce"):
        prepare.export_case(row,"ucsf",tmp_path/"work/nnUNet_raw/Dataset502_Test",True,"native")


def test_native_brats_does_not_apply_the_study_crop_or_minmax(tmp_path):
    shape=(210,210,120)
    source=np.broadcast_to(np.arange(210,dtype=np.int16)[:,None,None],shape)
    affine=np.diag([1.,2.,3.,1.]);affine[:3,3]=[5,6,7]
    image_path=tmp_path/"image.nii.gz"
    label_path=tmp_path/"seg.nii.gz"
    nib.save(nib.Nifti1Image(source,affine),image_path)
    labels=np.zeros(shape,dtype=np.uint8)
    labels[40,40,20],labels[41,40,20],labels[42,40,20]=1,2,4
    nib.save(nib.Nifti1Image(labels,affine),label_path)
    row={"case_id":"BraTS20_Training_001","split":"test","patient_group":"001",
         "source_files":[image_path]*4+[label_path]}
    record=prepare.export_case(row,"brats",tmp_path/"work/nnUNet_raw/Dataset501_Test",True,"native")
    image=nib.load(record["image_files"][0])
    assert image.shape==shape and np.asarray(image.dataobj).max()==209
    np.testing.assert_array_equal(image.affine,affine)
    target=nib.load(record["label_file"])
    assert target.shape==(170,170,100)
    np.testing.assert_array_equal(np.asarray(target.dataobj)[:3,0,0],[1,2,3])
    np.testing.assert_array_equal(target.affine[:3,3],[45,86,67])
    assert record["prediction_alignment"]=="brats-study-crop"
    predictions=tmp_path/"brats_predictions";predictions.mkdir()
    native_gt=nib.load(record["native_label_file"])
    nib.save(native_gt,predictions/"BraTS20_Training_001.nii.gz")
    from scripts.eval_official_nnunet import label_regions
    probability=label_regions(np.asarray(native_gt.dataobj),"brats").astype(np.float32)*.9
    np.savez_compressed(predictions/"BraTS20_Training_001.npz",probabilities=probability)
    predicted=load_prediction(predictions,record,target,"brats","probabilities",.33,"official")
    np.testing.assert_array_equal(predicted,label_regions(np.asarray(target.dataobj),"brats"))


def test_probability_resize_precedes_threshold_and_legacy_predictions_can_use_study_layout(tmp_path):
    native=np.asarray([.2,.8],dtype=np.float32).reshape(2,1,1)
    reference=nib.Nifti1Image(np.zeros((2,1,1),dtype=np.uint8),np.eye(4))
    native_path=tmp_path/"native.nii.gz";nib.save(reference,native_path)
    target_reference=nib.Nifti1Image(np.zeros((3,1,1),dtype=np.uint8),np.diag([2/3,1,1,1]))
    case={"case_id":"test","shape_dhw":[1,1,3],"native_shape_dhw":[1,1,2],
          "prediction_reference_file":str(native_path),"prediction_alignment":"ucsf-canonical-study-resize"}
    predictions=tmp_path/"predictions";predictions.mkdir()
    nib.save(reference,predictions/"test.nii.gz")
    foreground=native.transpose(2,1,0)
    np.savez_compressed(predictions/"test.npz",probabilities=np.stack([1-foreground,foreground]))
    actual=load_prediction(predictions,case,target_reference,"ucsf","probabilities",.33,"official")
    assert actual.ravel().tolist()==[False,True,True]
    nib.save(target_reference,predictions/"test.nii.gz")
    np.savez_compressed(predictions/"test.npz",probabilities=np.asarray([.1,.9,.1],dtype=np.float32).reshape(1,1,1,3))
    legacy=load_prediction(predictions,case,target_reference,"ucsf","probabilities",.33,"study")
    assert legacy.ravel().tolist()==[False,True,False]
