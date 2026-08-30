from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "et_statistics.py").read_text(encoding="utf-8")


def test_et_statistics_exposes_connectivity_and_distribution_outputs():
    assert "--connectivity" in SOURCE
    assert "--distribution-output-path" in SOURCE
    assert "--plot-path" in SOURCE
    assert "structure=CONNECTIVITY_STRUCTURE" in SOURCE
    assert "lesion_volume_mm3" in SOURCE
    assert "lesion_id" in SOURCE


def test_et_statistics_supports_region_specific_connected_components():
    assert "--region" in SOURCE
    assert "choices=('ET', 'TC', 'WT')" in SOURCE or 'choices=("ET", "TC", "WT")' in SOURCE
    assert "REGION_LABELS" in SOURCE
    assert "default='WT'" in SOURCE or 'default="WT"' in SOURCE
    assert "region_mask" in SOURCE
    assert "region_prefix" in SOURCE


def test_lesion_detail_output_does_not_include_grade_or_case_min_max_fields():
    detail_block = SOURCE.split("# --- Record each component individually ---", 1)[1]
    detail_block = detail_block.split("except Exception", 1)[0]

    assert "'Grade'" not in detail_block
    assert "ET_smallest_comp" not in detail_block
    assert "ET_largest_comp" not in detail_block
