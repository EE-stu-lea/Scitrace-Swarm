import json
from pathlib import Path
import shutil
import pytest
from examples.scitrace_research.workflow.budget_effect_figure import load_budget_effects
from examples.scitrace_research.workflow.result_figures import budget_curve_data


def copied_fixture(tmp_path):
    fixture=Path(__file__).resolve().parents[5]/'examples/scitrace_research/fixtures/route_b2_final'
    target=tmp_path/'experiments/route_b2_artifacts'
    shutil.copytree(fixture,target)
    return target


def test_all_budgets_have_verified_intervals(tmp_path):
    copied_fixture(tmp_path)
    data=load_budget_effects(tmp_path)
    for series in data['series']:
        assert len(series['ci95_by_budget'])==4
        assert series['ci95_by_budget'][2]==series['primary_ci95']
    assert 'not zero-shot' in data['fiqa_scope']


def test_nonprimary_interval_drift_is_rejected(tmp_path):
    target=copied_fixture(tmp_path)/'scifact/results.json'
    data=json.loads(target.read_text())
    data['paired_bootstrap_by_budget']['2']['convex_fixed']['recall']['observed_difference']=.99
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='budget 2 difference'):
        load_budget_effects(tmp_path)


@pytest.mark.parametrize('invalid_recall', [float('nan'), float('inf'), -0.1, 1.1])
def test_recall_curve_rejects_invalid_values(invalid_recall):
    results = {'datasets': {dataset: {'methods_by_budget': {
        str(budget): {'knn_calibration': {'recall': 0.5}, 'convex_fixed': {'recall': 0.4}}
        for budget in (2, 3, 5, 8)
    }} for dataset in ('scifact', 'nfcorpus')}}
    results['datasets']['scifact']['methods_by_budget']['3']['knn_calibration']['recall'] = invalid_recall
    with pytest.raises(ValueError, match=r'scifact recall must be finite and in \[0, 1\]'):
        budget_curve_data(results)
