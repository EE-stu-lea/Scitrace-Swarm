import pytest
from examples.scitrace_research.workflow.method_contract import audit_method_contract

@pytest.mark.parametrize('text,code', [
    ('The depth range is 0..10.', 'paper_zero_depth_claim'),
    ('Every test query contributes exactly five documents.', 'paper_per_query_fixed_allocation'),
    ('Intervals were computed only at budget 5.', 'paper_missing_sweep_intervals_claim'),
    ('Other sweep points are without intervals.', 'paper_missing_sweep_intervals_claim'),
    ('Exact values are in the Structured References JSON block.', 'paper_audit_metadata_in_prose'),
    ('Figure 2 (figures/budget_effect.pdf).', 'paper_audit_metadata_in_prose'),
])
def test_rejects_observed_method_regressions(text, code):
    assert code in {x['code'] for x in audit_method_contract(text)}

def test_precise_description_and_markdown_target_are_allowed():
    assert audit_method_contract('Integer depths range from 1 to 10. The batch mean is 5. '
        'Pointwise intervals condition on the realized allocation. '
        '![Effect with intervals](figures/budget_effect.pdf)') == []


@pytest.mark.parametrize('text', [
    'Depths are not in the range 0..10; at least one document is allocated.',
    'Not every test query receives five documents.',
    'The fixed-depth baseline gives every query exactly five documents.',
    'Every query receives five documents in the fixed-depth baseline.',
    'FiQA is separately trained and does not evaluate zero-shot transfer.',
    'The separately trained FiQA calibration uses lexical retrieval.',
    'FiQA has its own calibration queries.',
    'FiQA does not test zero-shot transfer and is separately trained.',
    'Intervals are not computed only at budget 5.',
    'Training subsets do not refit the scaler.',
    '[Supplementary figure](figures/budget_effect.pdf)',
])
def test_correct_scope_and_explicit_negation_are_allowed(text):
    assert audit_method_contract(text) == []


@pytest.mark.parametrize('text,code', [
    ('Every query receives eight documents.', 'paper_per_query_fixed_allocation'),
    ('FiQA evaluates zero-shot transfer.', 'paper_fiqa_zero_shot_claim'),
    ('FiQA is a lexical out-of-domain stress test.', 'paper_fiqa_training_scope_missing'),
    ('FiQA is not separately trained.', 'paper_fiqa_training_scope_missing'),
    ('The training subsets refit the scaler.', 'paper_subsampling_refit_claim'),
    ('See figures/budget_effect.pdf for the effect.', 'paper_internal_workflow_in_prose'),
    ('The Team Handoff contains all source data.', 'paper_internal_workflow_in_prose'),
])
def test_additional_observed_scope_errors_are_actionable(text, code):
    assert code in {item['code'] for item in audit_method_contract(text)}
