"""Scientific contract integration and export, with no provider calls."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from ._stubs import install_openjiuwen_stubs
install_openjiuwen_stubs()

from examples.scitrace_research.paper_export import draft_tex, digest, export, verify
from examples.scitrace_research.workflow.method_overview import render_method_overview, verify_method_overview
from examples.scitrace_research.workflow.paper_quality import audit_draft, select_literature
from examples.scitrace_research.workflow.pipeline import _run_draft_generate
from examples.scitrace_research.workflow.query_diagnostics import build_query_diagnostics, verify_query_diagnostics

FIXTURE = Path(__file__).resolve().parents[5] / "examples/scitrace_research/fixtures/route_b2_final"


def outcome_table(data):
    labels = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa_lexical": "FiQA (lexical)"}
    rows = ["## Results", "**Table 3. Measured query outcomes at mean budget five.**", "",
            "| Dataset | Wins | Ties | Losses | Rescued | Newly missed |", "|---|---:|---:|---:|---:|---:|"]
    for r in data['summaries']:
        if r['budget'] == data['primary_budget']:
            rows.append(f"| {labels[r['dataset']]} | {r['wins']} | {r['ties']} | {r['losses']} | {r['rescued_from_no_evidence']} | {r['newly_no_evidence']} |")
    return "\n".join(rows) + "\n"


def test_diagnostic_gate_catches_changed_counts_missing_dataset_and_missing_evidence():
    data = build_query_diagnostics(FIXTURE)
    config = {"enabled": True, "require_query_outcome_table": True}
    draft = outcome_table(data)
    assert audit_draft(draft, evidence=[], config=config, query_diagnostics=data)['ok']
    wrong = draft.replace('| 62 | 215 | 46 | 5 | 17 |', '| 62 | 215 | 46 | 5 | 5 |')
    errors = audit_draft(wrong, evidence=[], config=config, query_diagnostics=data)['errors']
    assert [e['code'] for e in errors] == ['paper_query_outcome_count_mismatch']
    assert not audit_draft(draft, evidence=[], config=config)['ok']
    assert not audit_draft(draft.replace('NFCorpus', 'Unknown'), evidence=[], config=config, query_diagnostics=data)['ok']


def test_narrative_gate_rejects_status_chatter_and_missing_rqs_but_is_opt_in():
    draft = 'Delegation sent.\n# Paper\n## Results\nA mean gain.'
    assert audit_draft(draft, evidence=[], config={'enabled': True})['ok']
    report = audit_draft(draft, evidence=[], config={'enabled': True, 'require_research_narrative': True})
    codes = {e['code'] for e in report['errors']}
    assert {'paper_coordinator_preamble', 'paper_research_question_missing', 'paper_query_risk_figure_missing'} <= codes


def test_writer_receives_verified_risks_and_all_figures_before_generation(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / 'experiments/route_b2_artifacts')
    captured = {}
    def write(*, prompt):
        captured['prompt'] = json.loads(prompt)
        # Demonstrate generation cannot precede source-checked figures.
        verify_method_overview(tmp_path)
        verify_query_diagnostics(tmp_path)
        return '# Model draft'
    harness = SimpleNamespace(set_manuscript_context=lambda value: captured.update(context=value), write_draft=write)
    config = {'paper_quality': {'require_research_narrative': True, 'require_query_outcome_table': True}}
    result = _run_draft_generate(topic='scoped topic', config=config, harness=harness,
        stage_context=SimpleNamespace(build_stage_context=lambda *a, **k: {}), output_root=tmp_path,
        research_question_payload={'research_question': {}, 'hypothesis': {}},
        literature_payload={'evidence': []}, execution_payload={'results': {}, 'artifact': {}}, claim_payload={'claims': []})
    assert result['draft'] == '# Model draft'
    rows = captured['prompt']['required_query_outcome_rows']
    assert next(r for r in rows if r['dataset'] == 'nfcorpus')['newly_no_evidence'] == 17
    assert captured['context'] == json.loads((tmp_path / 'manuscript_context.json').read_text())
    assert 'RQ1' in captured['context']['research_questions']


def test_method_diagram_tamper_and_stale_contract_fail(tmp_path):
    render_method_overview(tmp_path)
    assert verify_method_overview(tmp_path)['kind'].startswith('implementation')
    manifest = tmp_path / 'figures/method_overview_manifest.json'
    record = json.loads(manifest.read_text()); record['contract_sha256'] = 'changed'
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='stale'):
        verify_method_overview(tmp_path)


def test_new_figures_and_appendix_are_typeset_without_allowing_arbitrary_images():
    draft = '# Title\n\n## Method\n![Training and test boundary in the verified allocation design.](figures/method_overview.pdf)\n\n'
    draft += '## Results\n![Gain and loss contributions across the three evaluated corpora.](figures/query_risk.pdf)\n\n'
    draft += '## References\nA verified reference.\n\n## Appendix\n### Protocol\nExtended protocol.'
    tex = draft_tex(draft)
    assert r'\begin{figure}[htbp]' in tex
    assert tex.index(r'\section*{References}') < tex.index(r'\appendix')
    assert '\\FloatBarrier\n\\section*{References}' in tex
    assert '\\usepackage{hyperref,array,graphicx,float,placeins,amsmath,amssymb}' in tex
    assert '\\clearpage\n\\normalsize\n\\appendix' in tex
    assert r'\normalsize' in tex and 'Extended protocol.' in tex
    with pytest.raises(ValueError, match='Unsupported image'):
        draft_tex(draft.replace('figures/query_risk.pdf', '../arbitrary.pdf'))


def test_reserved_foundational_reference_is_selected_only_from_retrieved_candidates():
    def source(title):
        return SimpleNamespace(title=title, summary='retrieval benchmark evidence', doi='', provider='live')
    candidates = [source('retrieval query budget search fusion ' + str(i)) for i in range(5)]
    candidates.append(source('BEIR: A Heterogeneous Benchmark'))
    results, _ = select_literature(candidates, focus_terms=['retriev', 'benchmark', 'budget'],
        minimum_focus_terms=1, max_results=2, reserved_title_phrases=['BEIR', 'Nonexistent paper'])
    assert len(results) == 2 and any('BEIR' in r.title for r in results)
    assert all(r in candidates for r in results)


def test_export_binds_new_figure_copies_and_rejects_later_source_change(tmp_path, monkeypatch):
    from examples.scitrace_research.workflow.query_diagnostics import render_query_diagnostics
    shutil.copytree(FIXTURE, tmp_path / 'experiments/route_b2_artifacts')
    render_method_overview(tmp_path)
    render_query_diagnostics(tmp_path / 'experiments/route_b2_artifacts', tmp_path / 'figures')
    draft = tmp_path / 'draft.md'
    draft.write_text('# Export test fixture\n\n## Method\n![Verified training and prediction schematic.](figures/method_overview.pdf)\n\n'
                     '## Results\n![Query risk for the three saved datasets.](figures/query_risk.pdf)\n')
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'status': 'success', 'draft_sha256': digest(draft)}))
    (tmp_path / 'validation_report.json').write_text(json.dumps({'ok': True, 'draft_sha256': digest(draft)}))
    (tmp_path / 'evidence').mkdir(); (tmp_path / 'evidence/literature.json').write_text('{}')
    def compiler(*args, **kwargs):
        (kwargs['cwd'] / 'paper.pdf').write_bytes(b'compiler-test-placeholder')
        return SimpleNamespace(stdout=b'', stderr=b'', returncode=0)
    monkeypatch.setattr('examples.scitrace_research.paper_export.subprocess.run', compiler)
    export(tmp_path)
    verify(tmp_path)
    source = tmp_path / 'experiments/route_b2_artifacts/nfcorpus/results.json'
    source.write_text(source.read_text() + ' ')
    with pytest.raises(ValueError, match='source outcomes'):
        verify(tmp_path)


def test_submission_source_is_installable_shape_with_current_example_configuration(tmp_path):
    from examples.scitrace_research.build_submission import _copy_code
    _copy_code(tmp_path)
    assert (tmp_path / 'examples/__init__.py').is_file()
    assert (tmp_path / 'jiuwenbox/src/jiuwenbox/__init__.py').is_file()
    assert 'require_research_narrative: true' in (tmp_path / 'config.yaml').read_text()
    assert '-r examples/scitrace_research/requirements-paper.txt' in (tmp_path / 'requirements.txt').read_text()
