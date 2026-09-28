import json
from pathlib import Path
import pytest
from examples.scitrace_research.build_submission import _valid_official_pr_url
from examples.scitrace_research.paper_export import draft_tex, digest, export, verify


def test_agent_text_and_table_preserved_metadata_removed():
    tex = draft_tex('# Unique title\n\n## Results\nAgent-specific result is **0.731**.\n\n| Method | Value |\n|---|---|\n| A | 0.731 |\n\n## Structured References\n```json\n{}\n```\n')
    assert 'Unique title' in tex
    assert 'Agent-specific result' in tex
    assert '0.731' in tex
    assert 'longtable' in tex
    assert 'Structured References' not in tex
    assert 'Scientific agents can produce fluent papers' not in tex


def test_unsupported_content_not_silently_lost():
    with pytest.raises(ValueError, match='Unsupported'):
        draft_tex('# Title\n\n![Figure](plot.png)')


def test_agent_heading_numbers_are_not_duplicated_by_latex():
    tex = draft_tex('# Title\n\n## 3 Method\nBody.\n\n### 3.1 Setup\nDetails.\n\n## References\n[1] Source.')
    assert r'\section{Method}' in tex
    assert r'\subsection{Setup}' in tex
    assert r'\section*{References}' in tex
    assert r'\section{3 Method}' not in tex


def test_team_shifted_heading_hierarchy_is_normalized():
    tex = draft_tex(
        '## Calibrating Retriever Disagreement\n\n'
        '### Abstract\n\nBounded abstract.\n\n'
        '### 1. Introduction\n\nBody.\n\n'
        '#### 1.1 Contributions\n\nDetails.'
    )
    assert 'Calibrating Retriever Disagreement}' in tex
    assert r'\begin{abstract}' in tex
    assert r'\section{Introduction}' in tex
    assert r'\subsection{Contributions}' in tex


@pytest.mark.parametrize(
    'abstract_markup',
    [
        '## Abstract\n\nBounded abstract text.',
        '**Abstract**\n\nBounded abstract text.',
    ],
)
def test_heading_and_bold_abstract_markers_use_latex_environment(abstract_markup):
    tex = draft_tex(f'# Title\n\n{abstract_markup}\n\n## Introduction\nBody.')
    assert tex.count(r'\begin{abstract}') == 1
    assert tex.count(r'\end{abstract}') == 1
    assert r'\textbf{Abstract}' not in tex
    assert r'\section{Introduction}' in tex


def test_inline_identifiers_use_body_font_and_remain_breakable():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '`citation_support_rate`, `char-3gram-cosine`, and '
        '`c8405cf48c4cbc0f506a139b095920ec4cd30968fb9801d923719321e09ff1de`.'
    )
    assert r'citation\_\allowbreak{}support\_\allowbreak{}rate' in tex
    assert r'\texttt{' not in tex
    assert r'char-\allowbreak{}3gram-\allowbreak{}cosine' in tex
    assert r'c8405cf48c4c\allowbreak{}bc0f506a139b\allowbreak{}' in tex


def test_evidence_ids_become_numeric_citations_and_are_hidden_in_references():
    tex = draft_tex(
        '# Title\n\n## Related Work\n'
        '`evidence_a40b9beb4a650f8a931e` supports this statement.\n\n'
        '## References\n\n'
        '1. `evidence_a40b9beb4a650f8a931e` — Tuia, D. *A paper.* 2022.\n'
    )
    assert '[1] supports this statement' in tex
    assert r'\item Tuia, D. \emph{A paper.} 2022.' in tex
    assert 'evidence\_' not in tex


def test_bullet_reference_ids_become_numeric_citations():
    tex = draft_tex(
        '# Title\n\n## Related Work\n'
        '`evidence_a135671286e77abdbd55` supports this statement.\n\n'
        '## References\n\n'
        '- [`evidence_a135671286e77abdbd55`] Frisoni, G. *BioReader.* 2022.\n'
    )
    assert '[1] supports this statement' in tex
    assert r'\begin{enumerate}' in tex
    assert r'\item Frisoni, G. \emph{BioReader.} 2022.' in tex
    assert 'evidence\_' not in tex


def test_trailing_reference_ids_become_numeric_citations():
    tex = draft_tex(
        '# Title\n\n## Related Work\n'
        'A prior study `evidence_a135671286e77abdbd55` supports this.\n\n'
        '## References\n'
        '- Smith (2024). A prior study. DOI: 10.1234/example. '
        '`evidence_a135671286e77abdbd55`\n'
    )
    assert 'A prior study [1] supports this.' in tex
    assert 'evidence\_' not in tex
    assert 'Smith (2024). A prior study.' in tex


def test_verified_result_figure_is_typeset_without_rewriting_prose():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        'The budget sweep is shown below.\n\n'
        '![Evidence recall across budgets](figures/recall_by_budget.pdf)\n'
    )
    assert r'\includegraphics[width=0.78\linewidth]{figures/recall_by_budget.pdf}' in tex
    assert r'\caption{Evidence recall across budgets}' in tex
    assert 'The budget sweep is shown below.' in tex


def test_agent_figure_caption_is_not_duplicated_in_pdf():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '![Evidence recall by budget](figures/recall_by_budget.pdf)\n\n'
        '**Figure 1: Evidence recall on SciFact and NFCorpus.**\n'
    )
    assert tex.count('Evidence recall on SciFact and NFCorpus.') == 1
    assert r'\caption{Evidence recall on SciFact and NFCorpus.}' in tex


def test_plain_agent_figure_caption_is_typeset_once():
    caption = 'Evidence recall across document budgets on SciFact and NFCorpus. Panels use separate scales.'
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '![Evidence recall by budget](figures/recall_by_budget.pdf)\n\n'
        + caption + '\n\nThe results differ by dataset.\n'
    )
    assert tex.count(caption) == 1
    assert r'\caption{' + caption + '}' in tex
    assert 'The results differ by dataset.' in tex


def test_audit_only_research_hypothesis_and_claim_ids_are_hidden():
    tex = draft_tex(
        '# Title\n\n## Introduction\n'
        'We study this question (`rq_983f79148df454021c03`): why?\n\n'
        'The hypothesis (`hyp_ec9a2fe38e49333cfcad`) states an effect.\n\n'
        'Claim `claim_0b331390033b36b96e1f` records this outcome with status `supported`.\n\n'
        '## Results\n'
        'Claim `claim_0b331390033b36b96e1f` (section: results, status: `supported`) '
        'states that the result is bounded.'
    )
    assert 'rq\_' not in tex
    assert 'hyp\_' not in tex
    assert 'claim\_' not in tex
    assert 'We study this question: why?' in tex
    assert 'The hypothesis states an effect.' in tex
    assert 'This outcome is recorded with status supported.' in tex
    assert 'The supported result claim states that the result is bounded.' in tex


def test_table_has_rules_bold_header_and_bounded_columns():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '| Condition | Retrieval policy | `citation_support_rate` | 95% CI |\n'
        '|---|---|---|---|\n'
        '| Evidence-aware | `char-3gram-cosine` | **1.0** | [0.8241, 1.0] |'
    )
    assert tex.count(r'\hline') == 3
    assert r'\textbf{Condition}' in tex
    assert tex.count(r'p{0.2125\linewidth}') == 4
    assert r'>{\raggedright\arraybackslash}' in tex


def test_six_column_table_wraps_long_confidence_intervals():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '| Dataset | Candidate | Baseline | Difference | 95% CI | Probability |\n'
        '|---|---|---|---|---|---|\n'
        '| SciFact | 0.745333 | 0.735889 | 0.009444 | '
        '[-0.007222222222222222, 0.028444444444444442] | 0.853 |'
    )
    assert r'\setlength{\tabcolsep}{3pt}' in tex
    assert r'\footnotesize' in tex
    assert r'p{0.25\linewidth}' in tex
    assert r'0.\allowbreak{}007222\allowbreak{}222222\allowbreak{}' in tex
    assert r'0.\allowbreak{}028444\allowbreak{}444444\allowbreak{}' in tex


def test_seven_column_results_table_gives_confidence_interval_room():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '| Dataset | Baseline | Candidate | Delta | 95% CI | P | Status |\n'
        '|---|---|---|---|---|---|---|\n'
        '| SciFact | 0.80 | 0.84 | 0.04 | [0.0169, 0.0675] | 1.0 | supported |'
    )
    assert r'\scriptsize' in tex
    assert r'p{0.22\linewidth}' in tex
    assert tex.count(r'p{0.12\linewidth}') == 3


def test_long_title_uses_balanced_manual_line_break():
    tex = draft_tex('# Continuous Difficulty Calibration for Budgeted Retrieval')
    title_command = tex.split(r'\title{', 1)[1].split('}\n', 1)[0]
    assert r'\\' in title_command
    assert r'\hyphenpenalty' not in title_command


def test_scientific_unicode_is_pdftex_safe():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        'A--B --- C -> D: Δ, ±1σ, σ-θ, CO₂, CH₄, and §5.'
        .replace('--', '\u2013', 1)
        .replace('---', '\u2014', 1)
        .replace('->', '\u2192')
    )
    for character in ('\u2013', '\u2014', '\u2192', '\u0394', '\u00b1', '\u03c3', '\u03b8', '\u2082', '\u2084', '\u00a7'):
        assert character not in tex
    assert r'\ensuremath{\sigma}' in tex
    assert r'\ensuremath{\Delta}' in tex
    assert r'CO\textsubscript{2}' in tex


def test_pdf_mutation_invalidates_binding(tmp_path):
    paper = tmp_path / 'paper'; paper.mkdir()
    files = {'draft':tmp_path / 'draft.md', 'pdf':paper / 'paper.pdf', 'tex':paper / 'paper.tex'}
    for path in files.values(): path.write_text('original')
    (paper / 'paper_manifest.json').write_text(json.dumps({key+'_sha256':digest(path) for key,path in files.items()}))
    verify(tmp_path)
    files['pdf'].write_text('changed')
    with pytest.raises(ValueError, match='pdf changed'): verify(tmp_path)


def test_export_rejects_draft_edited_after_gate(tmp_path):
    (tmp_path / 'run_manifest.json').write_text(json.dumps({
        'status': 'success', 'draft_sha256': digest(_write(tmp_path / 'draft.md', 'Agent draft')),
    }))
    (tmp_path / 'validation_report.json').write_text(json.dumps({
        'ok': True, 'draft_sha256': digest(tmp_path / 'draft.md'),
    }))
    (tmp_path / 'draft.md').write_text('Manually changed draft')
    with pytest.raises(ValueError, match='not bound to the successful validation'):
        export(tmp_path)


def test_export_replays_current_quality_gate_after_stored_pass(tmp_path):
    draft = _write(tmp_path / 'draft.md', '# Title\n\n## Introduction\nOld content.\n')
    (tmp_path / 'run_manifest.json').write_text(json.dumps({
        'status': 'success', 'draft_sha256': digest(draft), 'paper_quality_enabled': True,
    }))
    (tmp_path / 'validation_report.json').write_text(json.dumps({
        'ok': True, 'draft_sha256': digest(draft),
    }))
    (tmp_path / 'paper_quality_report.json').write_text(json.dumps({'ok': True, 'errors': []}))
    (tmp_path / 'run_config.yaml').write_text('paper_quality:\n  enabled: true\n  required_sections: [Results]\n')
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments' / 'results.json').write_text('{}')
    (tmp_path / 'evidence').mkdir()
    (tmp_path / 'evidence' / 'literature.json').write_text('{"evidence": []}')
    with pytest.raises(ValueError, match='current paper quality gate'):
        export(tmp_path)


def _write(path, content):
    path.write_text(content)
    return path


@pytest.mark.parametrize(
    'url',
    [
        'https://atomgit.com/openJiuwen/jiuwenswarm/pulls/123',
        'https://gitcode.com/openJiuwen/jiuwenswarm/merge_requests/456',
    ],
)
def test_official_pr_url_is_accepted(url):
    assert _valid_official_pr_url(url)


@pytest.mark.parametrize(
    'url',
    [
        'https://atomgit.com/someone/jiuwenswarm/pulls/123',
        'https://atomgit.com/openJiuwen/jiuwenswarm/pulls/not-a-number',
        'https://example.com/openJiuwen/jiuwenswarm/pulls/123',
    ],
)
def test_non_official_pr_url_is_rejected(url):
    assert not _valid_official_pr_url(url)
