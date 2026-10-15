import json
from pathlib import Path
import shutil
import pytest
from examples.scitrace_research.build_submission import _valid_official_pr_url
from examples.scitrace_research.paper_export import draft_tex, digest, export, verify
from examples.scitrace_research.workflow.budget_effect_figure import load_budget_effects, render_budget_effect_figure


def test_agent_text_and_table_preserved_metadata_removed():
    tex = draft_tex('# Unique title\n\n## Results\nAgent-specific result is **0.731**.\n\n| Method | Value |\n|---|---|\n| A | 0.731 |\n\n## Structured References\n```json\n{}\n```\n')
    assert 'Unique title' in tex
    assert 'Agent-specific result' in tex
    assert '0.731' in tex
    assert r'\begin{minipage}{\linewidth}' in tex
    assert r'\begin{tabular}' in tex
    assert 'Structured References' not in tex
    assert 'Scientific agents can produce fluent papers' not in tex


def test_team_handoff_preamble_is_not_typeset_as_paper_content():
    draft = (
        'Delegation is in place. Here is the returned handoff from the draft stage.\n\n'
        '---\n\n# Research title\n\n## Abstract\n\nScientific result.\n'
    )
    tex = draft_tex(draft)
    assert 'Delegation is in place' not in tex
    assert r'\begin{abstract}' in tex
    assert 'Scientific result.' in tex
    assert draft.startswith('Delegation is in place')


def test_pdf_latex_typesets_non_latin_reference_name_without_editing_draft():
    source = '# Title\n\n## References\nАртем Сахно. Evidence × Budget…'
    tex = draft_tex(source)
    assert 'Artem Sakhno' in tex
    assert r'\ensuremath{\times}' in tex
    assert r'\ldots{}' in tex
    assert 'Артем Сахно' in source


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
    assert 'evidence\\_' not in tex


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
    assert 'evidence\\_' not in tex


def test_trailing_reference_ids_become_numeric_citations():
    tex = draft_tex(
        '# Title\n\n## Related Work\n'
        'A prior study `evidence_a135671286e77abdbd55` supports this.\n\n'
        '## References\n'
        '- Smith (2024). A prior study. DOI: 10.1234/example. '
        '`evidence_a135671286e77abdbd55`\n'
    )
    assert 'A prior study [1] supports this.' in tex
    assert 'evidence\\_' not in tex
    assert 'Smith (2024). A prior study.' in tex


def test_verified_result_figure_is_typeset_without_rewriting_prose():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        'The budget sweep is shown below.\n\n'
        '![Evidence recall across budgets](figures/recall_by_budget.pdf)\n'
    )
    assert r'\includegraphics[width=\linewidth]{figures/recall_by_budget.pdf}' in tex
    assert r'\caption{Evidence recall across budgets}' in tex
    assert 'The budget sweep is shown below.' in tex


def test_budget_effect_figure_uses_verified_artifacts(tmp_path):
    fixture = Path(__file__).resolve().parents[5] / 'examples/scitrace_research/fixtures/route_b2_final'
    artifacts = tmp_path / 'experiments/route_b2_artifacts'
    shutil.copytree(fixture, artifacts)
    figure = render_budget_effect_figure(tmp_path, tmp_path / 'figures')
    assert figure is not None
    assert figure['figure'].is_file()
    assert len(figure['sources']) == 4
    assert figure['manifest']['figure_sha256'] == digest(figure['figure'])
    assert figure['manifest']['data_sha256'] == digest(figure['data'])
    data = load_budget_effects(tmp_path)
    assert data is not None
    assert [item['dataset'] for item in data['series']] == ['scifact', 'nfcorpus', 'fiqa_lexical']
    assert all(item['budgets'] == [2, 3, 5, 8] for item in data['series'])
    tex = draft_tex('# Title\n\n## Results\n![Change in evidence recall with budget-5 paired-bootstrap intervals on SciFact, NFCorpus, and FiQA](figures/budget_effect.pdf)')
    assert r'\includegraphics[width=\linewidth]{figures/budget_effect.pdf}' in tex
    aggregate = artifacts / 'aggregate.json'
    payload = json.loads(aggregate.read_text(encoding='utf-8'))
    payload['datasets'][0]['candidate_recall'] = 0.0
    aggregate.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from aggregate'):
        load_budget_effects(tmp_path)


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


@pytest.mark.parametrize('figure_path', [
    'figures/recall_by_budget.pdf', 'figures/budget_effect.pdf',
    'figures/method_overview.pdf', 'figures/query_risk.pdf',
])
@pytest.mark.parametrize('wrapper', ['', '*', '**'])
def test_numbered_wrapped_caption_and_artifact_path_are_typeset_once(figure_path, wrapper):
    draft = (
        '# Title\n\n## Results\n'
        f'![Short alt text]({figure_path})\n\n'
        f'{wrapper}Figure 2 ({figure_path}). Observed differences at each budget.\n'
        f'Intervals are conditional on the realized allocation.{wrapper}\n\n'
        'The negative result remains visible.\n'
    )
    tex = draft_tex(draft)
    assert tex.count(r'\caption{') == 1
    assert tex.count('Observed differences at each budget.') == 1
    assert 'Intervals are conditional on the realized allocation.' in tex
    assert 'Short alt text' not in tex
    assert 'Figure 2 (' not in tex
    assert 'The negative result remains visible.' in tex
    assert f'Figure 2 ({figure_path})' in draft


def test_caption_citations_and_following_figure_discussion_are_preserved():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '![Short caption](figures/budget_effect.pdf)\n\n'
        'Figure 2: Comparison to `evidence_aabbcc`.\n\n'
        'Figure 2 shows a negative difference on FiQA.\n\n'
        '## References\n1. `evidence_aabbcc` Author. Prior work.\n'
    )
    assert r'\caption{Comparison to [1].}' in tex
    assert 'Figure 2 shows a negative difference on FiQA.' in tex


def test_image_followed_by_result_discussion_keeps_original_caption():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '![Short caption](figures/budget_effect.pdf)\n\n'
        'Figure 2 shows a negative difference on FiQA.\n'
    )
    assert r'\caption{Short caption}' in tex
    assert 'Figure 2 shows a negative difference on FiQA.' in tex


def test_separate_emphasis_spans_in_caption_remain_balanced():
    tex = draft_tex(
        '# Title\n\n## Results\n'
        '![Short caption](figures/budget_effect.pdf)\n\n'
        '**Figure 2: Observed differences.** Intervals are **pointwise**.\n'
    )
    assert r'\caption{\textbf{Observed differences.} Intervals are \textbf{pointwise}.}' in tex


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
    assert 'rq\\_' not in tex
    assert 'hyp\\_' not in tex
    assert 'claim\\_' not in tex
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


@pytest.mark.parametrize('opening,closing', [('$', '$'), (r'\(', r'\)')])
def test_inline_allocation_math_preserves_scientific_notation(opening, closing):
    equation = r'\sum_{i=1}^{n} k_i = B,\quad k_i \in \{0,\ldots,K\}'
    tex = draft_tex(f'# Title\n\n## Method\nThe constraint is {opening}{equation}{closing}.')
    assert r'\(' + equation + r'\)' in tex
    assert r'\textbackslash{}sum' not in tex


def test_inverse_distance_formula_with_norm_and_epsilon_typesets():
    equation = (
        r'v_{ij} = \frac{1}{\lVert x_i-x_j \rVert_2 + \epsilon}, '
        r'\qquad \epsilon = 10^{-6}'
    )
    tex = draft_tex(f'# Title\n\n## Method\nThe weighting is \\({equation}\\).')
    assert r'\(' + equation + r'\)' in tex


def test_escaped_hash_in_token_count_math_typesets():
    equation = r'\log(1 + \#\mathrm{tokens})'
    tex = draft_tex(f'# Title\n\n## Method\nCost is \\({equation}\\).')
    assert r'\(' + equation + r'\)' in tex


@pytest.mark.parametrize('opening,closing', [('$$', '$$'), (r'\[', r'\]')])
@pytest.mark.parametrize('newline', ['', '\n'])
def test_display_allocation_math_preserves_objective(opening, closing, newline):
    equation = r'\max_{k_1,\ldots,k_n} \sum_{i=1}^{n} \hat{r}_i(k_i)'
    tex = draft_tex(f'# Title\n\n## Method\n{opening}{newline}{equation}{newline}{closing}\n\nNext paragraph.')
    assert '\\[\n' + equation + '\n\\]' in tex
    assert 'Next paragraph.' in tex


@pytest.mark.parametrize('unsafe', [
    r'\input{secret}', r'\include{secret}', r'\write18{command}',
    r'\newcommand{\x}{secret}', r'\csname input\endcsname',
    r'\begin{document}', r'\catcode1=2', r'\text{\input{secret}}',
    r'^^5cinput{secret}', 'x%comment', 'x#1', 'x&y',
    'x}', '{x', 'x' * 2001,
])
def test_math_rejects_control_commands_and_invalid_expressions(unsafe):
    with pytest.raises(ValueError, match='Unsupported math'):
        draft_tex(f'# Title\n\n## Method\n${unsafe}$')


def test_standard_floor_math_is_supported_for_neighbor_count():
    tex = draft_tex('# Title\n\n## Method\n' + r'\(k=\max(10,\lfloor\sqrt{M}\rfloor)\)')
    assert r'\lfloor\sqrt{M}\rfloor' in tex


def test_math_keeps_code_and_currency_as_literal_prose():
    tex = draft_tex('# Title\n\n## Method\nCost is $5 versus $10. Code: `$x_i$`.')
    assert r'Cost is \$5 versus \$10.' in tex
    assert r'Code: \$x\_\allowbreak{}i\$.' in tex


def test_unclosed_display_math_fails_before_compilation():
    with pytest.raises(ValueError, match='Unclosed display math'):
        draft_tex('# Title\n\n## Method\n$$\nx_i = 1\n')


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
