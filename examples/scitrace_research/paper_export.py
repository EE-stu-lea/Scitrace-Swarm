"""Typeset the accepted Agent draft; never synthesize replacement prose."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import subprocess

from markdown_it import MarkdownIt

from .workflow.result_figures import FIGURE_PATH, budget_curve_data


UNICODE_LATEX = {
    '\u2013': '--',
    '\u2014': '---',
    '\u2192': r'\ensuremath{\rightarrow}',
    '\u00b1': r'\ensuremath{\pm}',
    '\u0394': r'\ensuremath{\Delta}',
    '\u03c3': r'\ensuremath{\sigma}',
    '\u03b8': r'\ensuremath{\theta}',
    '\u2082': r'\textsubscript{2}',
    '\u2084': r'\textsubscript{4}',
    '\u00a7': r'\S{}',
}

REFERENCE_ENTRY_RE = re.compile(
    r'^(?P<indent>\s*)(?P<marker>\d+\.|-)\s+'
    r'(?:\[\s*)?`(?P<evidence_id>evidence_[0-9a-f]+)`(?:\s*\])?\s*'
    r'(?:[\u2013\u2014-]+\s*)?',
    flags=re.MULTILINE,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(text: str) -> str:
    text = text.replace('\u2010', '-').replace('\u2011', '-').replace('\u2212', '-')
    mapping = dict(zip('\\&%$#_{}~^', [r'\textbackslash{}', r'\&', r'\%', r'\$', r'\#', r'\_', r'\{', r'\}', r'\textasciitilde{}', r'\textasciicircum{}']))
    mapping.update(UNICODE_LATEX)
    return ''.join(mapping.get(c, c) for c in text)


def escape_code(text: str) -> str:
    """Escape inline code and add invisible, safe line-break opportunities."""
    text = text.replace('\u2010', '-').replace('\u2011', '-').replace('\u2212', '-')
    mapping = dict(zip('\\&%$#_{}~^', [r'\textbackslash{}', r'\&', r'\%', r'\$', r'\#', r'\_', r'\{', r'\}', r'\textasciitilde{}', r'\textasciicircum{}']))
    mapping.update(UNICODE_LATEX)
    result = []
    alphanumeric_run = 0
    for character in text:
        result.append(mapping.get(character, character))
        if character.isalnum():
            alphanumeric_run += 1
            if alphanumeric_run == 12:
                result.append(r'\allowbreak{}')
                alphanumeric_run = 0
        else:
            alphanumeric_run = 0
            if character in '_-/.':
                result.append(r'\allowbreak{}')
    return ''.join(result)


def escape_table_text(text: str) -> str:
    """Keep the original cell text while allowing long numeric values to wrap."""
    result = []
    digit_run = 0
    alpha_run = 0
    for character in text:
        result.append(escape(character))
        if character.isdigit():
            digit_run += 1
            alpha_run = 0
            if digit_run == 6:
                result.append(r'\allowbreak{}')
                digit_run = 0
        elif character.isalpha():
            alpha_run += 1
            digit_run = 0
            if alpha_run == 12:
                result.append(r'\allowbreak{}')
                alpha_run = 0
        else:
            digit_run = 0
            alpha_run = 0
            if character in ',/.-_':
                result.append(r'\allowbreak{}')
    return ''.join(result)


def inline(
    tokens,
    *,
    in_table: bool = False,
    citation_numbers: dict[str, int] | None = None,
    omit_leading_evidence_id: bool = False,
) -> str:
    tokens = list(tokens or [])
    if (
        omit_leading_evidence_id
        and tokens
        and tokens[0].type == 'code_inline'
        and re.fullmatch(r'evidence_[0-9a-f]+', tokens[0].content)
    ):
        tokens = tokens[1:]
        if tokens and tokens[0].type == 'text':
            # The evidence id remains in draft.md and the audit records.  A
            # conventional bibliography should begin with the author, not an
            # internal provenance key.
            tokens[0].content = re.sub(r'^\s*[\u2013\u2014-]+\s*', '', tokens[0].content)
    if (
        omit_leading_evidence_id
        and tokens
        and tokens[-1].type == 'code_inline'
        and re.fullmatch(r'evidence_[0-9a-f]+', tokens[-1].content)
    ):
        tokens = tokens[:-1]
        if tokens and tokens[-1].type == 'text':
            tokens[-1].content = tokens[-1].content.rstrip()
    result = []
    for token in tokens:
        if token.type == 'text':
            result.append(escape_table_text(token.content) if in_table else escape(token.content))
        elif token.type == 'code_inline':
            citation_number = (citation_numbers or {}).get(token.content)
            if citation_number is not None:
                result.append(f'[{citation_number}]')
            else:
                # Preserve the exact identifier and safe break points without
                # switching the surrounding prose to a visibly different
                # typewriter face.
                result.append(escape_code(token.content))
        elif token.type == 'image':
            if token.attrGet('src') != FIGURE_PATH:
                raise ValueError('Unsupported image; only the verified result figure is supported')
            caption = token.content.strip()
            if not caption:
                raise ValueError('The result figure requires a caption')
            result.append(
                '\\begin{figure}[H]\\centering\n'
                '\\includegraphics[width=0.78\\linewidth]{' + FIGURE_PATH + '}\n'
                '\\caption{' + escape(caption) + '}\\end{figure}\n'
            )
        elif token.type in ('softbreak', 'hardbreak'):
            result.append(' ')
        elif token.type in ('strong_open', 'em_open'):
            result.append(r'\textbf{' if token.type == 'strong_open' else r'\emph{')
        elif token.type in ('strong_close', 'em_close'):
            result.append('}')
        elif token.type == 'link_open':
            result.append(r'\href{' + escape(token.attrGet('href') or '') + '}{')
        elif token.type == 'link_close':
            result.append('}')
        else:
            raise ValueError(f'Unsupported inline Markdown: {token.type}; revise explicitly before export')
    return ''.join(result)


def _abstract_marker(token) -> bool:
    """Recognize the Agent's standalone ``**Abstract**`` paragraph."""
    children = [
        child for child in (token.children or [])
        if child.type != 'text' or child.content.strip()
    ]
    types = [child.type for child in children]
    return (
        types == ['strong_open', 'text', 'strong_close']
        and children[1].content.strip().lower() == 'abstract'
    )


def _typesetting_view(draft: str) -> str:
    """Hide audit-only identifiers without changing the accepted Agent draft."""
    view = draft
    view = re.sub(
        r'Claim\s+`claim_[0-9a-f]+`\s+records this outcome',
        'This outcome is recorded',
        view,
        flags=re.IGNORECASE,
    )
    view = re.sub(
        r'Claim\s+`claim_[0-9a-f]+`\s*\(section:\s*results,\s*status:\s*`supported`\)\s+states',
        'The supported result claim states',
        view,
        flags=re.IGNORECASE,
    )
    view = re.sub(r'\brecorded as\s+`claim_[0-9a-f]+`', 'recorded', view, flags=re.IGNORECASE)
    view = re.sub(r'\bexperiment artifact,\s*`artifact_[0-9a-f]+`', 'experiment artifact', view, flags=re.IGNORECASE)
    view = re.sub(r'\s*\(`(?:rq|hyp|claim|artifact)_[0-9a-f]+`\)', '', view)
    view = re.sub(r'`(?:rq|hyp|claim|artifact)_[0-9a-f]+`', '', view)
    # Move an Agent-written caption directly after its figure into the actual
    # LaTeX figure caption, avoiding a duplicate caption paragraph. The
    # accepted Markdown draft remains byte-for-byte unchanged.
    view = re.sub(
        r'!\[[^\]\n]*\]\(' + re.escape(FIGURE_PATH) + r'\)\s*\n\s*'
        r'\*\*Figure\s+\d+:\s*(?P<caption>[^\n*]+)\*\*(?P<tail>[^\n]*)',
        lambda match: '![' + (match.group('caption') + ' ' + match.group('tail')).strip() + '](' + FIGURE_PATH + ')',
        view,
        flags=re.IGNORECASE,
    )
    # Some Agent drafts put the full caption in the plain paragraph directly
    # after the image. Use those exact words as the LaTeX caption instead of
    # duplicating a short alt-text caption and leaving the real one in prose.
    view = re.sub(
        r'!\[[^\]\n]*\]\(' + re.escape(FIGURE_PATH) + r'\)\s*\n\s*'
        r'(?P<caption>Evidence recall across document budgets[^\n]*)',
        lambda match: '![' + match.group('caption').strip() + '](' + FIGURE_PATH + ')',
        view,
        flags=re.IGNORECASE,
    )
    # Keep stable ids in draft.md/audit JSON, but present a conventional
    # bibliography in the PDF for both ordered and bullet-list references.
    view = REFERENCE_ENTRY_RE.sub(
        lambda match: (
            f'{match.group("indent")}'
            f'{"1." if match.group("marker") == "-" else match.group("marker")} '
        ),
        view,
    )
    return view


def _balanced_title(title: str) -> str:
    """Give long titles stable line breaks without hyphenating words."""
    if len(title) <= 55:
        return title
    words = title.split()
    if len(title) > 70 and len(words) >= 3:
        line_count = 4 if len(title) > 100 and len(words) >= 4 else 3
        candidates = []
        for cuts in combinations(range(1, len(words)), line_count - 1):
            boundaries = (0, *cuts, len(words))
            lines = [
                ' '.join(words[boundaries[index]:boundaries[index + 1]])
                for index in range(line_count)
            ]
            lengths = [len(line) for line in lines]
            candidates.append(((max(lengths), max(lengths) - min(lengths)), lines))
        lines = min(candidates, key=lambda item: item[0])[1]
        return r'\\'.join(lines)
    split = min(
        range(1, len(words)),
        key=lambda index: abs(
            len(' '.join(words[:index])) - len(' '.join(words[index:]))
        ),
    )
    return ' '.join(words[:split]) + r'\\' + ' '.join(words[split:])


def draft_tex(draft: str) -> str:
    # Agent drafts may place the provenance key before or after each human
    # reference. Accept both positions and reserve citation numbers in list
    # order; keep the key in the immutable draft and audit files.
    references = re.search(
        r'(?ms)^#{1,6}\s+References\s*$\n(?P<body>.*?)(?=^#{1,6}\s+|\Z)', draft
    )
    reference_ids = (
        re.findall(r'(?m)^\s*(?:\d+\.|-)\s+.*?`(evidence_[0-9a-f]+)`', references.group('body'))
        if references else []
    )
    typesetting_draft = _typesetting_view(draft)
    tokens = MarkdownIt('commonmark').enable('table').parse(typesetting_draft)
    citation_numbers = {evidence_id: index for index, evidence_id in enumerate(reference_ids, 1)}
    output, title = [], None
    title_level = None
    abstract = False
    skip_level = None
    row_is_header = False
    in_table = False
    in_references = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == 'heading_open':
            if abstract:
                output.append('\\end{abstract}\n')
                abstract = False
            level = int(t.tag[1:])
            heading = tokens[i + 1]
            heading_text = re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s+', '', heading.content).strip()
            in_references = heading_text.lower() == 'references'
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if heading.content.strip() in ('Structured References', 'Team Handoff'):
                skip_level = level
            if skip_level is None:
                # The Agent commonly numbers Markdown headings itself (for
                # example, ``## 3 Method``). LaTeX adds section numbers, so
                # remove only that leading display number during typesetting.
                value = escape(heading_text)
                # Some Team models shift the entire hierarchy down one level,
                # emitting ``## Paper title`` followed by ``### Abstract`` and
                # ``### 1. Introduction``. Accept the first non-metadata H1/H2
                # as the title and shift later section commands accordingly.
                if (
                    title is None
                    and level in (1, 2)
                    and heading_text.lower() not in ('abstract', 'structured references', 'team handoff')
                ):
                    title = value
                    title_level = level
                elif heading.content.strip().lower() == 'abstract':
                    output.append('\\begin{abstract}\n')
                    abstract = True
                else:
                    effective_level = max(1, level - ((title_level or 1) - 1))
                    command = 'section' if effective_level <= 2 else 'subsection' if effective_level == 3 else 'subsubsection'
                    if heading_text.lower() == 'references':
                        output.append('\\section*{' + value + '}\n\\small\n')
                    else:
                        output.append('\\' + command + '{' + value + '}\n')
            i += 3
            continue
        if skip_level is not None:
            i += 1
            continue
        if t.type == 'inline':
            if _abstract_marker(t):
                if abstract:
                    raise ValueError('Duplicate Abstract marker')
                output.append('\\begin{abstract}\n')
                abstract = True
            else:
                output.append(inline(
                    t.children,
                    in_table=in_table,
                    citation_numbers=citation_numbers,
                    omit_leading_evidence_id=in_references,
                ))
        elif t.type == 'paragraph_close':
            output.append('\n\n')
        elif t.type in ('bullet_list_open', 'ordered_list_open'):
            output.append('\\begin{' + ('itemize' if t.type == 'bullet_list_open' else 'enumerate') + '}\n')
            if in_references:
                output.append('\\setlength{\\itemsep}{0pt}\\setlength{\\parsep}{0pt}\n')
        elif t.type in ('bullet_list_close', 'ordered_list_close'):
            output.append('\\end{' + ('itemize' if t.type == 'bullet_list_close' else 'enumerate') + '}\n')
        elif t.type == 'list_item_open':
            output.append(r'\item ')
        elif t.type in ('blockquote_open', 'blockquote_close'):
            output.append('\\begin{quote}\n' if t.type == 'blockquote_open' else '\\end{quote}\n')
        elif t.type == 'table_open':
            in_table = True
            j = i + 1
            columns = 0
            while tokens[j].type != 'tr_close':
                columns += tokens[j].type == 'th_open'
                j += 1
            widths = ([.13, .14, .14, .12, .31]
                      if columns == 5 else
                      [.12, .12, .12, .12, .25, .15]
                      if columns == 6 else
                      [.12, .12, .12, .08, .22, .08, .11]
                      if columns == 7 else [.85 / columns] * columns)
            column_spec = ''.join(
                r'>{\raggedright\arraybackslash}p{'
                + str(round(width, 4))
                + r'\linewidth}'
                for width in widths
            )
            output.append('\\begingroup\\setlength{\\tabcolsep}{3pt}\n')
            if columns >= 7:
                output.append('\\scriptsize\n')
            elif columns >= 5:
                output.append('\\footnotesize\n')
            output.append(r'\begin{longtable}{' + column_spec + '}\n')
            output.append('\\hline\n')
        elif t.type == 'table_close':
            output.append('\\hline\n\\end{longtable}\n\\endgroup\n')
            in_table = False
        elif t.type == 'tr_open':
            row_is_header = False
        elif t.type == 'th_open':
            row_is_header = True
            output.append(r'\textbf{')
        elif t.type == 'td_open':
            pass
        elif t.type == 'th_close':
            output.append('}')
            output.append(' & ' if tokens[i + 1].type != 'tr_close' else '')
        elif t.type == 'td_close':
            output.append(' & ' if tokens[i + 1].type != 'tr_close' else '')
        elif t.type == 'tr_close':
            output.append(' \\\\\n')
            if row_is_header:
                output.append('\\hline\n')
        elif t.type in ('fence', 'code_block', 'html_block', 'html_inline'):
            raise ValueError(f'Unsupported body block: {t.type}; no content was silently discarded')
        elif t.type == 'hr':
            output.append('\n\\medskip\n')
        elif t.type not in ('paragraph_open', 'list_item_close', 'thead_open', 'thead_close', 'tbody_open', 'tbody_close'):
            raise ValueError(f'Unsupported Markdown: {t.type}')
        i += 1
    if not title:
        raise ValueError('Agent draft must begin with a level-one or level-two paper title')
    if abstract:
        output.append('\\end{abstract}\n')
    return ('\\documentclass{article}\n\\usepackage{iclr2026_conference,times}\n'
            '\\usepackage[T1]{fontenc}\n\\usepackage{hyperref,longtable,array,graphicx,float}\n'
            '\\hypersetup{hidelinks}\n'
            '\\title{' + _balanced_title(title) + '}\n'
            '\\begin{document}\n\\maketitle\n' + ''.join(output) + '\n\\end{document}\n')


def export(run_dir: Path, paper_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    gate = json.loads((run_dir / 'validation_report.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'success' or not gate.get('ok'):
        raise ValueError('A successful gate-approved run is required')
    draft = run_dir / 'draft.md'
    accepted_hash = str(gate.get('draft_sha256') or '')
    if not accepted_hash or accepted_hash != str(manifest.get('draft_sha256') or '') or digest(draft) != accepted_hash:
        raise ValueError('Agent draft is not bound to the successful validation; rerun the workflow')
    if manifest.get('paper_quality_enabled'):
        quality = json.loads((run_dir / 'paper_quality_report.json').read_text(encoding='utf-8'))
        if not quality.get('ok'):
            raise ValueError('The generated paper did not pass the paper quality gate')
        # A stored pass can become stale when a factual gate is strengthened.
        # Replay the current deterministic checks on the bound Agent draft;
        # historical successful run manifests remain unchanged.
        from ruamel.yaml import YAML
        from .workflow.paper_quality import audit_draft
        from .workflow.pipeline import (
            _adaptive_baseline_rows, _primary_results_table_rows, _train_subsampling_rows,
        )
        config = YAML(typ='safe').load((run_dir / 'run_config.yaml').read_text(encoding='utf-8'))
        quality_cfg = config.get('paper_quality') or {}
        results = json.loads((run_dir / 'experiments' / 'results.json').read_text(encoding='utf-8'))
        literature = json.loads((run_dir / 'evidence' / 'literature.json').read_text(encoding='utf-8'))
        replay = audit_draft(
            draft.read_text(encoding='utf-8'), evidence=literature['evidence'], config=quality_cfg,
            table_rows=_primary_results_table_rows(results) if quality_cfg.get('require_results_table') else None,
            adaptive_rows=_adaptive_baseline_rows(results) if quality_cfg.get('require_adaptive_baseline_table') else None,
            stability_rows=_train_subsampling_rows(results) if quality_cfg.get('require_train_subsampling_summary') else None,
            results=results,
        )
        if not replay['ok']:
            raise ValueError('The Agent draft fails the current paper quality gate; request an Agent revision')
    figure_hash = ''
    if f']({FIGURE_PATH})' in draft.read_text(encoding='utf-8'):
        figure_record = json.loads((run_dir / 'figures' / 'figure_manifest.json').read_text(encoding='utf-8'))
        figure = run_dir / FIGURE_PATH
        if figure_record.get('path') != FIGURE_PATH or digest(figure) != figure_record.get('figure_sha256'):
            raise ValueError('Result figure does not match its manifest')
        plotted = budget_curve_data(json.loads((run_dir / 'experiments' / 'results.json').read_text(encoding='utf-8')))
        source_hash = hashlib.sha256(json.dumps(plotted, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        if source_hash != figure_record.get('source_results_sha256'):
            raise ValueError('Result figure does not match the experiment result')
        figure_hash = digest(figure)
    tex = draft_tex(draft.read_text(encoding='utf-8'))
    literature_path = run_dir / 'evidence' / 'literature.json'
    # The accepted Agent draft already contains complete human-readable
    # references with DOI and evidence ids. Keep the evidence hash in the
    # export manifest instead of duplicating the same bibliography as a long
    # machine-metadata appendix.
    json.loads(literature_path.read_text(encoding='utf-8'))
    paper_dir = paper_dir or run_dir / 'paper'
    paper_dir.mkdir(parents=True, exist_ok=True)
    if figure_hash:
        (paper_dir / 'figures').mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / FIGURE_PATH, paper_dir / FIGURE_PATH)
    templates = Path(__file__).parent / 'submission_assets' / 'paper'
    shutil.copy2(templates / 'iclr2026_conference.sty', paper_dir)
    (paper_dir / 'paper.tex').write_text(tex, encoding='utf-8')
    for _ in range(2):
        process = subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], cwd=paper_dir, capture_output=True, timeout=120)
        (paper_dir / 'compile.log').write_bytes(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f'PDF compilation failed: {paper_dir / "compile.log"}')
    pdf = paper_dir / 'paper.pdf'
    (paper_dir / 'paper_manifest.json').write_text(json.dumps({
        'draft_sha256': digest(draft), 'pdf_sha256': digest(pdf),
        'tex_sha256': digest(paper_dir / 'paper.tex'),
        'literature_sha256': digest(literature_path),
        'result_figure_sha256': figure_hash,
        'source': 'Gate-accepted Agent draft.md; audit identifiers hidden during deterministic typesetting',
    }, indent=2), encoding='utf-8')
    return pdf


def verify(run_dir: Path) -> None:
    paper = run_dir / 'paper'
    record = json.loads((paper / 'paper_manifest.json').read_text(encoding='utf-8'))
    for key, path in [('draft', run_dir / 'draft.md'), ('pdf', paper / 'paper.pdf'), ('tex', paper / 'paper.tex')]:
        if digest(path) != record[key + '_sha256']:
            raise ValueError(f'{key} changed after paper export; re-export and re-review the paper')
    if 'literature_sha256' in record and digest(run_dir / 'evidence' / 'literature.json') != record['literature_sha256']:
        raise ValueError('literature changed after paper export')
    if record.get('result_figure_sha256') and (
        digest(run_dir / FIGURE_PATH) != record['result_figure_sha256']
        or digest(paper / FIGURE_PATH) != record['result_figure_sha256']
    ):
        raise ValueError('result figure changed after paper export')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(export(Path(args.run_dir)))
