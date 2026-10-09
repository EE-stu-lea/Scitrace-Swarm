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
from .workflow.budget_effect_figure import (
    FIGURE_PATH as EFFECT_FIGURE_PATH,
    DATA_PATH as EFFECT_DATA_PATH,
    MANIFEST_PATH as EFFECT_MANIFEST_PATH,
    load_budget_effects,
)
from .workflow.result_rows import _primary_results_table_rows, _adaptive_baseline_rows, _train_subsampling_rows
from .workflow.method_overview import (
    FIGURE_PATH as METHOD_FIGURE_PATH, MANIFEST_PATH as METHOD_MANIFEST_PATH, verify_method_overview,
)
from .workflow.query_diagnostics import (
    FIGURE_PATH as QUERY_FIGURE_PATH, DATA_PATH as QUERY_DATA_PATH, MANIFEST_PATH as QUERY_MANIFEST_PATH,
    build_query_diagnostics, verify_query_diagnostics,
)


UNICODE_LATEX = {
    '\u2013': '--',
    '\u2014': '---',
    '\u2192': r'\ensuremath{\rightarrow}',
    '\u00b1': r'\ensuremath{\pm}',
    '\u00d7': r'\ensuremath{\times}',
    '\u0394': r'\ensuremath{\Delta}',
    '\u03c3': r'\ensuremath{\sigma}',
    '\u03b8': r'\ensuremath{\theta}',
    '\u2082': r'\textsubscript{2}',
    '\u2084': r'\textsubscript{4}',
    '\u00a7': r'\S{}',
    '\u2026': r'\ldots{}',
}

# The bundled ICLR style is compiled with pdfLaTeX and has no Cyrillic glyphs.
# Transliterate only in the generated TeX; the hash-bound Agent draft and
# cached literature metadata retain the source spelling verbatim.
_CYRILLIC = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
_CYRILLIC_LATIN = (
    'a', 'b', 'v', 'g', 'd', 'e', 'yo', 'zh', 'z', 'i', 'y', 'k', 'l',
    'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'kh', 'ts', 'ch',
    'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya',
)
for _letter, _latin in zip(_CYRILLIC, _CYRILLIC_LATIN):
    UNICODE_LATEX[_letter] = _latin
    UNICODE_LATEX[_letter.upper()] = _latin.capitalize()

REFERENCE_ENTRY_RE = re.compile(
    r'^(?P<indent>\s*)(?P<marker>\d+\.|-)\s+'
    r'(?:\[\s*)?`(?P<evidence_id>evidence_[0-9a-f]+)`(?:\s*\])?\s*'
    r'(?:[\u2013\u2014-]+\s*)?',
    flags=re.MULTILINE,
)


_MATH_COMMANDS = set('''
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
sum prod max min arg log ln exp lim sup inf sin cos tan
hat widehat bar overline underline vec tilde widetilde
frac dfrac tfrac sqrt left right
le leq ge geq ne neq approx sim simeq equiv propto
in notin subset subseteq supset supseteq cup cap emptyset forall exists
times cdot pm mp div to rightarrow leftarrow mapsto infty partial nabla
ldots cdots vdots ddots dots mathbb mathcal mathrm mathit mathbf mathsf
text operatorname underbrace overbrace substack limits nolimits
displaystyle textstyle quad qquad lVert rVert lfloor rfloor lceil rceil
'''.split())
_MATH_SYMBOL_COMMANDS = set(',;:! {}|%_&#')


def _safe_math(expression: str) -> str:
    """Accept bounded scientific notation, never arbitrary TeX commands."""
    expression = expression.strip()
    if not expression or len(expression) > 2000 or '^^' in expression:
        raise ValueError('Unsupported math: empty, excessive, or unsafe expression')
    depth = 0
    position = 0
    while position < len(expression):
        character = expression[position]
        if character == '\\':
            command = re.match(r'\\([A-Za-z]+|.)', expression[position:], flags=re.DOTALL)
            if not command or (
                command.group(1) not in _MATH_COMMANDS
                and command.group(1) not in _MATH_SYMBOL_COMMANDS
            ):
                raise ValueError('Unsupported math command; use bounded scientific notation')
            position += len(command.group(0))
            continue
        if character == '{':
            depth += 1
            if depth > 16:
                raise ValueError('Unsupported math: excessive nesting')
        elif character == '}':
            depth -= 1
            if depth < 0:
                raise ValueError('Unsupported math: unbalanced braces')
        elif not (character.isascii() and (
            character.isalnum() or character in ' \t\n.,:;+-=*/^_()[]|<>!?'
        )):
            raise ValueError('Unsupported math character; use ASCII notation and allowed commands')
        position += 1
    if depth:
        raise ValueError('Unsupported math: unbalanced braces')
    return expression


def _math_inline(state, silent: bool) -> bool:
    position = state.pos
    if state.src.startswith('$$', position):
        raise ValueError('Display math must occupy its own paragraph')
    if state.src.startswith('\\(', position):
        opener, closer = '\\(', '\\)'
    elif state.src.startswith('$', position):
        opener = closer = '$'
        if position + 1 >= state.posMax or state.src[position + 1].isspace():
            return False
        # A bare currency amount must not consume a later equation as its
        # closing dollar. Use \(...\) for numeric-leading spaced formulas.
        if re.match(r'\d[\d,.]*(?:\s+(?=[A-Za-z])|(?=[.!?](?:\s|$)|$))', state.src[position + 1:state.posMax]):
            return False
    else:
        return False
    end = state.src.find(closer, position + len(opener), state.posMax)
    while end >= 0:
        backslashes = len(state.src[:end]) - len(state.src[:end].rstrip('\\'))
        if backslashes % 2 == 0:
            break
        end = state.src.find(closer, end + len(closer), state.posMax)
    if end < 0:
        return False
    if '`' in state.src[position + len(opener):end]:
        return False  # Never absorb a separate inline-code span.
    if opener == '$' and (state.src[end - 1].isspace() or (
        end + 1 < state.posMax and state.src[end + 1].isdigit()
    )):
        return False  # Currency prose such as "$5 versus $10" is not math.
    if not silent:
        token = state.push('math_inline', 'math', 0)
        token.content = state.src[position + len(opener):end]
    state.pos = end + len(closer)
    return True


def _math_block(state, start_line: int, end_line: int, silent: bool) -> bool:
    start = state.bMarks[start_line] + state.tShift[start_line]
    line = state.src[start:state.eMarks[start_line]].strip()
    if state.sCount[start_line] - state.blkIndent >= 4:
        return False
    if line.startswith('$$'):
        opener = closer = '$$'
    elif line.startswith('\\['):
        opener, closer = '\\[', '\\]'
    else:
        return False
    if silent:
        return True
    chunks = [line[len(opener):]]
    last_line = start_line
    while not chunks[-1].rstrip().endswith(closer):
        last_line += 1
        if last_line >= end_line:
            raise ValueError('Unclosed display math; use a standalone closing delimiter')
        start = state.bMarks[last_line] + state.tShift[last_line]
        chunks.append(state.src[start:state.eMarks[last_line]].strip())
    chunks[-1] = chunks[-1].rstrip()[:-len(closer)]
    token = state.push('math_block', 'math', 0)
    token.block = True
    token.content = '\n'.join(chunks).strip()
    token.map = [start_line, last_line + 1]
    state.line = last_line + 1
    return True


def _markdown_parser() -> MarkdownIt:
    parser = MarkdownIt('commonmark').enable('table')
    parser.inline.ruler.before('escape', 'scitrace_math', _math_inline)
    parser.block.ruler.before('fence', 'scitrace_math', _math_block,
                              {'alt': ['paragraph', 'reference', 'blockquote', 'list']})
    return parser


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
        elif token.type == 'math_inline':
            result.append(r'\(' + _safe_math(token.content) + r'\)')
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
            figure_path = token.attrGet('src')
            if figure_path not in (FIGURE_PATH, EFFECT_FIGURE_PATH, METHOD_FIGURE_PATH, QUERY_FIGURE_PATH):
                raise ValueError('Unsupported image; only verified result figures are supported')
            caption = token.content.strip()
            if not caption:
                raise ValueError('The result figure requires a caption')
            caption_tex = (
                inline(token.meta['caption_tokens'], citation_numbers=citation_numbers)
                if token.meta.get('caption_tokens') else escape(caption)
            )
            result.append(
                '\\begin{figure}[htbp]\\centering\n'
                '\\includegraphics[width=\\linewidth]{' + figure_path + '}\n'
                '\\caption{' + caption_tex + '}\\end{figure}\n'
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
    # Some Team handoffs prefix the accepted paper with an orchestration
    # status line. Keep it in the hash-bound draft, but omit this exact
    # non-manuscript preamble from the PDF.
    handoff_preamble = (
        'Delegation is in place. Here is the returned handoff from the draft stage.\n\n'
        '---\n\n'
    )
    if view.startswith(handoff_preamble):
        view = view[len(handoff_preamble):]
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


def _merge_figure_captions(tokens):
    """Move a following caption paragraph into its verified image token.

    Work on parsed Markdown so emphasis, wrapped lines, and citation tokens
    survive. Only a standalone image followed by an explicit caption is
    folded; ordinary result discussion must remain body prose.
    """
    i = 0
    while i + 5 < len(tokens):
        block = tokens[i:i + 6]
        if [token.type for token in block] != [
            'paragraph_open', 'inline', 'paragraph_close',
            'paragraph_open', 'inline', 'paragraph_close',
        ]:
            i += 1
            continue
        children = block[1].children or []
        if len(children) != 1 or children[0].type != 'image':
            i += 1
            continue
        image = children[0]
        path = image.attrGet('src')
        if path not in (FIGURE_PATH, EFFECT_FIGURE_PATH, METHOD_FIGURE_PATH, QUERY_FIGURE_PATH):
            i += 1
            continue
        caption_tokens = list(block[4].children or [])
        visible = ''.join(
            ' ' if token.type in ('softbreak', 'hardbreak') else token.content
            for token in caption_tokens
            if token.type in ('text', 'code_inline', 'softbreak', 'hardbreak')
        )
        label = re.match(
            r'^Figure\s+\d+(?:\s*\(' + re.escape(path) + r'\))?\s*[:.]\s*',
            visible, flags=re.IGNORECASE,
        )
        if not label and not (
            path == FIGURE_PATH and visible.startswith('Evidence recall across document budgets')
        ):
            i += 1
            continue
        # Remove the display number and optional local artifact path. LaTeX
        # supplies the number; every scientific word remains from the draft.
        remaining = label.end() if label else 0
        for token in caption_tokens:
            if token.type in ('text', 'code_inline') and remaining:
                consumed = min(len(token.content), remaining)
                token.content = token.content[consumed:]
                remaining -= consumed
            elif token.type in ('softbreak', 'hardbreak') and remaining:
                token.type = 'text'
                token.content = ''
                remaining -= 1
        # A caption commonly uses one outer emphasis span just to distinguish
        # it from body prose. Its words are now in the caption environment.
        caption_tokens = [token for token in caption_tokens if token.type != 'text' or token.content]
        if (caption_tokens and caption_tokens[0].type in ('em_open', 'strong_open')
                and caption_tokens[-1].type == caption_tokens[0].type.replace('_open', '_close')):
            depth = 0
            for index, token in enumerate(caption_tokens):
                depth += token.nesting
                if depth == 0:
                    # Separate emphasis spans at the two ends are not one
                    # outer wrapper; retain both to keep LaTeX balanced.
                    if index == len(caption_tokens) - 1:
                        caption_tokens = caption_tokens[1:-1]
                    break
        image.meta['caption_tokens'] = caption_tokens
        del tokens[i + 3:i + 6]
        i += 3
    return tokens


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
    tokens = _merge_figure_captions(_markdown_parser().parse(typesetting_draft))
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
                        # Keep main-text figures ahead of the bibliography
                        # without forcing the references onto a sparse page.
                        output.append('\\FloatBarrier\n\\section*{' + value + '}\n\\small\n')
                    elif heading_text.lower() == 'appendix':
                        output.append('\\clearpage\n\\normalsize\n\\appendix\n\\section{Additional details}\n')
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
        elif t.type == 'math_block':
            output.append('\\[\n' + _safe_math(t.content) + '\n\\]\n')
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
            # Short results tables stay on one page. The Agent's cells and
            # captions remain unchanged; only the LaTeX container differs.
            output.append('\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\\centering\n')
            output.append('\\begingroup\\setlength{\\tabcolsep}{3pt}\n')
            if columns >= 7:
                output.append('\\scriptsize\n')
            elif columns >= 5:
                output.append('\\footnotesize\n')
            output.append(r'\begin{tabular}{' + column_spec + '}\n')
            output.append('\\hline\n')
        elif t.type == 'table_close':
            output.append('\\hline\n\\end{tabular}\n\\endgroup\n\\end{minipage}\\par\\medskip\n')
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
            '\\usepackage[T1]{fontenc}\n\\usepackage{hyperref,array,graphicx,float,placeins,amsmath,amssymb}\n'
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
            query_diagnostics=build_query_diagnostics(run_dir / 'experiments' / 'route_b2_artifacts')
            if quality_cfg.get('require_query_outcome_table') else None,
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
    effect_hash = ''
    effect_data_hash = ''
    effect_sources: dict[str, str] = {}
    if f']({EFFECT_FIGURE_PATH})' in draft.read_text(encoding='utf-8'):
        effect = load_budget_effects(run_dir)
        if effect is None:
            raise ValueError('Budget-effect source artifacts are missing')
        effect_sources = effect['source_sha256']
        effect_path = run_dir / EFFECT_FIGURE_PATH
        effect_data_path = run_dir / EFFECT_DATA_PATH
        effect_manifest = json.loads((run_dir / EFFECT_MANIFEST_PATH).read_text(encoding='utf-8'))
        if json.loads(effect_data_path.read_text(encoding='utf-8')) != effect:
            raise ValueError('Budget-effect data does not match the experiment results')
        effect_hash = digest(effect_path)
        effect_data_hash = digest(effect_data_path)
        if (effect_manifest.get('path') != EFFECT_FIGURE_PATH
                or effect_manifest.get('data_path') != EFFECT_DATA_PATH
                or effect_manifest.get('figure_sha256') != effect_hash
                or effect_manifest.get('data_sha256') != effect_data_hash
                or effect_manifest.get('source_sha256') != effect_sources):
            raise ValueError('Budget-effect figure does not match its source manifest')
    # Every new graphic has an explicit allowlist and a source-bound verifier;
    # accepting an arbitrary Markdown image must never bypass provenance.
    additional_figures = {}
    extra_copy_paths = []
    draft_text = draft.read_text(encoding='utf-8')
    if f']({METHOD_FIGURE_PATH})' in draft_text:
        verify_method_overview(run_dir)
        extra_copy_paths.extend([METHOD_FIGURE_PATH, METHOD_MANIFEST_PATH])
    if f']({QUERY_FIGURE_PATH})' in draft_text:
        verify_query_diagnostics(run_dir)
        extra_copy_paths.extend([QUERY_FIGURE_PATH, QUERY_DATA_PATH, QUERY_MANIFEST_PATH, 'figures/query_risk.png'])
    additional_figures = {path: digest(run_dir / path) for path in extra_copy_paths}
    query_sources = (json.loads((run_dir / QUERY_MANIFEST_PATH).read_text())['source_sha256']
                     if QUERY_FIGURE_PATH in additional_figures else {})
    tex = draft_tex(draft_text)
    literature_path = run_dir / 'evidence' / 'literature.json'
    # The accepted Agent draft already contains complete human-readable
    # references with DOI and evidence ids. Keep the evidence hash in the
    # export manifest instead of duplicating the same bibliography as a long
    # machine-metadata appendix.
    json.loads(literature_path.read_text(encoding='utf-8'))
    paper_dir = paper_dir or run_dir / 'paper'
    paper_dir.mkdir(parents=True, exist_ok=True)
    for relative in extra_copy_paths:
        (paper_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / relative, paper_dir / relative)
    if figure_hash:
        (paper_dir / 'figures').mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / FIGURE_PATH, paper_dir / FIGURE_PATH)
    if effect_hash:
        (paper_dir / 'figures').mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / EFFECT_FIGURE_PATH, paper_dir / EFFECT_FIGURE_PATH)
        shutil.copy2(run_dir / EFFECT_DATA_PATH, paper_dir / EFFECT_DATA_PATH)
        shutil.copy2(run_dir / EFFECT_MANIFEST_PATH, paper_dir / EFFECT_MANIFEST_PATH)
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
        'budget_effect_figure_sha256': effect_hash,
        'budget_effect_data_sha256': effect_data_hash,
        'budget_effect_manifest_sha256': digest(run_dir / EFFECT_MANIFEST_PATH) if effect_hash else '',
        'budget_effect_source_sha256': effect_sources,
        'additional_figure_files_sha256': additional_figures,
        'query_source_sha256': query_sources,
        'source': 'Gate-accepted Agent draft.md; audit identifiers hidden during deterministic typesetting',
    }, indent=2), encoding='utf-8')
    return pdf


def verify(run_dir: Path, paper_dir: Path | None = None) -> None:
    paper = paper_dir or run_dir / 'paper'
    record = json.loads((paper / 'paper_manifest.json').read_text(encoding='utf-8'))
    for key, path in [('draft', run_dir / 'draft.md'), ('pdf', paper / 'paper.pdf'), ('tex', paper / 'paper.tex')]:
        if digest(path) != record[key + '_sha256']:
            raise ValueError(f'{key} changed after paper export; re-export and re-review the paper')
    if 'literature_sha256' in record and digest(run_dir / 'evidence' / 'literature.json') != record['literature_sha256']:
        raise ValueError('literature changed after paper export')
    allowed_extra = {METHOD_FIGURE_PATH, METHOD_MANIFEST_PATH, QUERY_FIGURE_PATH, QUERY_DATA_PATH,
                     QUERY_MANIFEST_PATH, 'figures/query_risk.png'}
    for relative, expected in record.get('additional_figure_files_sha256', {}).items():
        if relative not in allowed_extra:
            raise ValueError('Unexpected additional figure path in paper manifest')
        if digest(run_dir / relative) != expected or digest(paper / relative) != expected:
            raise ValueError('additional figure or provenance data changed after paper export')
    if QUERY_FIGURE_PATH in record.get('additional_figure_files_sha256', {}):
        verify_query_diagnostics(run_dir)
    if METHOD_FIGURE_PATH in record.get('additional_figure_files_sha256', {}):
        verify_method_overview(run_dir)
    if record.get('result_figure_sha256') and (
        digest(run_dir / FIGURE_PATH) != record['result_figure_sha256']
        or digest(paper / FIGURE_PATH) != record['result_figure_sha256']
    ):
        raise ValueError('result figure changed after paper export')
    if record.get('budget_effect_figure_sha256'):
        if (digest(run_dir / EFFECT_FIGURE_PATH) != record['budget_effect_figure_sha256']
                or digest(paper / EFFECT_FIGURE_PATH) != record['budget_effect_figure_sha256']
                or digest(run_dir / EFFECT_DATA_PATH) != record['budget_effect_data_sha256']
                or digest(paper / EFFECT_DATA_PATH) != record['budget_effect_data_sha256']
                or digest(run_dir / EFFECT_MANIFEST_PATH) != record['budget_effect_manifest_sha256']
                or digest(paper / EFFECT_MANIFEST_PATH) != record['budget_effect_manifest_sha256']):
            raise ValueError('budget-effect figure or data changed after paper export')
        for relative, expected in record['budget_effect_source_sha256'].items():
            if digest(run_dir / relative) != expected:
                raise ValueError(f'budget-effect source changed after paper export: {relative}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--paper-dir', default='')
    args = parser.parse_args()
    print(export(Path(args.run_dir), Path(args.paper_dir) if args.paper_dir else None))
