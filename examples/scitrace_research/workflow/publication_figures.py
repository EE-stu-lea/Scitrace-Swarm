"""Publication figures from audited paired reanalysis, with an explicit manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASETS = [("scifact", "SciFact", "#23689B"), ("nfcorpus", "NFCorpus", "#BD613C"),
            ("fiqa_lexical", "FiQA (lexical)", "#3D846C")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(report_path: Path, artifacts: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    data = json.loads(report_path.read_text())
    effects = data['effects']
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    outputs = {}

    def save(fig, name):
        for suffix in ('pdf', 'png'):
            target = output / f'{name}.{suffix}'
            fig.savefig(target, bbox_inches='tight', dpi=200,
                        metadata={'Creator': 'SciTrace audited publication figures',
                                  'CreationDate': None} if suffix == 'pdf' else None)
            outputs[target.name] = sha(target)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True, layout='constrained')
    for ax, (slug, label, color) in zip(axes, DATASETS):
        rows = sorted((r for r in effects if r['dataset'] == slug), key=lambda r:r['budget'])
        x = [r['budget'] for r in rows]
        y = [100*r['difference'] for r in rows]
        ax.axhline(0, color='#626A72', lw=.8)
        ax.axvline(5, color='#CAD0D6', ls=':', lw=.8)
        # Keep uncertainty behind the circular estimates. Between measured
        # budgets, the shading and connecting lines are visual guides only.
        ax.fill_between(x, [100*r['ci_family95_low'] for r in rows],
                        [100*r['ci_family95_high'] for r in rows],
                        color=color, alpha=.10, linewidth=0, zorder=1,
                        label='Family-adjusted interval')
        ax.fill_between(x, [100*r['ci95_low'] for r in rows],
                        [100*r['ci95_high'] for r in rows],
                        color=color, alpha=.20, linewidth=0, zorder=2,
                        label='Pointwise 95% interval')
        ax.plot(x, y, 'o-', color=color, lw=1.2, ms=6,
                markeredgecolor='white', markeredgewidth=.8, zorder=4,
                label='Observed difference')
        ax.set(title=label, xticks=x, xlabel='Mean document budget')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Recall change (percentage points)')
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=3, frameon=False, loc='outside lower center', fontsize=7)
    save(fig, 'effect_intervals')

    fig = plt.figure(figsize=(7.1, 4.05), layout='constrained')
    grid = fig.add_gridspec(2,3,height_ratios=[1.65,1.0])
    for j,(slug,label,color) in enumerate(DATASETS):
        ax=fig.add_subplot(grid[0,j])
        rows={r['allocated_depth']:r for r in data['allocation_histograms']
              if r['dataset']==slug and r['budget']==5}
        ax.bar(range(1,11), [100*rows.get(k,{}).get('fraction',0) for k in range(1,11)],
               color=color, width=.76)
        ax.axvline(5, color='#39434B', ls='--', lw=.8)
        ax.set(title=label, xticks=[1,3,5,7,10], xlabel='Assigned depth', ylim=(0,45))
        ax.yaxis.set_major_formatter(PercentFormatter())
        if j==0: ax.set_ylabel('Queries')
        ax.grid(axis='y',alpha=.18)
    ax=fig.add_subplot(grid[1,:])
    labels=[d[1] for d in DATASETS]
    rows=[next(r for r in effects if r['dataset']==d[0] and r['budget']==5) for d in DATASETS]
    left=np.zeros(3)
    for key,label,color in [('loss_fraction','Loss','#BA594D'),('tie_fraction','Tie','#D7DCE2'),
                            ('win_fraction','Win','#23689B')]:
        values=np.array([100*r[key] for r in rows])
        ax.barh(labels,values,left=left,color=color,label=label,height=.58)
        for i,(v,l) in enumerate(zip(values,left)):
            if v>=5: ax.text(l+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=7,
                            color='#29333C' if key=='tie_fraction' else 'white')
        left+=values
    ax.set_xlim(0,100);ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel('Query recall compared with fixed depth 5')
    ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.55),fontsize=7)
    save(fig, 'allocation')

    fig,axes=plt.subplots(1,3,figsize=(7.1,2.55),sharex=True,sharey=True,layout='constrained')
    sources={}
    for ax,(slug,label,color) in zip(axes,DATASETS):
        source=artifacts/slug/'results.json';sources[str(source)]=sha(source)
        raw=json.loads(source.read_text());ab=raw['feature_ablation']
        reference=ab['all_features']['recall']
        keys=['rank_overlap','jaccard_only','rbo_only']
        values=[100*(ab[k]['recall']-reference) for k in keys]
        ax.axvline(0,lw=.8,color='#626A72')
        ax.barh(['Rank disagreement','Top-10 Jaccard only','RBO only'],values,color=color,height=.55)
        ax.invert_yaxis();ax.set_title(label);ax.grid(axis='x',alpha=.18)
    fig.supxlabel('Recall change versus full features (percentage points)', fontsize=8)
    save(fig,'ablations')
    manifest={'analysis_sha256':sha(report_path), 'renderer_sha256':sha(Path(__file__)),
              'source_results_sha256':sources,'files_sha256':outputs,
              'interval_scope':'Pointwise conditional bootstrap and Bonferroni-adjusted family intervals; frozen original allocation, no new batch evaluation.',
              'allocation_budget':5, 'ablation_scope':'Stored point estimates only; no paired ablation outcomes retained.'}
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args.analysis,args.artifacts,args.output)

if __name__=='__main__': main()
