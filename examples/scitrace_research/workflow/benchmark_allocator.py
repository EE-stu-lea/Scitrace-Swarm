"""Synthetic solver microbenchmark; does not rerun retrieval or measure LLM cost."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
import numpy as np
from .daber_experiment import _optimal_allocation_scalar, optimal_allocation_from_values


def curves(n,seed):
    values=np.cumsum(np.random.default_rng(seed).random((n,10)),axis=1)
    values/=values[:,-1:]
    return {f'q{i:04d}':[0.]+row.tolist() for i,row in enumerate(values)}


def benchmark(output: Path, repeats=3):
    rows=[]
    # Warm array machinery; excluded from measured solver trials.
    optimal_allocation_from_values(curves(5,0),average_budget=5)
    for n in (300,323,648):
        inputs=curves(n,20261008+n);times={'scalar':[],'vectorized':[]};reference=None
        for repeat in range(repeats):
            order=[('scalar',_optimal_allocation_scalar),('vectorized',optimal_allocation_from_values)]
            if repeat%2:order.reverse()
            for name,solver in order:
                started=time.perf_counter();allocation=solver(inputs,average_budget=5)
                times[name].append(time.perf_counter()-started)
                if reference is None:reference=allocation
                if allocation!=reference:raise AssertionError('allocation/tie-breaking mismatch')
        rows.append({'queries':n,'mean_budget':5,'depth_cap':10,'allocation_equal':True,
                     'seconds':times,'median_scalar_seconds':statistics.median(times['scalar']),
                     'median_vectorized_seconds':statistics.median(times['vectorized']),
                     'speedup':statistics.median(times['scalar'])/statistics.median(times['vectorized'])})
    memory={}
    inputs=curves(300,20261308)
    for name,solver in [('scalar',_optimal_allocation_scalar),('vectorized',optimal_allocation_from_values)]:
        tracemalloc.start();solver(inputs,average_budget=5);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        memory[name+'_tracemalloc_peak_bytes']=peak
    memory['queries']=300
    memory['scope']='Python/NumPy allocations visible to tracemalloc; not process RSS or full retrieval memory.'
    data={'scope':'Synthetic monotone recall curves, exact same allocations. Solver only, no end-to-end or LLM measurement.',
          'seed':20261008,'repeats':repeats,'python':platform.python_version(),'numpy':np.__version__,
          'source_sha256':hashlib.sha256(Path(__file__).with_name('daber_experiment.py').read_bytes()).hexdigest(),
          'rows':rows,'memory':memory}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2)+'\n')
    lines=['# Exact allocator microbenchmark','',data['scope'],'',
           '| Queries | Scalar median (s) | Vectorized median (s) | Speedup | Identical depths |',
           '|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['queries']} | {r['median_scalar_seconds']:.4f} | {r['median_vectorized_seconds']:.4f} | {r['speedup']:.1f}x | Yes |")
    lines+=['',f"For 300 queries, tracemalloc peaks were {memory['scalar_tracemalloc_peak_bytes']:,} and {memory['vectorized_tracemalloc_peak_bytes']:,} bytes.",
            memory['scope'],'','Alternating order, three repeats, fixed generated inputs. Timing varies by host; saved retrieval metrics are not replaced.']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();benchmark(args.output)
