"""Offline fault injection against the actual rail and persisted SEG."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import random

from jiuwenswarm.agents.harness.research.provenance.models import LiteratureEvidence, ExperimentArtifact, Claim, ClaimStatus
from jiuwenswarm.agents.harness.research.provenance.store import ScientificProvenanceStore
from jiuwenswarm.agents.harness.research.rails import ScientificConsistencyGateRail, StageAwareContextRail


def evaluate(output: Path) -> dict:
    rows = []
    with tempfile.TemporaryDirectory(prefix='scitrace-gate-') as directory:
        store = ScientificProvenanceStore(Path(directory))
        evidence = LiteratureEvidence(title='Synthetic control', authors=['Fixture'], year=2026, source_url='https://example.org/control')
        artifact = ExperimentArtifact(experiment_id='control', file_path='control.json', sha256='synthetic', command='controlled fixture', seed=0, dataset='fault-injection', metric_name='accuracy', metric_value=0.8)
        claim = Claim(text='Controlled measurement.', evidence_ids=[evidence.id], artifact_ids=[artifact.id], section='results', status=ClaimStatus.SUPPORTED)
        for node in (evidence, artifact, claim):
            store.upsert_node(node)
        gate = ScientificConsistencyGateRail(enabled=True, output_root=directory)
        base = {'draft_text':'Controlled measurement.', 'citations':[evidence.id], 'claims':[claim.id], 'reported_metrics':[{'artifact_id':artifact.id,'metric_name':'accuracy','claimed_value':0.8}]}
        cases = [('clean', copy.deepcopy(base), False)]
        for value in [0.0, 0.1, 0.5, 0.79, 0.81, 0.9, 1.0]:
            payload = copy.deepcopy(base)
            payload['reported_metrics'][0]['claimed_value'] = value
            cases.append((f'metric_{value}', payload, True))
        for key in ('citations', 'claims'):
            payload = copy.deepcopy(base); payload[key] = ['nonexistent']
            cases.append((key + '_missing', payload, True))
        # Negative control: mismatches only in prose may escape structured checks.
        payload = copy.deepcopy(base); payload['draft_text'] = 'The accuracy is 0.99.'
        cases.append(('prose_only_numeric_drift', payload, True))
        for name, payload, faulty in cases:
            result = gate.check_payload(payload, session_id='')
            rows.append({'case':name, 'faulty':faulty, 'no_gate_accepts':True, 'gate_accepts':result['ok'], 'errors':result['errors'], 'input':payload})
        context_rows = []
        for index in range(16):
            store.upsert_node(LiteratureEvidence(title=f'Synthetic distractor {index}', authors=['Fixture'], year=2026, source_url=f'https://example.org/{index}', summary='Background context with no experiment result.', reliability_score=0.2))
        context_rail = StageAwareContextRail(enabled=True, output_root=directory)
        required = {artifact.id, claim.id}
        for budget in (64, 96, 128, 256):
            bundle = context_rail.build_stage_context('', stage='writing', token_budget=budget)
            candidates = context_rail._rank_items(context_rail.graph(''), 'writing')
            for seed in (11, 23, 42, 77, 101):
                shuffled = list(candidates); random.Random(seed).shuffle(shuffled)
                selected, used = [], 0
                for item in shuffled:
                    if used + item.estimated_tokens <= budget:
                        selected.append(item.node_id); used += item.estimated_tokens
                context_rows.append({'budget':budget,'seed':seed,
                    'stage_aware_coverage':len(required.intersection(bundle['selected_node_ids'])) / len(required),
                    'random_order_coverage':len(required.intersection(selected)) / len(required),
                    'stage_aware_estimated_tokens':bundle['used_tokens'], 'random_order_estimated_tokens':used})
    faults = [row for row in rows if row['faulty']]
    result = {'experiment':'actual-rail-controlled-fault-injection', 'cases':rows,
              'context_budget_ablation':context_rows,
              'fault_count':len(faults), 'gate_detected':sum(not row['gate_accepts'] for row in faults),
              'no_gate_detected':0,
              'false_positives':sum(not row['gate_accepts'] for row in rows if not row['faulty']),
              'limitations':['Synthetic mechanism evaluation, not real-world paper accuracy.', 'No-gate baseline deliberately accepts all outputs.', 'Prose-only errors are included to expose structured-check blind spots.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='runs/gate_ablation/results.json')
    args = parser.parse_args()
    result = evaluate(Path(args.output))
    print(json.dumps({key:value for key,value in result.items() if key != 'cases'}, indent=2))
