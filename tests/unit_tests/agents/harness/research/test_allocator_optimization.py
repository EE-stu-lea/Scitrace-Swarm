import builtins
import itertools
import random
import pytest
from examples.scitrace_research.workflow.daber_experiment import optimal_allocation_from_values, _optimal_allocation_scalar

@pytest.mark.parametrize('seed', range(15))
def test_vectorized_allocation_matches_scalar_and_exhaustive(seed):
    rng=random.Random(seed)
    n=rng.randint(1,5); k=rng.randint(2,5); b=rng.randint(1,k)
    curves={f'q{i}':[0.]+[rng.randint(0,6)/6 for _ in range(k)] for i in range(n)}
    expected=_optimal_allocation_scalar(curves,average_budget=b,maximum_budget=k)
    actual=optimal_allocation_from_values(curves,average_budget=b,maximum_budget=k)
    assert actual==expected
    achieved=sum(curves[q][d] for q,d in actual.items())
    optimum=max(sum(curves[q][d] for q,d in zip(sorted(curves),choice))
                for choice in itertools.product(range(1,k+1),repeat=n) if sum(choice)==n*b)
    assert achieved==pytest.approx(optimum)

@pytest.mark.parametrize('budget',[1,2,5,10,12])
def test_ties_and_saturation_are_preserved(budget):
    curves={f'q{i}':[0.]*11 for i in range(8)}
    assert optimal_allocation_from_values(curves,average_budget=budget)==_optimal_allocation_scalar(curves,average_budget=budget)

@pytest.mark.parametrize('curve',[[0.,.1],[0.]+[float('nan')]*10,[0.]+[float('inf')]*10])
def test_invalid_curves_rejected(curve):
    with pytest.raises(ValueError):optimal_allocation_from_values({'q':curve},average_budget=5)

@pytest.mark.parametrize('budget',[0,-1,1.5,True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):optimal_allocation_from_values({'q':[0.]*11},average_budget=budget)

def test_optional_numpy_fallback(monkeypatch):
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name=='numpy':raise ImportError('optional dependency absent')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    curves={'a':[0.]+[.2]*10,'b':[i/10 for i in range(11)]}
    assert optimal_allocation_from_values(curves,average_budget=5)==_optimal_allocation_scalar(curves,average_budget=5)
