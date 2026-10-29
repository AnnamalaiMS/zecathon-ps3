import numpy as np
from src.buffer_policy import buffer_from_forecast, critical_quantile, policy_cost


def test_critical_quantile_equal_costs_is_075():
    assert abs(critical_quantile(1, 1, 3) - 0.75) < 1e-12


def test_quantile_buffer_minimises_expected_cost_numerically():
    rng = np.random.default_rng(1)
    outflow = np.maximum(rng.normal(1000, 400, 200000), 0)
    grid = np.linspace(0, 3000, 301)
    costs = [policy_cost(b, outflow, 1, 1, 3).mean() for b in grid]
    best = grid[int(np.argmin(costs))]
    assert abs(best - np.quantile(outflow, 0.75)) < 30


def test_buffer_from_forecast_nonnegative():
    b = buffer_from_forecast(np.array([500.0]), np.array([300.0]), np.array([700.0]), 0.75)
    assert b[0] == 0.0
