import math

import pytest
import torch

from training.grpo import clipped_token_loss, group_advantages


def test_group_advantages_use_population_std_and_handle_constant_rewards():
    expected = 1 / (math.sqrt(2 / 3) + 1e-4)
    assert group_advantages([0, 1, 2]) == pytest.approx([-expected, 0, expected])
    assert group_advantages([0, 0, 0, 0]) == [0.0] * 4
    assert group_advantages([1.09] * 4) == [0.0] * 4


@pytest.mark.parametrize("returns", [[1], [float("nan"), 0], [0, float("inf")]])
def test_invalid_group_rewards_rejected(returns):
    with pytest.raises(ValueError):
        group_advantages(returns)


@pytest.mark.parametrize("ratio, advantage, expected_loss, expected_gradient", [
    (1.5, 1, -1.2, 0.0),
    (0.5, -1, 0.8, 0.0),
    (0.5, 1, -0.5, -0.5),
    (1.5, -1, 1.5, 1.5),
])
def test_clipping_direction_and_detached_behavior(ratio, advantage, expected_loss, expected_gradient):
    old = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    current = torch.tensor(-2.0 + math.log(ratio), dtype=torch.float64, requires_grad=True)
    loss = clipped_token_loss(current, old, advantage)
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert current.grad.item() == pytest.approx(expected_gradient)
    assert old.grad is None
