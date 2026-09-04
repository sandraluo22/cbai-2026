"""Shared fixtures: a smoke-sized config, worlds, and a mock backend."""

from __future__ import annotations

import numpy as np
import pytest

from belief_feedback.agents.protocol import SteeringContext
from belief_feedback.config import Config, load_config
from belief_feedback.models.mock_backend import MockBackend
from belief_feedback.paths import REPO_ROOT
from belief_feedback.world.generator import build_ordinary_world, build_recycling_pair


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "smoke.yaml")


@pytest.fixture(scope="session")
def world(cfg):
    return build_ordinary_world(cfg, "w_unittest_0000", "exogenous_train", 0)


@pytest.fixture(scope="session")
def test_world(cfg):
    return build_ordinary_world(cfg, "w_unittest_t_0001", "endogenous_test", 1)


@pytest.fixture(scope="session")
def recycling_pair(cfg):
    return build_recycling_pair(cfg, 0)


@pytest.fixture()
def backend(cfg) -> MockBackend:
    return MockBackend(cfg)


@pytest.fixture()
def steer_ctx() -> SteeringContext:
    v = np.zeros(16)
    v[0] = 2.0
    return SteeringContext(vector=v, layer=2)
