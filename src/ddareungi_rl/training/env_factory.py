"""학습/평가 코드가 사용할 ToyDdareungiEnv 생성 helper."""

from __future__ import annotations

from pathlib import Path

from ddareungi_rl.data import load_profile_config
from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.envs.toy_ddareungi_env import RenderMode


def make_env(
    profile_path: Path | None = None,
    render_mode: RenderMode | None = None,
    seed: int | None = None,
) -> ToyDdareungiEnv:
    """선택적 real-data profile을 반영한 ToyDdareungiEnv를 만든다."""
    config = load_profile_config(profile_path) if profile_path is not None else None
    return ToyDdareungiEnv(config=config, render_mode=render_mode, seed=seed)
