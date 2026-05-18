"""이전 import 경로를 위한 DQN 계열 호환 모듈.

실제 알고리즘 구현은 `ddareungi_rl.algorithms` 패키지에 있다.
수업용으로 알고리즘별 수식을 더 잘 보이게 하려고 파일을 분리했다.
"""

from ddareungi_rl.algorithms.common import (
    DQNConfig,
    evaluate_policy,
    evaluate_policy_with_trace,
    save_model,
)
from ddareungi_rl.algorithms.double_dqn import train_double_dqn
from ddareungi_rl.algorithms.dqn import train_dqn
from ddareungi_rl.algorithms.dueling_dqn import train_dueling_dqn

__all__ = [
    "DQNConfig",
    "evaluate_policy",
    "evaluate_policy_with_trace",
    "save_model",
    "train_double_dqn",
    "train_dqn",
    "train_dueling_dqn",
]
