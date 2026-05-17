"""PyTorch 기반 DQN agent 구현.

기존 ``agents.dqn``은 신경망과 backprop을 순수 Python으로 직접 풀어 쓴
교육용 prototype이다. 이 모듈은 같은 toy MDP 위에서 PyTorch의
``nn.Module``과 optimizer를 사용해 더 일반적인 DQN 학습 구조를 보여준다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - torch 미설치 환경 안내용
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from ddareungi_rl.agents.dqn import Transition
from ddareungi_rl.envs import ToyDdareungiEnv


def require_torch() -> None:
    """PyTorch가 설치되어 있지 않으면 실행 방법을 포함한 예외를 발생시킨다."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch가 설치되어 있지 않습니다. "
            '`python -m pip install -e ".[torch]"` 또는 `python -m pip install torch`로 설치하세요.'
        ) from _TORCH_IMPORT_ERROR


@dataclass
class TorchDQNConfig:
    """PyTorch DQN 학습에 필요한 hyperparameter를 저장한다."""

    hidden_size: int = 64
    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 32
    replay_capacity: int = 5000
    min_replay_size: int = 100
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000


class TorchReplayBuffer:
    """PyTorch DQN에서 사용하는 고정 크기 transition replay buffer다."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        """저장 용량과 seed를 받아 replay buffer를 만든다."""
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.transitions: list[Transition] = []
        self.position = 0

    def __len__(self) -> int:
        """현재 buffer에 저장된 transition 개수를 반환한다."""
        return len(self.transitions)

    def add(self, transition: Transition) -> None:
        """transition 하나를 buffer에 추가한다."""
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        """무작위 mini-batch를 반환한다."""
        if batch_size > len(self.transitions):
            raise ValueError("batch_size cannot exceed replay buffer size")
        return self.rng.sample(self.transitions, batch_size)


if nn is not None:

    class TorchQNetwork(nn.Module):
        """state vector를 action별 Q-value로 변환하는 작은 MLP다."""

        def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
            """입력 차원, hidden 크기, action 개수로 network를 만든다."""
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
            """batch state tensor를 받아 action별 Q-value tensor를 반환한다."""
            return self.layers(state_batch)

else:

    class TorchQNetwork:  # pragma: no cover - torch 미설치 환경 안내용
        """PyTorch 미설치 환경에서 import 실패를 늦추기 위한 placeholder다."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """PyTorch가 필요하다는 예외를 발생시킨다."""
            require_torch()


class TorchDQNAgent:
    """PyTorch Q-network, target network, replay buffer를 묶은 DQN agent다."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: TorchDQNConfig | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        """환경 차원, config, seed, device를 받아 PyTorch DQN agent를 만든다."""
        require_torch()
        assert torch is not None

        if seed is not None:
            torch.manual_seed(seed)

        self.config = config or TorchDQNConfig()
        self.observation_size = observation_size
        self.action_size = action_size
        self.rng = random.Random(seed)
        self.device = torch.device(device or "cpu")
        self.online_network = TorchQNetwork(
            observation_size,
            self.config.hidden_size,
            action_size,
        ).to(self.device)
        self.target_network = TorchQNetwork(
            observation_size,
            self.config.hidden_size,
            action_size,
        ).to(self.device)
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay_buffer = TorchReplayBuffer(self.config.replay_capacity, seed=seed)
        self.update_count = 0

    def epsilon(self, step: int) -> float:
        """학습 step에 따른 epsilon 값을 선형 decay로 계산한다."""
        if step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end
        progress = step / max(1, self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: list[float], epsilon: float) -> int:
        """epsilon-greedy 방식으로 학습 action을 선택한다."""
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.action_size)
        return self.select_greedy_action(state)

    def select_greedy_action(self, state: list[float]) -> int:
        """Q-value가 가장 큰 action을 반환한다."""
        require_torch()
        assert torch is not None

        self.online_network.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, transition: Transition) -> None:
        """transition을 replay buffer에 저장한다."""
        self.replay_buffer.add(transition)

    def update(self) -> float | None:
        """replay buffer mini-batch로 DQN TD loss를 계산하고 online network를 갱신한다."""
        require_torch()
        assert torch is not None
        assert F is not None

        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        batch_size = min(self.config.batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(
            np.asarray([t.state for t in batch], dtype=np.float32),
            device=self.device,
        )
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.asarray([t.next_state for t in batch], dtype=np.float32),
            device=self.device,
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)

        # 현재 network가 예측한 Q(s, a)만 action index로 골라낸다.
        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DQN Bellman target: y = r + gamma * max_a' Q_target(s', a')
        # done=True인 마지막 step에서는 미래 보상 bootstrap 항을 제거한다.
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            target_q = rewards + self.config.gamma * next_q * (~dones).float()

        loss = F.mse_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """online network 파라미터를 target network로 복사한다."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: Path) -> None:
        """PyTorch DQN agent 설정과 online network 파라미터를 저장한다."""
        require_torch()
        assert torch is not None

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "network_state_dict": self.online_network.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> TorchDQNAgent:
        """저장된 파일에서 PyTorch DQN agent를 복원한다."""
        require_torch()
        assert torch is not None

        payload = torch.load(path, map_location=device or "cpu")
        config = TorchDQNConfig(**payload["config"])
        agent = cls(
            observation_size=int(payload["observation_size"]),
            action_size=int(payload["action_size"]),
            config=config,
            device=device,
        )
        agent.online_network.load_state_dict(payload["network_state_dict"])
        agent.update_target_network()
        return agent


class TorchDQNPolicy:
    """저장된 PyTorch DQN 모델을 greedy 평가 policy로 감싼다."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        """모델 경로에서 PyTorch DQN agent를 로드한다."""
        self.agent = TorchDQNAgent.load(model_path, device=device)

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """현재 환경 observation에서 Q-value가 가장 큰 action을 선택한다."""
        return self.agent.select_greedy_action(env.current_observation())
