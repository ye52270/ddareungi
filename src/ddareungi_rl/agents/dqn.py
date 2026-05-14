"""V1 순수 Python DQN agent 구현.

PyTorch 같은 외부 딥러닝 의존성 없이 toy MDP에서 실행 가능한 작은
one-hidden-layer Q-network를 제공한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from ddareungi_rl.envs import ToyDdareungiEnv


@dataclass
class DQNConfig:
    """DQN 학습에 필요한 hyperparameter를 저장한다."""

    hidden_size: int = 32
    gamma: float = 0.95
    learning_rate: float = 0.01
    batch_size: int = 32
    replay_capacity: int = 5000
    min_replay_size: int = 100
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    gradient_clip: float = 10.0


@dataclass
class Transition:
    """Replay buffer에 저장되는 transition 하나를 표현한다."""

    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """고정 크기 transition replay buffer다."""

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


class QNetwork:
    """작은 one-hidden-layer ReLU Q-network다."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int | None = None,
    ) -> None:
        """network shape와 seed를 받아 가중치를 초기화한다."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = random.Random(seed)
        scale1 = 1.0 / max(1, input_size) ** 0.5
        scale2 = 1.0 / max(1, hidden_size) ** 0.5
        self.w1 = [
            [self.rng.uniform(-scale1, scale1) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [
            [self.rng.uniform(-scale2, scale2) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.b2 = [0.0 for _ in range(output_size)]

    def predict(self, state: list[float]) -> list[float]:
        """state 하나에 대한 action별 Q-value를 반환한다."""
        _, hidden = self._hidden(state)
        return [
            sum(weight * value for weight, value in zip(row, hidden)) + bias
            for row, bias in zip(self.w2, self.b2)
        ]

    def train_one(
        self,
        state: list[float],
        action: int,
        target: float,
        learning_rate: float,
        gradient_clip: float,
    ) -> float:
        """transition 하나의 TD target으로 network를 한 번 갱신하고 loss를 반환한다."""
        pre_activation, hidden = self._hidden(state)
        q_values = [
            sum(weight * value for weight, value in zip(row, hidden)) + bias
            for row, bias in zip(self.w2, self.b2)
        ]
        error = self._clip(q_values[action] - target, gradient_clip)
        loss = 0.5 * (q_values[action] - target) ** 2
        old_w2_action = self.w2[action].copy()

        for hidden_id, hidden_value in enumerate(hidden):
            self.w2[action][hidden_id] -= learning_rate * error * hidden_value
        self.b2[action] -= learning_rate * error

        for hidden_id, old_weight in enumerate(old_w2_action):
            if pre_activation[hidden_id] <= 0:
                continue
            hidden_grad = self._clip(error * old_weight, gradient_clip)
            for input_id, input_value in enumerate(state):
                self.w1[hidden_id][input_id] -= learning_rate * hidden_grad * input_value
            self.b1[hidden_id] -= learning_rate * hidden_grad

        return loss

    def copy_from(self, other: QNetwork) -> None:
        """다른 QNetwork의 파라미터를 복사한다."""
        self.w1 = [row.copy() for row in other.w1]
        self.b1 = other.b1.copy()
        self.w2 = [row.copy() for row in other.w2]
        self.b2 = other.b2.copy()

    def to_dict(self) -> dict[str, Any]:
        """network 파라미터를 JSON 저장 가능한 dict로 변환한다."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QNetwork:
        """dict에 저장된 파라미터로 QNetwork를 복원한다."""
        network = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
        )
        network.w1 = [[float(value) for value in row] for row in data["w1"]]
        network.b1 = [float(value) for value in data["b1"]]
        network.w2 = [[float(value) for value in row] for row in data["w2"]]
        network.b2 = [float(value) for value in data["b2"]]
        return network

    def _hidden(self, state: list[float]) -> tuple[list[float], list[float]]:
        """state를 hidden pre-activation과 ReLU activation으로 변환한다."""
        pre_activation = [
            sum(weight * value for weight, value in zip(row, state)) + bias
            for row, bias in zip(self.w1, self.b1)
        ]
        hidden = [max(0.0, value) for value in pre_activation]
        return pre_activation, hidden

    def _clip(self, value: float, limit: float) -> float:
        """gradient 폭주를 줄이기 위해 값을 범위 안으로 자른다."""
        return max(-limit, min(limit, value))


class DQNAgent:
    """Q-network, target network, replay buffer를 묶은 DQN agent다."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """환경 차원, config, seed를 받아 DQN agent를 만든다."""
        self.config = config or DQNConfig()
        self.observation_size = observation_size
        self.action_size = action_size
        self.rng = random.Random(seed)
        self.online_network = QNetwork(
            observation_size,
            self.config.hidden_size,
            action_size,
            seed=seed,
        )
        self.target_network = QNetwork(
            observation_size,
            self.config.hidden_size,
            action_size,
            seed=None if seed is None else seed + 1,
        )
        self.target_network.copy_from(self.online_network)
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, seed=seed)
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
        q_values = self.online_network.predict(state)
        return max(range(self.action_size), key=lambda action: q_values[action])

    def remember(self, transition: Transition) -> None:
        """transition을 replay buffer에 저장한다."""
        self.replay_buffer.add(transition)

    def update(self) -> float | None:
        """replay buffer에서 batch를 뽑아 DQN TD loss로 online network를 갱신한다."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        batch_size = min(self.config.batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample(batch_size)
        losses = []
        for transition in batch:
            target = self._target_value(transition)
            losses.append(
                self.online_network.train_one(
                    transition.state,
                    transition.action,
                    target,
                    self.config.learning_rate,
                    self.config.gradient_clip,
                )
            )
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.update_target_network()
        return sum(losses) / len(losses)

    def update_target_network(self) -> None:
        """online network 파라미터를 target network로 복사한다."""
        self.target_network.copy_from(self.online_network)

    def save(self, path: Path) -> None:
        """DQN agent 설정과 online network 파라미터를 JSON 파일로 저장한다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "network": self.online_network.to_dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DQNAgent:
        """저장된 JSON 파일에서 DQN agent를 복원한다."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = DQNConfig(**payload["config"])
        agent = cls(
            observation_size=int(payload["observation_size"]),
            action_size=int(payload["action_size"]),
            config=config,
        )
        agent.online_network = QNetwork.from_dict(payload["network"])
        agent.target_network.copy_from(agent.online_network)
        return agent

    def _target_value(self, transition: Transition) -> float:
        """DQN Bellman target 값을 계산한다."""
        if transition.done:
            return transition.reward
        next_q = self.target_network.predict(transition.next_state)
        return transition.reward + self.config.gamma * max(next_q)


class DQNPolicy:
    """저장된 DQN 모델을 greedy 평가 policy로 감싼다."""

    def __init__(self, model_path: Path) -> None:
        """모델 경로에서 DQN agent를 로드한다."""
        self.agent = DQNAgent.load(model_path)

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """현재 환경 observation에서 Q-value가 가장 큰 action을 선택한다."""
        return self.agent.select_greedy_action(env.current_observation())
