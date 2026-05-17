# Toy DQN 기반 따릉이 재배치 MDP

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-V2%20PyTorch%20DQN%20smoke-2E8B57)
![Test](https://img.shields.io/badge/tests-pytest%20passing-4C9A2A)
![RL](https://img.shields.io/badge/RL-PyTorch%20DQN%20ready-F28C28)

## Abstract

본 프로젝트는 서울 공공자전거 **따릉이 재배치 문제**를 작은 Markov Decision Process(MDP)로 정식화하고, DQN 에이전트가 시간대별 수요와 현재 재고를 이용해 **헛걸음(unmet demand)을 줄이는 방문 순서 정책**을 학습할 수 있는지 검증한다.

V0는 실제 서울 전체 따릉이 운영을 재현하지 않는다. 대신 대여소 3개와 재배치 트럭 1대를 사용하는 toy environment를 만들고, `State`, `Action`, `Reward`, `Transition`이 명확한 최소 MDP를 먼저 검증한다. 현재 저장소는 V0 baseline을 기준선으로 고정하고, 같은 MDP 위에서 pure Python DQN과 PyTorch DQN 학습/greedy 평가를 실행할 수 있는 단계다.

> [!IMPORTANT]
> V0에서 DQN이 학습할 대상은 “자전거를 몇 대 옮길지”가 아니라 **다음에 어느 대여소를 방문할지**이다. 싣기/내리기 수량은 환경의 rule-based heuristic이 처리한다.

> [!NOTE]
> 현재 저장소에는 두 가지 DQN이 공존한다. V1은 외부 딥러닝 의존성 없이 구현한 **순수 Python one-hidden-layer Q-network smoke prototype**이고, V2는 같은 MDP를 **PyTorch `nn.Module` 기반 DQN**으로 학습한다. 성능 주장은 학습 중 점수가 아니라 held-out seed에서 greedy 평가한 결과에 한정한다.

> [!TIP]
> 처음 코드를 읽는다면 `ddareungi-walkthrough-env`로 Gymnasium 환경의 `reset()`과 `step(action)` 흐름을 먼저 확인하는 것을 권장한다.

## Problem Definition

공공자전거 시스템에서는 시간대와 지역에 따라 대여와 반납 수요가 불균형하게 발생한다. 특정 대여소의 자전거가 부족하면 사용자는 대여에 실패하고, 이는 `unmet_demand` 또는 customer loss로 해석할 수 있다. 반대로 특정 대여소에 자전거가 과도하게 몰리면 반납 실패 가능성도 생긴다.

재배치 트럭은 제한된 적재량과 이동비용을 가지므로, 현재 가장 부족해 보이는 대여소만 방문하는 단순 규칙이 항상 좋은 정책이 되지는 않는다. 현재의 방문 결정은 다음 시간대의 대여소 재고, 수요 충족률, 트럭 위치에 영향을 준다. 따라서 따릉이 재배치는 현재 상태에 기반해 행동을 선택하고, 그 행동이 미래 상태와 보상에 영향을 주는 **순차적 의사결정 문제**로 볼 수 있다.

<ins>연구 질문</ins>

> 작은 toy MDP에서 DQN이 시간대별 수요 패턴과 현재 재고를 보고, Random, Low-stock, Demand-aware heuristic보다 평균 헛걸음을 줄이면서 이동비용을 관리하는 방문 순서 정책을 학습할 수 있는가?

목표는 불필요한 트럭 이동을 과도하게 늘리지 않으면서, 하루 동안 발생하는 미충족 대여 수요를 줄이는 것이다.

## Academic Motivation

공공자전거 재배치 문제는 기존 연구에서도 시간/공간 수요 불균형을 줄이기 위한 순차적 의사결정 문제로 다루어진다.

> "empty stations ... lead to severe customer loss"
> Li, Zheng, Yang, **Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach**, KDD 2018

위 관점은 이 프로젝트의 핵심 지표인 `unmet_demand`와 연결된다. 즉, 사용자가 대여소에 왔지만 자전거가 없어 빌리지 못한 경우를 사용자 손실로 보고, 이를 reward에서 강하게 벌점화한다.

DQN은 Q-learning의 action-value function을 신경망으로 근사하는 방식이다. Mnih et al.은 DQN을 다음과 같이 설명한다.

> "deep Q-network ... can learn successful policies"
> Mnih et al., **Human-level control through deep reinforcement learning**, Nature 2015

이 프로젝트에서 DQN은 이미지 픽셀이 아니라 정규화된 재고/시간/트럭 상태를 입력으로 받고, 가능한 action 중 다음 방문 대여소를 선택하는 Q-network를 사용한다. 현재 구현은 교육용 pure Python prototype과 PyTorch implementation을 함께 제공하며, 학술적으로는 DQN의 핵심 구성인 Q-network, replay buffer, target network, epsilon-greedy exploration을 작은 toy MDP에서 검증하는 단계다.

## MDP Formulation

V0 환경은 다음과 같은 MDP로 정의한다.

| MDP 구성 | V0 정의 | 설명 |
|---|---|---|
| Agent | baseline policy, 향후 DQN policy | 다음 방문 대여소를 선택한다. |
| Environment | `ToyDdareungiEnv(gymnasium.Env)` | 트럭 이동, 자동 재배치, 수요/반납, reward 계산을 처리한다. |
| State `s_t` | `[b_0, b_1, b_2, l_t, c_t, h_t]` | 대여소별 재고, 트럭 위치, 트럭 적재량, 현재 시간 |
| Action `a_t` | `{0, 1, 2}` | 다음에 방문할 대여소 ID |
| Transition | `P(s_{t+1} given s_t, a_t)` | 이동, 자동 싣기/내리기, 수요/반납 샘플링 후 다음 상태 생성 |
| Reward `r_t` | `-10 * unmet_demand_t - movement_cost_t` | 헛걸음과 이동비용에 대한 벌점 |
| Horizon | 24 steps | 하루 운영을 24개 step으로 단순화 |

State vector는 코드에서 정규화된 observation으로 반환된다.

```text
s_t = [station_bikes_0, station_bikes_1, station_bikes_2,
       truck_location, truck_load, time_step]
```

Action space는 다음 방문 대여소 선택으로 제한한다.

```text
a_t ∈ {0, 1, 2}

0 -> 마포구청역 방문
1 -> 여의도역 방문
2 -> 서울숲입구 방문
```

V0의 action은 방문 대여소 선택만 포함한다. 자전거를 몇 대 싣거나 내릴지는 `target_stock=5`를 기준으로 한 rule-based heuristic이 결정한다. 이 단순화는 DQN이 먼저 **방문 순서 정책**을 학습할 수 있는지 확인하기 위한 의도적인 설계다.

## Environment Dynamics

각 step의 transition은 다음 순서로 진행된다.

```text
1. 에이전트가 현재 state s_t를 관찰한다.
2. 에이전트가 다음 방문 대여소 action a_t를 선택한다.
3. 트럭이 선택한 대여소로 이동한다.
4. 환경이 target_stock 기준으로 자동 싣기/내리기를 수행한다.
5. 시간대별 demand와 returns가 발생한다.
6. unmet_demand, movement_cost, reward를 계산한다.
7. 환경이 next_state s_{t+1}와 reward r_t를 반환한다.
```

현재 toy demand/return pattern은 실제 따릉이 데이터가 아니라 시간대별 경향을 흉내 낸 샘플링 범위다. 예를 들어 아침 시간대에는 특정 대여소의 대여 수요가 커지고, 저녁 시간대에는 다른 대여소의 반납 또는 대여 패턴이 달라지도록 구성했다.

## Reward Design

V0 reward는 사용자 불편을 직접 나타내는 `unmet_demand`를 가장 큰 벌점으로 두고, 트럭 이동에는 작은 비용을 부여한다.

```text
r_t = -10 × unmet_demand_t - movement_cost_t
```

| 항목 | 계수 | 이유 |
|---|---:|---|
| `unmet_demand_t` | `-10` | 사용자가 자전거를 빌리지 못한 직접 손실 |
| `movement_cost_t` | `-1` | 불필요한 트럭 이동 억제 |
| `full_returns_t` | reward 미포함 | V0에서는 진단 지표로만 기록하고 V1 이후 reward 후보로 둔다. |

Reward가 대부분 음수인 것은 문제가 아니다. 이 환경에서는 episode reward가 덜 음수가 되고, `Avg Unmet Demand`가 줄어들며, `Service Rate`가 높아지는지가 학습 개선의 핵심이다.

## Baselines and Evaluation Protocol

V0에서는 DQN을 바로 주장하지 않고, 먼저 같은 환경에서 비교할 baseline을 정의한다.

| Policy | 설명 | 역할 |
|---|---|---|
| Random | 대여소를 균등 무작위로 선택한다. | 무학습 stochastic baseline |
| Low-stock heuristic | 현재 자전거가 가장 적은 대여소를 선택한다. | 단순 운영 규칙 기준선 |
| Demand-aware heuristic | 현재 시간대 예상 수요와 재고 차이가 가장 큰 대여소를 선택한다. | 수요 패턴을 보는 진단 기준선 |
| Pure Python DQN | 직접 구현한 작은 Q-network로 `Q(s, a)`를 학습한다. | V1 교육용 smoke prototype |
| PyTorch DQN | `torch.nn.Module`과 Adam optimizer로 `Q(s, a)`를 학습한다. | V2 학습 기반 정책 |

모든 policy는 같은 환경 설정, 같은 episode 길이, 같은 seed 묶음에서 평가한다. DQN은 학습 중 exploration reward가 아니라, 학습 완료 후 greedy action으로 별도 평가한다.

향후 DQN 학습 목표는 다음 Bellman target을 기반으로 한다.

```text
y_t = r_t + γ max_a' Q(s_{t+1}, a'; θ^-)

loss = (y_t - Q(s_t, a_t; θ))^2
```

terminal step에서는 bootstrap term을 제거한다. DQN 구현은 replay buffer, target network, epsilon-greedy exploration을 사용해 방문 순서 정책을 학습하고, 학습 후에는 baseline과 동일한 평가 조건에서 비교한다.

## Evaluation Metrics

| Metric | 정의 | 해석 |
|---|---|---|
| Avg Unmet Demand | episode별 미충족 대여 수요 평균 | 가장 중요한 사용자 손실 지표 |
| Service Rate | `served_demand / total_demand` | 전체 대여 수요 중 처리 성공률. 여기서 `total_demand = served_demand + unmet_demand` |
| Avg Reward | episode reward 합의 평균 | reward 기준 종합 성능 |
| Avg Movement Cost | episode별 이동비용 합의 평균 | 운영 효율 |
| Avg Full Returns | 반납 실패 평균 | V1 이후 reward 확장 후보 |
| Action Distribution | action별 선택 횟수 | 한 대여소로 정책이 쏠리는지 확인 |
| Same-location Rate | 현재 위치를 다시 선택한 비율 | 트럭이 머무는 행동이 많은지 확인 |

주요 성공 기준은 Random, Low-stock, Demand-aware baseline보다 낮은 `Avg Unmet Demand`와 높은 `Service Rate`다. `Avg Reward`는 unmet demand와 movement cost를 함께 반영하는 보조 종합 지표로 사용한다.

## Current V0 Results

아래 값은 같은 seed로 5 episode를 실행한 예시 결과다.

```bash
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
```

| Policy | Avg Reward | Avg Unmet Demand | Service Rate | Avg Full Returns | Avg Movement Cost |
|---|---:|---:|---:|---:|---:|
| Random | -423.60 | 40.80 | 68.54% | 0.60 | 15.60 |
| Low-stock | -455.60 | 44.80 | 65.46% | 6.00 | 7.60 |
| Demand-aware | -455.20 | 44.80 | 65.46% | 6.00 | 7.20 |

현재 V0 예시 결과에서는 Low-stock과 Demand-aware heuristic이 Random보다 낮은 성능을 보인다. 이는 단순히 현재 재고가 낮은 대여소를 방문하거나 toy 수요 패턴의 평균 부족분만 보는 규칙이 트럭 적재 상태와 이후 반납까지 충분히 반영하지 못할 수 있음을 보여준다. 이 관찰은 학습 기반 방문 정책을 비교할 필요성을 만든다.

## Scope, Assumptions, and Limitations

V0의 한계는 의도적인 단순화다. 실제 복잡도를 추가하기 전에 MDP loop, reward, baseline 비교가 작동하는지 먼저 검증하는 것이 V0의 목표다.

현재 V0는 baseline과 replay까지 완료한 안정 기준선으로 두고, 이후 작업은 같은 MDP 위에서 PyTorch DQN의 multi-seed 평가와 baseline 비교로 진행한다.

| 한계 | 현재 처리 | 향후 개선 |
|---|---|---|
| 실제 따릉이 데이터 미사용 | toy demand/return pattern 사용 | 서울 열린데이터 기반 시간대별 수요 구성 |
| 작은 환경 | 대여소 3개, 트럭 1대 | 대여소 수 확장, 클러스터 단위 실험 |
| 단순 action | 다음 방문 대여소만 선택 | 방문지와 싣기/내리기 수량을 함께 선택 |
| 단순 이동비용 | 이동 여부만 비용으로 반영 | 거리/시간 기반 이동비용 |
| 반납 실패 | `full_returns`를 진단 지표로만 기록 | reward에 반납 실패 벌점 추가 |
| DQN 초기 단계 | 순수 Python DQN과 PyTorch DQN 학습/평가 가능 | multi-seed 평가와 reward ablation |

## Roadmap

| Stage | Research Question | Implementation | Status |
|---|---|---|---|
| V0 | Toy MDP가 실행 가능하고 해석 가능한가? | environment, baselines, logs, replay | 완료 |
| V1 | DQN 학습 루프가 작동하는가? | pure Python DQN train/evaluate | 완료 |
| V2 | PyTorch DQN이 baseline보다 나은 방문 순서 정책을 학습하는가? | PyTorch DQN train/evaluate, greedy evaluation | 초기 구현 |
| V2.5 | 결과가 seed와 reward 설계에 안정적인가? | multi-seed evaluation, reward ablation | 예정 |
| V3 | DQN 변형이 안정성을 개선하는가? | Double DQN, Dueling DQN 비교 | 예정 |
| V4 | reward/dynamics를 더 현실적으로 만들 수 있는가? | 거리 기반 이동비용, 반납 실패 reward | 예정 |
| V5 | toy demand를 실제 데이터 패턴으로 대체할 수 있는가? | 실제 따릉이 데이터 일부 반영 | 예정 |

## Implementation Notes

현재 구현은 강화학습 실험을 위한 최소 루프에 집중한다.

| 모듈 | 역할 |
|---|---|
| `ToyDdareungiEnv` | V0 MDP 환경. `reset()`, `step(action)`, `render()` 제공 |
| `RandomPolicy` | 무작위 baseline |
| `LowStockPolicy` | 가장 재고가 낮은 대여소를 방문하는 heuristic baseline |
| `DQNAgent` | 순수 Python MLP, replay buffer, target network, epsilon-greedy 학습 |
| `TorchDQNAgent` | PyTorch MLP, replay buffer, target network, Adam optimizer 학습 |
| `train_dqn.py` | DQN 학습, model/metrics/log 저장 |
| `train_torch_dqn.py` | PyTorch DQN 학습, `.pt` model/metrics/log 저장 |
| `evaluate.py` | baseline/DQN episode 실행, metric 집계, episode log 저장 |
| `pygame_replay.py` | 저장된 episode log를 FrozenLake 스타일 창으로 replay |
| `env_step_walkthrough.py` | Gymnasium 환경의 한 step transition을 수업용으로 출력 |

핵심 reward 계산은 환경의 `step()`에서 다음과 같이 구현된다.

```python
movement_cost = 1 if action != previous_location else 0
reward = -10 * unmet_demand - movement_cost
```

Low-stock heuristic baseline은 다음처럼 현재 재고가 가장 적은 대여소를 선택한다.

```python
action = argmin(station_bikes)
```

자세한 MDP 설계, reward 계산 예시, 단계별 개발 계획은 [docs/01_detailed_design.md](docs/01_detailed_design.md)에 정리되어 있다.

## Visualization

V0의 시각화는 학습 자체가 아니라 episode log를 replay하는 용도다. 시각화는 다음 정보를 보여준다.

- 대여소별 자전거 수, 수요, 반납
- 트럭 위치와 적재량
- 현재 time step과 하루 진행 bar
- 현재 점수, 누적 점수, 헛걸음 수, 이동비용
- 대여소별 `안정`, `부족 주의`, `헛걸음 위험` 상태
- 대여 경험을 빠르게 읽기 위한 얼굴 표정: 웃음, 무표정, 찡그림, 화남

시각화는 학습 코드와 분리되어 있으며, 저장된 episode log를 읽어 replay한다.

## Reproducibility / How to Run

개발 환경은 프로젝트 로컬 `.venv` 사용을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

실습용 콘솔 메뉴는 다음 명령으로 실행한다.

```bash
ddareungi
```

메뉴에서는 baseline 평가, pure Python DQN 학습/평가, PyTorch DQN 학습/평가, visualization 포함 평가를 선택할 수 있다. visualization 메뉴는 창을 닫으면 프로그램도 함께 종료된다.

환경의 한 step 흐름을 먼저 보고 싶다면 다음 명령을 실행한다.

```bash
ddareungi-walkthrough-env
```

baseline 평가는 다음 명령으로 재현할 수 있다.

```bash
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
ddareungi-evaluate --policy demand-aware --episodes 5 --seed 42
```

DQN은 학습과 평가를 분리해서 실행한다.

```bash
ddareungi-train-dqn \
  --episodes 300 \
  --seed 42 \
  --model-out outputs/models/dqn_v1.json \
  --metrics-out outputs/metrics/dqn_train_metrics.json

ddareungi-evaluate \
  --policy dqn \
  --model-path outputs/models/dqn_v1.json \
  --episodes 20 \
  --seed 1000
```

PyTorch DQN도 같은 환경에서 학습과 평가를 분리해서 실행한다.

```bash
python -m pip install -e ".[torch]"

ddareungi-train-torch-dqn \
  --episodes 300 \
  --seed 42 \
  --model-out outputs/models/torch_dqn_v1.pt \
  --metrics-out outputs/metrics/torch_dqn_train_metrics.json

ddareungi-evaluate \
  --policy torch-dqn \
  --model-path outputs/models/torch_dqn_v1.pt \
  --episodes 20 \
  --seed 1000
```

짧은 학습 결과는 smoke check이며, 성능 주장은 held-out seed 묶음에서 Random, Low-stock, Demand-aware, DQN을 같은 조건으로 비교한 뒤에만 한다.

episode log를 저장하고 창 기반 replay를 실행하려면 다음 명령을 사용한다.

```bash
ddareungi-evaluate \
  --policy low-stock \
  --episodes 1 \
  --seed 42 \
  --save-log outputs/low_stock_episode.json

ddareungi-replay-window outputs/low_stock_episode.json --max-steps 10
```

간단히 시각화만 실행하려면 다음 명령을 사용할 수 있다.

```bash
ddareungi-demo
```

## References

| 구분 | 참고 문헌 | 이 프로젝트에서 참고하는 점 |
|---|---|---|
| 공공자전거 재배치 RL | Yexin Li, Yu Zheng, Qiang Yang, "Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach", KDD 2018 | 대여소의 시간/공간적 불균형을 강화학습 문제로 모델링한다는 문제의식 |
| Dockless bike sharing DRL | Ling Pan et al., "A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems", AAAI 2019 | bike sharing rebalancing을 MDP로 구성한다는 관점 |
| Bike sharing rebalancing | Jiming Chen et al., "Rebalance Bike-Sharing System With Deep Sequential Learning", IEEE Intelligent Transportation Systems Magazine, 2020 | empty/full station이 사용자 경험을 악화시킨다는 배경 |
| DQN | Volodymyr Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015 | Q-network, replay buffer, target network 기반 학습 구조 |

이 프로젝트는 위 논문을 그대로 재현하지 않는다. 공공자전거 재배치를 시간·공간 수요 불균형 문제로 본다는 관점을 참고하고, 이를 교육용 toy MDP로 축소해 DQN 방문 정책이 baseline보다 나아지는지 검증한다.

참고 링크:

- [Dynamic Bike Reposition, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)
- [A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems, AAAI 2019](https://researchportal.hkust.edu.hk/en/publications/a-deep-reinforcement-learning-framework-for-rebalancing-dockless-)
- [Rebalance Bike-Sharing System With Deep Sequential Learning, IEEE ITS Magazine 2020](https://www.microsoft.com/en-us/research/publication/rebalance-bike-sharing-system-with-deep-sequential-learning/)
- [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236)
