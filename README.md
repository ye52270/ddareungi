# DQN 기반 따릉이 재배치 시뮬레이터

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-V0%20baseline%20ready-2E8B57)
![Test](https://img.shields.io/badge/tests-pytest%20passing-4C9A2A)
![RL](https://img.shields.io/badge/RL-DQN%20planned-F28C28)

서울 공공자전거 **따릉이 재배치 문제**를 작은 강화학습 환경으로 단순화해 실험하는 기말 프로젝트다. 현재 저장소는 DQN 학습 전 단계인 **V0 baseline-ready** 상태이며, 환경, baseline 평가, episode log, replay 시각화를 먼저 검증한다.

V0의 핵심 질문은 다음과 같다.

> 작은 MDP에서 DQN이 시간대별 수요 패턴을 보고 트럭의 **방문 순서 정책**을 학습해, Random과 Low-stock heuristic보다 헛걸음(`unmet_demand`)을 줄일 수 있는가?

주의할 점은 V0의 DQN이 자전거를 몇 대 옮길지 직접 학습하지 않는다는 것이다. 싣기/내리기 수량은 환경의 rule-based heuristic이 처리하고, DQN은 **어느 대여소를 먼저 방문할지**를 학습한다.

자세한 MDP 설계, reward 계산 예시, 단계별 개발 계획은 [docs/01_detailed_design.md](docs/01_detailed_design.md)에 정리되어 있다.

## 강화학습 문제 정의

| 요소 | V0 정의 |
|---|---|
| State | 정규화된 관측값: 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 현재 시간 |
| Action | 트럭이 다음에 방문할 대여소 선택: `0`, `1`, `2` |
| Transition | 트럭 이동 → `target_stock=5` 기준 자동 싣기/내리기 → 수요/반납 발생 |
| Reward | `-10 * unmet_demand - movement_cost` |
| Episode | 24 step, 하루 운영을 단순화 |
| 학습 목표 | 시간대별 수요와 현재 재고를 보고, baseline보다 헛걸음을 줄이되 이동비용도 함께 관리하는 방문 순서 정책 학습 |

`unmet_demand`는 사용자가 대여소에 왔지만 자전거가 부족해서 빌리지 못한 수요다. README와 시각화에서는 이를 직관적으로 **헛걸음**이라고 표현한다.

## 검증 기준

| 지표 | 의미 | 중요도 |
|---|---|---|
| Avg Unmet Demand | 하루 평균 헛걸음 수 | 핵심 |
| Avg Reward | reward 식으로 계산한 평균 episode 점수 | 핵심 |
| Service Rate | 전체 대여 수요 중 처리에 성공한 비율 | 핵심 |
| Avg Movement Cost | 트럭이 얼마나 자주 움직였는지 | 보조 |
| Avg Full Returns | 반납 공간 부족으로 처리하지 못한 반납량 | V1 후보 지표 |

DQN의 성능은 절대 reward가 아니라 **같은 환경, 같은 episode 길이, 같은 seed 묶음**에서 Random과 Low-stock 대비 `Avg Unmet Demand`가 낮아지고 `Avg Reward`와 `Service Rate`가 개선되는지를 기준으로 평가한다.

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| `ToyDdareungiEnv` | 구현 완료 |
| Random baseline | 구현 완료 |
| Low-stock heuristic baseline | 구현 완료 |
| Episode log 저장 | 구현 완료 |
| `none` / `ansi` / `human` text render | 구현 완료 |
| Episode log tile replay | 구현 완료 |
| 누적 reward / unmet 시각화 | 구현 완료 |
| `pytest` 기반 smoke test | 구현 완료 |
| DQN 학습 | 예정 |
| pygame 창 replay | 구현 완료 |
| PNG/GIF asset export | 예정 |
| 실제 따릉이 데이터 replay | 예정 |

## Quick Start

개발 환경은 프로젝트 로컬 `.venv` 사용을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

창 기반 시각화만 추가로 설치하려면 다음 extra를 사용할 수 있다.

```bash
python -m pip install -e ".[viz]"
```

baseline 평가는 설치 후 CLI로 실행할 수 있다.

```bash
ddareungi-demo
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
```

`ddareungi-demo`는 V0 episode log를 생성한 뒤 pygame 창 replay를 바로 실행한다.

설치 없이 바로 실행하려면 다음처럼 `PYTHONPATH`를 지정한다.

```bash
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy random --episodes 5 --seed 42
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy low-stock --episodes 5 --seed 42
```

## V0 핵심 아이디어

V0는 구현 가능성을 확인하기 위한 가장 작은 환경이다. 실제 따릉이 데이터를 바로 쓰지 않고, 가상 대여소 3개와 재배치 트럭 1대로 시작한다.

| 항목 | V0 설계 |
|---|---|
| 대여소 | 가상 대여소 3개 |
| 대여소 이름 | 역할 기반 한국식 가상 이름: `마포구청역`, `여의도역`, `서울숲입구` |
| 트럭 | 1대 |
| 시간 | 1 step = 1시간, 1 episode = 24 step |
| 수요/반납 | 시간대별 범위 랜덤 패턴 |
| Action | 다음 방문 대여소 선택: `0`, `1`, `2` |
| 재배치 | 도착 후 `target_stock` 기준 자동 싣기/내리기 |
| Reward | `-10 * unmet_demand - movement_cost` |

현재 에이전트는 아직 DQN이 아니다. 먼저 Random과 Low-stock baseline을 같은 환경에서 실행해, 이후 DQN이 비교될 기준선을 만든다.

## Baseline 평가

아래 값은 같은 seed로 5 episode를 실행한 예시 결과다.

```bash
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
```

| Policy | Avg Reward | Avg Unmet Demand | Service Rate | Avg Full Returns | Avg Movement Cost |
|---|---:|---:|---:|---:|---:|
| Random | -423.60 | 40.80 | 68.54% | 0.60 | 15.60 |
| Low-stock | -455.60 | 44.80 | 65.46% | 6.00 | 7.60 |

현재 toy 환경에서는 Random이 Low-stock보다 약간 좋은 결과를 보인다. 이는 단순히 자전거가 적은 대여소만 따라가는 정책이 시간대별 수요/반납 패턴과 트럭 적재 상태를 충분히 고려하지 못할 수 있음을 보여준다.

평가 프로토콜:

- 모든 policy는 같은 `ToyDdareungiEnv` 설정에서 평가한다.
- episode 길이는 24 step으로 고정한다.
- 같은 seed 묶음으로 Random, Low-stock, 이후 DQN을 비교한다.
- DQN 평가는 학습이 끝난 모델을 불러와 탐험 없이 greedy action으로 수행한다.
- 개선 주장은 baseline 대비 헛걸음 감소, reward 개선, service rate 개선이 함께 보일 때만 한다.

`Avg Full Returns`는 현재 V0 reward에 포함하지 않고 진단 지표로만 기록한다. 반납 실패까지 reward에 반영하는 것은 V1 후보로 둔다.

## Text Render와 Window Replay

첫 episode를 콘솔 맵으로 확인할 수 있다.

```bash
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode ansi
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode human
```

예시 frame:

```text
+---------------- Toy Ddareungi V0 ----------------+
| time=07/24 truck_loc=2 truck_bikes=0 |
| 마포구청역   bikes=02    | 여의도역   bikes=10    |
| 서울숲입구 T bikes=01    | demand=[4, 1, 3]   |
| returns=[1, 4, 1]  | reward=-30 unmet=3 |
+--------------------------------------------------+
```

`ansi`는 평가 코드가 text frame을 받아 출력하고, `human`은 환경의 `step()`이 직접 frame을 출력한다.

FrozenLake 같은 별도 창 replay도 실행할 수 있다.

```bash
ddareungi-demo
```

직접 log를 저장한 뒤 replay하려면 다음 명령을 사용한다.

```bash
ddareungi-evaluate \
  --policy low-stock \
  --episodes 1 \
  --seed 42 \
  --save-log outputs/low_stock_episode.json

ddareungi-replay-window outputs/low_stock_episode.json --max-steps 10
```

창 조작:

- `Space`: 일시정지 / 재생
- `Right`: 다음 step
- `R`: 처음부터 replay
- `Q` 또는 `Esc`: 종료

## Episode Log와 시각화 계획

첫 episode log는 JSON으로 저장할 수 있다.

```bash
ddareungi-evaluate \
  --policy low-stock \
  --episodes 3 \
  --save-log outputs/low_stock_episode.json
```

저장된 log는 `state`, `action`, `reward`, `next_state`, 종료 flag, `demand`, `returns`, `unmet_demand`, `full_returns`, 트럭 상태, 대여소별 재고를 포함한다. Window replay에서 흐름을 읽기 쉽도록 `policy_name`, `learning_stage`, `episode_reward_so_far`, `episode_unmet_demand_so_far` 같은 표시용 누적 지표도 함께 저장한다.

저장된 log는 `ddareungi-replay`로 다시 볼 수 있다.

```bash
ddareungi-replay outputs/low_stock_episode.json --max-steps 5
ddareungi-replay outputs/low_stock_episode.json --max-steps 5 --no-color
```

시각화는 학습 코드와 렌더링 코드를 분리하고, 저장된 episode log를 replay하는 방식으로 동작한다.

```text
Ddareungi Tile Replay
-----------------------------------
+--------------+--------------+
| HOME         | WORK       T |
| bikes 06     | bikes 05     |
+--------------+--------------+
| PARK         | TRUCK        |
| bikes 02     | load 0       |
+--------------+--------------+
time=01/24  action=1  reward=0
unmet=0  full_returns=0  move_cost=1
```

시각화에서 보여줄 정보:

- 대여소별 자전거 수
- 트럭 위치
- 트럭 적재량
- 현재 time step
- 현재 점수와 누적 점수
- 현재 헛걸음과 누적 헛걸음
- 누적 이동비용
- 현재 policy와 학습 상태: 지금은 `DQN 학습 전 기준 정책`
- 하루 진행 bar와 헛걸음 발생 지점
- 대여소별 `안정`, `부족 주의`, `헛걸음 위험` 상태
- 대여 경험을 빠르게 읽기 위한 얼굴 표정: 웃음, 무표정, 찡그림, 화남
- episode 종료 결과: `우수`, `양호`, `주의`, `개선 필요`
- demand / returns

여기서 `헛걸음`은 대여 수요가 있었지만 대여소에 자전거가 부족해서 처리하지 못한 `unmet_demand`를 뜻한다. 트럭이 너무 많이 움직이는 문제는 `누적 이동비용`으로 따로 표시한다.

현재 pygame 화면은 DQN 학습 결과가 아니라 baseline 정책의 replay다. DQN이 추가되면 replay 화면은 한 episode를 설명하고, 별도 학습 화면에서 episode reward moving average, epsilon, loss, baseline 대비 개선율을 보여줘 초기 정책이 baseline보다 개선되는지 비교할 계획이다.

## Roadmap

| 단계 | 목표 | 상태 |
|---|---|---|
| V0-A | Toy environment 구현 | 완료 |
| V0-B | Random / Low-stock baseline 평가 | 완료 |
| V0-C | Episode log 저장과 text render | 완료 |
| V0-D | Episode log tile replay 시각화 | 완료 |
| V0-E | pygame 창 기반 FrozenLake 스타일 replay | 완료 |
| V0-F | 헛걸음 중심 compact game-style replay | 완료 |
| V1 | DQN 학습과 baseline 비교 | 예정 |
| V2 | Double DQN 비교 | 예정 |
| V3 | Dueling DQN 비교 | 예정 |
| V4 | 실제 따릉이 데이터 일부 replay | 예정 |

## 프로젝트 구조

```text
src/ddareungi_rl/
  envs/
    toy_ddareungi_env.py
  policies/
    baselines.py
  training/
    evaluate.py
  visualization/
    pixel_replay.py
    pygame_replay.py
tests/
  test_toy_ddareungi_env.py
  test_pixel_replay.py
  test_pygame_replay.py
docs/
  01_detailed_design.md
```

## 관련 연구

| 구분 | 참고 문헌 | 이 프로젝트에서 참고하는 점 |
|---|---|---|
| 공공자전거 재배치 RL | Yexin Li, Yu Zheng, Qiang Yang, "Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach", KDD 2018 | 대여소의 시간/공간적 불균형을 강화학습 문제로 모델링한다는 문제의식 |
| Dockless bike sharing DRL | Ling Pan et al., "A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems", AAAI 2019 | bike sharing rebalancing을 MDP로 구성한다는 관점 |
| Bike sharing rebalancing | Jiming Chen et al., "Rebalance Bike-Sharing System With Deep Sequential Learning", IEEE Intelligent Transportation Systems Magazine, 2020 | empty/full station이 사용자 경험을 악화시킨다는 배경 |
| DQN | Volodymyr Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015 | Q-network, replay buffer, target network 기반 학습 구조 |

이 프로젝트는 위 논문을 그대로 재현하지 않는다. 공공자전거 재배치를 시간·공간 수요 불균형 문제로 본다는 관점을 참고하고, 이를 교육용 toy MDP로 축소해 DQN 방문 정책이 baseline보다 나아지는지 검증한다.

참고 링크:

- [Dynamic Bike Reposition, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)
- [A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems, AAAI 2019](https://researchportal.hkust.edu.hk/en/publications/a-deep-reinforcement-learning-framework-for-rebalancing-dockless-)
- [Rebalance Bike-Sharing System With Deep Sequential Learning, IEEE ITS Magazine 2020](https://www.microsoft.com/en-us/research/publication/rebalance-bike-sharing-system-with-deep-sequential-learning/)
- [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236)
