# DQN 기반 따릉이 재배치 시뮬레이터

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-V0%20baseline%20ready-2E8B57)
![Test](https://img.shields.io/badge/tests-pytest%20passing-4C9A2A)
![RL](https://img.shields.io/badge/RL-DQN%20planned-F28C28)

서울 공공자전거 **따릉이 재배치 문제**를 작은 강화학습 환경으로 단순화해 실험하는 기말 프로젝트다. 현재는 DQN 학습 전 단계로, V0 toy 환경과 baseline 평가 루프를 먼저 완성했다.

자세한 MDP 설계, reward 계산 예시, 단계별 개발 계획은 [docs/01_detailed_design.md](docs/01_detailed_design.md)에 정리되어 있다.

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| `ToyDdareungiEnv` | 구현 완료 |
| Random baseline | 구현 완료 |
| Low-stock heuristic baseline | 구현 완료 |
| Episode log 저장 | 구현 완료 |
| `none` / `ansi` / `human` text render | 구현 완료 |
| `pytest` 기반 smoke test | 구현 완료 |
| DQN 학습 | 예정 |
| FrozenLake 스타일 pixel replay | 예정 |
| 실제 따릉이 데이터 replay | 예정 |

## Quick Start

개발 환경은 프로젝트 로컬 `.venv` 사용을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

baseline 평가는 설치 후 CLI로 실행할 수 있다.

```bash
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
```

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
| 트럭 | 1대 |
| 시간 | 1 step = 1시간, 1 episode = 24 step |
| 수요/반납 | 시간대별 범위 랜덤 패턴 |
| Action | 다음 방문 대여소 선택: `0`, `1`, `2` |
| 재배치 | 도착 후 `target_stock` 기준 자동 싣기/내리기 |
| Reward | `-10 * unmet_demand - movement_cost + service_bonus` |

현재 에이전트는 아직 DQN이 아니다. 먼저 Random과 Low-stock baseline을 같은 환경에서 실행해, 이후 DQN이 비교될 기준선을 만든다.

## Baseline 평가

아래 값은 같은 seed로 5 episode를 실행한 예시 결과다.

```bash
ddareungi-evaluate --policy random --episodes 5 --seed 42
ddareungi-evaluate --policy low-stock --episodes 5 --seed 42
```

| Policy | Avg Reward | Avg Unmet Demand | Avg Full Returns | Avg Movement Cost |
|---|---:|---:|---:|---:|
| Random | -413.20 | 40.80 | 0.60 | 15.60 |
| Low-stock | -445.80 | 44.80 | 6.00 | 7.60 |

현재 toy 환경에서는 Random이 Low-stock보다 약간 좋은 결과를 보인다. 이는 단순히 자전거가 적은 대여소만 따라가는 정책이 시간대별 수요/반납 패턴과 트럭 적재 상태를 충분히 고려하지 못할 수 있음을 보여준다.

## Text Render

첫 episode를 콘솔 맵으로 확인할 수 있다.

```bash
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode ansi
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode human
```

예시 frame:

```text
+---------------- Toy Ddareungi V0 ----------------+
| time=07/24 truck_loc=2 truck_bikes=0 |
| HOME   bikes=02    | WORK   bikes=10    |
| PARK T bikes=01    | demand=[4, 1, 3]   |
| returns=[1, 4, 1]  | reward=-30 unmet=3 |
+--------------------------------------------------+
```

`ansi`는 평가 코드가 text frame을 받아 출력하고, `human`은 환경의 `step()`이 직접 frame을 출력한다.

## Episode Log와 시각화 계획

첫 episode log는 JSON으로 저장할 수 있다.

```bash
ddareungi-evaluate \
  --policy low-stock \
  --episodes 3 \
  --save-log outputs/low_stock_episode.json
```

저장된 log는 `state`, `action`, `reward`, `next_state`, 종료 flag, `demand`, `returns`, `unmet_demand`, `full_returns`, 트럭 상태, 대여소별 재고를 포함한다.

다음 시각화 단계에서는 학습 코드와 렌더링 코드를 분리하고, 저장된 episode log를 replay하는 방식으로 FrozenLake 같은 작은 tile map을 만들 계획이다.

```text
+---------+---------+
| HOME    | WORK    |
| bikes 2 | bikes10 |
+---------+---------+
| PARK T  | DEPOT   |
| bikes 1 | load 0  |
+---------+---------+
```

시각화에서 보여줄 정보:

- 대여소별 자전거 수
- 트럭 위치
- 트럭 적재량
- 현재 time step
- reward
- unmet demand
- demand / returns

## Roadmap

| 단계 | 목표 | 상태 |
|---|---|---|
| V0-A | Toy environment 구현 | 완료 |
| V0-B | Random / Low-stock baseline 평가 | 완료 |
| V0-C | Episode log 저장과 text render | 완료 |
| V0-D | Pixel/tile replay 시각화 | 예정 |
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
tests/
  test_toy_ddareungi_env.py
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

참고 링크:

- [Dynamic Bike Reposition, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)
- [A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems, AAAI 2019](https://researchportal.hkust.edu.hk/en/publications/a-deep-reinforcement-learning-framework-for-rebalancing-dockless-)
- [Rebalance Bike-Sharing System With Deep Sequential Learning, IEEE ITS Magazine 2020](https://www.microsoft.com/en-us/research/publication/rebalance-bike-sharing-system-with-deep-sequential-learning/)
- [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236)
