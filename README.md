# DQN 기반 따릉이 재배치 시뮬레이터

이 프로젝트는 서울 공공자전거 **따릉이 재배치 문제**를 강화학습으로 단순화해 실험하는 기말 프로젝트다. 목표는 특정 대여소에 자전거가 부족하거나 너무 많은 상황을 줄이도록, 재배치 트럭의 이동 정책을 학습하는 것이다.

프로젝트는 두 단계로 나누어 진행한다.

```text
V0: Proof of Concept
  작은 가상 환경에서 DQN이 재배치 정책을 학습할 수 있는지 확인한다.

V1: Final / Data-driven Version
  V0에서 검증한 구조를 실제 따릉이 데이터와 더 현실적인 운영 조건으로 확장한다.
```

자세한 V0 설계 예시와 reward 계산 과정은 [docs/01_detailed_design.md](docs/01_detailed_design.md)에 정리한다.

## 프로젝트 목표

- 따릉이 재배치 문제를 강화학습의 `Agent`, `Environment`, `State`, `Action`, `Reward`로 명확하게 정의한다.
- 먼저 작은 V0 환경에서 DQN 학습이 가능한지 확인한다.
- 이후 실제 공공데이터 기반 수요, 반납 실패, 이동 비용 등을 반영한 V1 환경으로 확장한다.
- DQN, Double DQN, Dueling DQN을 비교할 수 있는 구조를 만든다.
- Random, NO-OP, heuristic baseline과 비교하여 강화학습의 효과를 확인한다.

## 관련 연구

공공자전거 재배치 문제는 시간대와 지역에 따라 수요가 달라지는 **spatio-temporal imbalance** 문제로 볼 수 있다. 기존 연구에서도 자전거가 비거나 가득 찬 대여소가 고객 손실을 만든다고 보고, 이를 줄이기 위해 재배치 정책을 학습하거나 최적화하는 접근을 제안했다.

| 구분 | 참고 문헌 | 이 프로젝트에서 참고하는 점 |
|---|---|---|
| 공공자전거 재배치 RL | Yexin Li, Yu Zheng, Qiang Yang, "Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach", KDD 2018 | 대여소의 시간/공간적 불균형을 강화학습 문제로 모델링하고, 장기 고객 손실을 줄이는 재배치 정책을 학습한다는 문제의식 |
| Dockless bike sharing DRL | Ling Pan et al., "A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems", AAAI 2019 | bike sharing rebalancing을 MDP로 보고, 공간/시간 정보를 고려한 deep RL framework를 구성한다는 점 |
| Bike sharing rebalancing | Jiming Chen et al., "Rebalance Bike-Sharing System With Deep Sequential Learning", IEEE Intelligent Transportation Systems Magazine, 2020 | bike station이 empty/full 상태가 되면 사용자 경험이 나빠지고, 대규모 재배치 문제가 어렵다는 배경 설명 |
| DQN | Volodymyr Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015 | Q-network, replay buffer, target network를 사용하는 DQN 학습 구조 |

참고 링크:

- [Dynamic Bike Reposition, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)
- [A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems, AAAI 2019](https://researchportal.hkust.edu.hk/en/publications/a-deep-reinforcement-learning-framework-for-rebalancing-dockless-)
- [Rebalance Bike-Sharing System With Deep Sequential Learning, IEEE ITS Magazine 2020](https://www.microsoft.com/en-us/research/publication/rebalance-bike-sharing-system-with-deep-sequential-learning/)
- [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236)

## V0: Proof of Concept

V0는 구현 가능성을 확인하기 위한 가장 작은 환경이다. 실제 따릉이 데이터를 바로 쓰지 않고, 가상 대여소 3개와 트럭 1대로 시작한다.

| 항목 | V0 설계 |
|---|---|
| 목적 | DQN이 재배치 방문 정책을 학습할 수 있는지 확인 |
| 대여소 | 가상 대여소 3개 |
| 트럭 | 1대 |
| 시간 | 1 step = 1시간, 1 episode = 24 step |
| 수요 | 시간대별 범위 랜덤 수요 |
| 재배치 | 트럭 도착 후 싣기/내리기는 규칙 기반 자동 처리 |
| 알고리즘 | DQN 우선 구현 |
| 종료 조건 | 24 step 이후 episode 종료 |

### V0 MDP 정의

| 구성 요소 | V0 정의 |
|---|---|
| Agent | 트럭을 어느 대여소로 보낼지 결정하는 DQN 모델 |
| Environment | 대여소 재고, 트럭 위치, 수요 발생, 자동 재배치, reward 계산을 담당하는 시뮬레이터 |
| State | 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 현재 시간 |
| Action | 방문할 대여소 선택: `0`, `1`, `2` |
| Reward | 대여 실패 패널티, 이동 비용, 수요 만족 보상 |

V0의 state 예시는 다음과 같다.

```text
state = [
  station_0_bikes,
  station_1_bikes,
  station_2_bikes,
  truck_location,
  truck_bikes,
  time_step
]
```

V0의 action은 단순하다.

```text
0: 0번 대여소 방문
1: 1번 대여소 방문
2: 2번 대여소 방문
```

중요한 점은 V0에서 DQN이 **몇 대를 싣고 내릴지 직접 결정하지 않는다**는 것이다. 적재/하차 수량은 `target_stock` 기준의 rule-based heuristic으로 환경이 처리한다. 따라서 V0의 학습 목표는 수량 결정이 아니라 **시간대별 수요와 현재 재고를 보고 방문할 대여소를 고르는 정책**을 학습하는 것이다.

V0 reward는 단순하게 시작한다.

```text
reward = -10 * unmet_demand - movement_cost + service_bonus
```

## V1: Final / Data-driven Version

V1은 팀원 설계안을 반영해 V0를 실제 따릉이 운영 문제에 가깝게 확장하는 단계다. 단, 모든 기능을 한 번에 구현하지 않고 필수 구현과 선택 구현을 나눈다.

### V1 필수 구현

| 항목 | V1 필수 설계 |
|---|---|
| 대여소 | 실제 따릉이 대여소 3~5개 또는 특정 소규모 권역 |
| 시간 | 가능하면 1 step = 10분, 1 episode = 24시간 |
| 수요 | 서울시 공공자전거 대여이력 replay |
| State | 정류소별 재고, 트럭 위치/적재량, 시간 정보 |
| Action | 다음 방문 정류소 선택 |
| Reward | 대여 실패(stockout), 반납 실패(full), 이동 비용 |
| Baseline | NO-OP, Random, heuristic |
| Algorithm | DQN, Double DQN, Dueling DQN 비교 |

V1 reward는 다음처럼 현실적인 비용 중심으로 확장한다.

```text
reward =
  stockout_weight * stockout_count
  + full_weight * full_count
  + distance_weight * travel_distance
  + time_weight * travel_steps
```

예상 가중치:

| 항목 | 예시 값 | 의미 |
|---|---:|---|
| `stockout_weight` | `-1.0` | 대여 실패 1건당 패널티 |
| `full_weight` | `-0.8` | 반납 실패 1건당 패널티 |
| `distance_weight` | `-0.01` | 이동 거리 1km당 비용 |
| `time_weight` | `-0.005` | 이동 중인 step당 비용 |

### V1 선택 구현

시간이 남으면 다음 요소를 추가한다.

| 항목 | 설명 |
|---|---|
| 다중 트럭 | 트럭 2~3대로 확장 |
| Parameter sharing | 여러 트럭이 하나의 Q-network 정책을 공유 |
| 날씨 feature | 기온, 강수량, 습도, 풍속 등 |
| 공휴일/요일 feature | 평일/주말/공휴일 수요 차이 반영 |
| Action mask | 이동 중인 트럭, 불가능한 목적지 등을 선택하지 않도록 제한 |
| 거리 기반 이동 시간 | 대여소 위도/경도 기반 이동 거리와 이동 step 계산 |

## V0와 V1 비교

| 항목 | V0: PoC | V1: Final |
|---|---|---|
| 목적 | DQN 구조 검증 | 실제 데이터 기반 확장 |
| 대여소 | 가상 3개 | 실제 대여소 3~5개 또는 소규모 권역 |
| 트럭 | 1대 | 1대 우선, 선택적으로 여러 대 |
| 시간 단위 | 1시간 | 10분 목표 |
| episode | 24 step | 144 step 목표 |
| 수요 | 시간대별 랜덤 패턴 | 실제 대여/반납 이력 replay |
| State | 재고, 트럭 위치, 적재량, 시간 | 재고, 트럭 상태, 시간/요일, 선택적으로 날씨 |
| Action | 방문 대여소 선택 | 방문 정류소 선택 |
| Reward | 대여 실패 중심 | 대여 실패 + 반납 실패 + 이동 비용 |
| 장점 | 구현이 쉽고 설명이 명확함 | 현실성과 발표 설득력이 높음 |
| 위험 | 너무 단순해 보일 수 있음 | 구현 범위가 커질 수 있음 |

## Baseline

강화학습 결과를 해석하려면 비교 대상이 필요하다.

| Policy | 설명 |
|---|---|
| NO-OP | 트럭이 움직이지 않음. 재배치 자체가 필요한지 확인하는 기준 |
| Random | 매 step 무작위 대여소 선택 |
| Low-stock | 현재 자전거 수가 가장 적은 대여소 선택 |
| Most-imbalanced | 트럭이 비어 있으면 많은 곳으로, 트럭이 차 있으면 부족한 곳으로 이동 |

## 알고리즘 비교 계획

| 알고리즘 | 역할 |
|---|---|
| DQN | 기본 value-based deep RL |
| Double DQN | Q-value 과대추정을 줄이는 DQN 변형 |
| Dueling DQN | state value와 action advantage를 분리해 추정하는 DQN 변형 |

Stable-Baselines3의 기본 DQN은 vanilla DQN 중심이므로, Double DQN과 Dueling DQN 비교가 필요하면 직접 구현하거나 별도 구현체를 사용해야 한다.

## 공공데이터 사용 계획

V1에서 사용할 데이터 후보는 다음과 같다.

| 데이터 | 사용할 내용 |
|---|---|
| 서울시 공공자전거 대여소 정보 | 대여소 ID, 이름, 위도/경도, 거치대 수 |
| 서울시 공공자전거 대여이력 정보 | 시간대별 대여량, 반납량 |
| 서울시 공공자전거 실시간 대여정보 | episode 시작 시 초기 자전거 수 |
| 날씨 데이터 | 선택 feature: 기온, 강수량, 습도, 풍속 |
| 공휴일 데이터 | 선택 feature: 평일/주말/공휴일 구분 |

처음에는 전체 서울 데이터를 사용하지 않고, 특정 지역의 3~5개 대여소만 선택해 replay 환경을 만든다.

## 최종 결과물

- V0 환경 설계와 DQN 학습 결과
- V1 데이터 기반 확장 설계 또는 일부 구현
- Baseline 대비 DQN 성능 비교
- DQN, Double DQN, Dueling DQN 비교
- reward curve, stockout/full 횟수, 이동 비용 그래프
- 한계점과 향후 확장 방향

