# DQN 기반 따릉이 재배치 시뮬레이터

이 프로젝트는 **강화학습으로 서울 공공자전거 따릉이의 자전거 재배치 문제를 어떻게 단순화하고 해결할 수 있는지**를 실험하기 위한 기말 프로젝트다. 실제 따릉이 운영의 모든 복잡한 조건을 처음부터 구현하기보다는, 먼저 `State`, `Action`, `Reward`가 명확한 작은 문제를 만들고 DQN이 재배치 정책을 학습하는지 확인하는 데 초점을 둔다.

초기 버전인 V0에서는 대여소 3개와 재배치 트럭 1대를 가진 작은 시뮬레이션 환경을 만든다. 에이전트는 매 step마다 어느 대여소를 방문할지 선택하고, 트럭이 도착한 뒤 몇 대를 싣거나 내릴지는 환경의 간단한 규칙으로 자동 처리한다. 이를 통해 DQN은 복잡한 수량 결정 대신 **시간대별 수요 패턴을 고려한 방문 순서 정책**을 학습한다.

## 프로젝트 목표

- 따릉이 재배치 문제를 강화학습의 `State`, `Action`, `Reward`로 명확하게 정의한다.
- 작은 Gymnasium 스타일 환경에서 DQN을 학습시킨다.
- Random policy 같은 baseline과 DQN의 결과를 비교한다.
- 이후 Double DQN, Dueling DQN 등 여러 알고리즘을 선택 실행할 수 있는 구조로 확장한다.
- 최종적으로 서울 공공데이터의 따릉이 데이터를 일부 활용하여 현실적인 수요 패턴을 반영한다.

## 관련 연구

이 프로젝트에서 가장 핵심적으로 참고할 연구는 다음 논문이다.

| 구분 | 참고 문헌 | 이 프로젝트에서 사용한 점 |
|---|---|---|
| 공공자전거 재배치 문제 | Yexin Li, Yu Zheng, Qiang Yang, "Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach", KDD 2018 | 공공자전거 재배치 문제를 시간적, 공간적 수요 불균형 문제로 보고, 장기적인 고객 손실을 줄이기 위해 강화학습 기반 재배치 정책을 학습한다는 아이디어 |
| DQN | Volodymyr Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015 | Q-table 대신 neural network로 action-value function을 근사하고, replay buffer와 target network를 사용해 학습을 안정화하는 방식 |

Li, Zheng, Yang의 KDD 2018 논문은 공공자전거 시스템에서 특정 대여소가 비거나 가득 차면 고객 손실이 발생한다고 설명한다. 또한 운영 중인 시스템에서 자전거를 계속 재배치해야 하며, 장기적인 고객 손실을 줄이는 정책을 학습하기 위해 spatio-temporal reinforcement learning 모델을 제안한다.

본 프로젝트는 이 연구를 그대로 재현하지는 않는다. 대신 논문의 문제의식을 작은 교육용 환경으로 축소한다. 즉, 실제 도시 규모의 대여소와 여러 재배치 차량 대신, V0에서는 3개 대여소와 1대의 트럭만 사용하여 DQN이 재배치 의사결정을 학습할 수 있는 최소 환경을 만든다.

참고 링크:

- [Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)
- [Human-level control through deep reinforcement learning, Nature 2015](https://www.nature.com/articles/nature14236)

## 문제 정의

따릉이와 같은 공공자전거 시스템에서는 시간대와 지역에 따라 수요가 다르게 발생한다. 예를 들어 아침에는 주거지역에서 자전거를 빌리려는 수요가 많고, 업무지역이나 지하철역 근처로 자전거가 이동할 수 있다. 반대로 저녁에는 다른 방향의 이동이 많아질 수 있다.

이런 수요 불균형 때문에 어떤 대여소는 자전거가 부족해지고, 어떤 대여소는 자전거가 너무 많아질 수 있다. 자전거가 부족하면 사용자가 대여에 실패하고, 자전거가 너무 많으면 반납 공간 부족이나 운영 비효율이 발생할 수 있다.

이 프로젝트에서는 문제를 다음처럼 단순화한다.

```text
목표:
  제한된 재배치 트럭을 이용해 대여소별 자전거 부족 상황을 줄인다.

강화학습 관점:
  에이전트는 현재 대여소 상태와 시간 정보를 보고,
  어느 대여소를 방문할지 선택한다.

성공 기준:
  수요가 발생했을 때 자전거가 부족해서 실패하는 사용자의 수를 줄인다.
```

## V0 환경 설계

V0는 DQN을 처음 적용하기 위한 가장 작은 환경이다.

### 기본 조건

| 항목 | 내용 |
|---|---|
| 대여소 수 | 3개 |
| 재배치 차량 | 트럭 1대 |
| 대여소 용량 | 각 10대 |
| 트럭 용량 | 5대 |
| episode 길이 | 24 step |
| 시간 의미 | 하루 24시간을 단순화 |
| 초기 자전거 수 | episode 시작 시 랜덤 |
| 수요 발생 | 시간대별 범위 랜덤 패턴 |
| 재배치 방식 | 도착한 대여소에서 환경 규칙으로 자동 싣기/내리기 |
| 종료 조건 | 24 step이 지나면 episode 종료 |

### 대여소 가정

V0에서는 실제 특정 따릉이 대여소를 그대로 재현하지 않고, 역할만 가진 가상 대여소를 사용한다. 다만 발표와 시각화에서 한국적인 맥락이 보이도록 역할 기반 한국식 가상 이름을 붙인다.

| 대여소 | 표시 이름 | 의미 | 수요 특징 |
|---|---|---|---|
| station 0 | 마포구청역 | 주거지역 | 아침 수요 높음 |
| station 1 | 여의도역 | 업무지역 | 낮 또는 저녁 수요 높음 |
| station 2 | 서울숲입구 | 공원/중간지역 | 전반적으로 중간 수준 |

이렇게 설정하면 단순한 랜덤 수요보다 학습할 패턴이 생긴다. DQN은 현재 시간과 대여소별 자전거 수를 보고 어느 대여소를 먼저 방문해야 장기적으로 수요 실패가 줄어드는지 학습할 수 있다.

### V0 수요 패턴

V0에서는 완전 랜덤 수요가 아니라 **시간대별 범위 랜덤 수요**를 사용한다. 수요가 완전히 랜덤이면 에이전트가 학습할 규칙이 약하고, 반대로 항상 같은 값이면 단순 암기에 가까워질 수 있다. 따라서 시간대별 경향은 유지하되, 매 episode와 매 step마다 약간의 변동성을 둔다.

| 시간대 | 의미 | station 0 주거지역 | station 1 업무지역 | station 2 공원/중간지역 |
|---|---|---:|---:|---:|
| 0-5 | 심야 | 0-1 | 0-1 | 0-1 |
| 6-10 | 아침 | 3-5 | 1-2 | 1-3 |
| 11-16 | 낮 | 1-2 | 2-3 | 2-4 |
| 17-21 | 저녁 | 1-3 | 3-5 | 1-2 |
| 22-23 | 밤 | 0-1 | 1-2 | 0-1 |

예를 들어 `time_step = 8`이면 아침 시간대로 보고 다음과 같이 수요를 샘플링한다.

```text
demand = [
  random integer between 3 and 5,
  random integer between 1 and 2,
  random integer between 1 and 3
]
```

이 설계에서는 station 0이 아침에 자주 부족해지고, station 1은 저녁에 자주 부족해질 가능성이 높다. 따라서 state에 `time_step`을 포함하는 이유가 명확해진다.

### 현재 구현된 V0 상태

현재 저장소에는 V0의 최소 실행 루프가 구현되어 있다.

| 항목 | 현재 구현 |
|---|---|
| 환경 | `ToyDdareungiEnv` |
| API | Gymnasium 스타일 `reset()` / `step()` |
| 관측값 | 정규화된 station stock, truck location, truck load, time step |
| 원본 값 | `info`와 episode log에 station/truck/reward component 저장 |
| 대여소 이름 | `마포구청역`, `여의도역`, `서울숲입구` |
| baseline | Random, Low-stock |
| render mode | `none`, `ansi`, `human` |
| episode log | JSON 저장 가능 |
| 시각화 | terminal tile replay, pygame window replay |

실행 예시는 다음과 같다.

```bash
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy random --episodes 5 --seed 42
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy low-stock --episodes 5 --seed 42
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy low-stock --episodes 1 --render-mode ansi
```

현재 `ansi`는 평가 코드가 환경의 text frame을 받아 출력하고, `human`은 `ToyDdareungiEnv.step()`이 직접 frame을 출력한다. 이후 픽셀/타일 기반 시각화는 학습 코드와 직접 연결하지 않고, 저장된 episode log를 replay하는 방식으로 확장한다.

현재 구현은 V0 baseline을 기준선으로 고정하고, 같은 MDP 위에서 V1 DQN 학습과 greedy 평가를 실행할 수 있는 단계다. 짧은 학습 결과는 기능 확인용 smoke check로 보고, 성능 주장은 held-out seed 묶음에서 baseline과 비교한 뒤에만 한다.

### V0 반납 패턴

초기 V0 설계는 대여 수요만 사용했지만, 실제 구현에서는 toy 환경이 중반 이후 완전히 고갈되는 것을 줄이기 위해 단순한 시간대별 반납 패턴도 함께 사용한다.

처리 순서는 다음과 같다.

```text
1. 트럭이 선택된 대여소로 이동한다.
2. target_stock 기준 자동 재배치를 수행한다.
3. 시간대별 대여 demand를 샘플링하고 적용한다.
4. 시간대별 returns를 샘플링하고 적용한다.
5. unmet_demand, full_returns, movement_cost, reward를 기록한다.
```

현재 reward는 여전히 V0의 단순식을 사용한다.

```text
reward = -10 * unmet_demand - movement_cost
```

`full_returns`는 현재 `info`와 episode log에 기록하지만, reward에는 아직 직접 반영하지 않는다. 이는 V1에서 반납 실패 비용을 추가하기 위한 준비 값이다.

## Agent, Environment, State, Action, Reward

강화학습 문제는 에이전트가 환경을 관찰하고, 행동을 선택하고, 그 결과로 보상을 받는 구조로 볼 수 있다.

```text
Agent(DQN)
  -> action 선택
  -> Environment(ToyDdareungiEnv)
  -> state 변화와 reward 반환
  -> Agent가 경험을 학습
```

V0에서 각 구성 요소는 다음처럼 정의한다.

| 구성 요소 | 이 프로젝트에서의 의미 |
|---|---|
| Agent | 트럭을 어느 대여소로 보낼지 결정하는 DQN 모델 |
| Environment | 대여소 재고, 트럭 위치, 자동 재배치, 수요 발생, reward 계산을 관리하는 시뮬레이션 환경 |
| State | 현재 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 시간 |
| Action | 트럭이 방문할 대여소 선택 |
| Reward | 수요 실패를 줄이고 불필요한 이동을 줄이도록 주는 점수 |

### Agent

V0의 에이전트는 사람이 아니라 DQN 모델이다. 에이전트는 현재 state를 입력으로 받고, 가능한 action 각각의 Q-value를 예측한 뒤 하나의 action을 선택한다.

```text
state
  -> DQN
  -> Q(go_station_0), Q(go_station_1), Q(go_station_2)
  -> action 선택
```

예를 들어 DQN이 아래와 같은 Q-value를 예측했다고 하면:

```text
go_station_0: 1.2
go_station_1: 3.8
go_station_2: 2.1
```

탐험을 하지 않는 평가 상황에서는 가장 값이 큰 `go_station_1`을 선택한다.

### Environment

V0의 환경은 작은 따릉이 재배치 시뮬레이터다. 환경은 에이전트가 선택한 action을 받아서 트럭 이동, 자동 재배치, 수요 발생, reward 계산을 순서대로 처리한다.

환경이 관리하는 내부 값:

| 값 | 의미 |
|---|---|
| `station_bikes` | 각 대여소의 현재 자전거 수 |
| `truck_location` | 트럭이 현재 위치한 대여소 |
| `truck_bikes` | 트럭에 실린 자전거 수 |
| `time_step` | 현재 시간 단계 |
| `demand_pattern` | 시간대별 수요 발생 범위 |

환경의 역할:

```text
1. 현재 state를 에이전트에게 제공한다.
2. 에이전트가 선택한 action을 받는다.
3. 트럭 위치를 선택된 대여소로 변경한다.
4. target_stock 기준으로 자동 재배치를 수행한다.
5. 현재 시간대에 맞는 수요를 생성한다.
6. 대여 성공/실패를 계산한다.
7. reward와 next_state를 반환한다.
8. time_step이 24에 도달하면 episode를 종료한다.
```

즉, V0에서 환경은 단순한 데이터 저장소가 아니라, 강화학습의 규칙을 정의하는 핵심 요소다. 에이전트는 환경 내부 규칙을 직접 알지 못하고, state와 reward를 통해 어떤 행동이 좋은지 학습한다.

### State

DQN이 보는 상태는 다음과 같은 숫자 벡터로 정의한다.

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

| 요소 | 의미 | 예시 값 |
|---|---|---|
| `station_i_bikes` | i번 대여소의 현재 자전거 수 | `0`부터 `10` |
| `truck_location` | 현재 트럭 위치 | `0`, `1`, `2` |
| `truck_bikes` | 트럭에 실린 자전거 수 | `0`부터 `5` |
| `time_step` | 현재 시간 단계 | `0`부터 `23` |

예를 들어 아래 상태는 0번 대여소에 3대, 1번 대여소에 0대, 2번 대여소에 5대가 있고, 트럭은 1번 대여소에 있으며 트럭에는 2대가 실려 있고, 현재 시간이 7번째 step이라는 뜻이다.

```text
[3, 0, 5, 1, 2, 7]
```

구현 시에는 학습 안정성을 위해 자전거 수, 위치, 시간 값을 `0~1` 범위로 정규화할 수 있다.

### Action

V0에서는 action space를 작게 유지한다.

| 번호 | 행동 | 의미 |
|---|---|---|
| `0` | `go_station_0` | 0번 대여소로 이동 |
| `1` | `go_station_1` | 1번 대여소로 이동 |
| `2` | `go_station_2` | 2번 대여소로 이동 |

즉, 에이전트는 트럭이 어느 대여소로 갈지만 결정한다. 도착 후 자전거를 몇 대 싣거나 내릴지는 환경이 자동으로 결정한다.

V0의 action은 재배치 수량 결정이 아니라 **방문할 대여소 선택**이다. 적재와 하차 수량은 환경의 rule-based heuristic에 위임한다. 따라서 V0에서 DQN이 학습하는 것은 "몇 대를 옮길 것인가"가 아니라, 현재 재고와 시간대별 수요 패턴을 바탕으로 "어느 대여소를 먼저 방문할 것인가"이다.

### 자동 재배치 규칙

대여소마다 목표 보유량을 정한다.

```text
target_stock = 5
```

트럭이 어떤 대여소에 도착했을 때:

```text
if station_bikes > target_stock:
    가능한 만큼 트럭에 싣는다.

if station_bikes < target_stock:
    가능한 만큼 트럭에서 내린다.
```

예를 들어 대여소에 자전거가 8대 있고 목표 보유량이 5대라면, 트럭의 남은 공간만큼 자전거를 싣는다. 반대로 대여소에 자전거가 2대 있고 트럭에 자전거가 있다면, 목표 보유량에 가까워지도록 자전거를 내려준다.

이 규칙을 사용하는 이유는 첫 번째 실험에서 action space를 단순하게 유지하기 위해서다. V0의 목표는 "몇 대를 옮길 것인가"보다 "어느 대여소를 방문할 것인가"를 학습하는 것이다.

이 단순화는 V0의 중요한 가정이자 한계다. 실제 따릉이 재배치에서는 방문 위치와 수량을 모두 결정해야 하지만, 첫 실험에서는 DQN 학습이 가능한 작은 문제를 만들기 위해 수량 결정을 환경에 맡긴다. 이후 확장 버전에서는 action을 `(방문 대여소, 싣기/내리기 수량)` 형태로 넓힐 수 있다.

### Reward

V0의 reward는 수요 실패를 줄이는 방향으로 단순하게 정의한다.

```text
reward = -10 * unmet_demand - movement_cost
```

| 요소 | 의미 |
|---|---|
| `unmet_demand` | 자전거가 부족해서 대여하지 못한 사용자 수 |
| `movement_cost` | 트럭 이동에 따른 비용 |

첫 실험에서는 다음처럼 단순한 값을 사용할 수 있다.

| 상황 | 값 |
|---|---|
| 대여 실패 1명 발생 | `-10` |
| 트럭이 다른 대여소로 이동 | `-1` |

예를 들어 어떤 step에서 전체 수요가 6명인데 실제로 4명만 자전거를 빌릴 수 있었고, 트럭이 다른 대여소로 이동했다면:

```text
unmet_demand = 2
movement_cost = 1
reward = -10 * 2 - 1
       = -21
```

반대로 어떤 step에서 트럭이 이동했고, 수요 3명을 모두 만족했다면:

```text
unmet_demand = 0
movement_cost = 1
reward = -10 * 0 - 1
       = -1
```

현재 트럭 위치와 에이전트가 선택한 대여소가 같아서 실제 이동이 없고, 수요도 모두 만족했다면:

```text
unmet_demand = 0
movement_cost = 0
reward = -10 * 0 - 0
       = 0
```

이 환경에서는 reward가 대부분 음수로 나올 수 있다. 중요한 것은 reward가 양수가 되는지가 아니라, 학습이 진행되면서 episode reward가 덜 음수로 개선되는지다.

## 행동 예시

아래 예시는 V0 환경에서 에이전트의 action이 실제로 어떤 변화를 만드는지 보여준다.

### 예시 1: 자전거가 부족한 대여소에 방문

현재 상태:

```text
station_bikes = [3, 8, 5]
truck_location = 1
truck_bikes = 4
time_step = 7
target_stock = 5
```

DQN이 선택한 행동:

```text
action = 0
```

의미:

```text
트럭을 0번 대여소로 보낸다.
```

환경 처리:

```text
0번 대여소 현재 자전거 수 = 3
목표 보유량 = 5
트럭 적재량 = 4

0번 대여소는 목표보다 2대 부족하다.
트럭에 자전거가 충분히 있으므로 2대를 내린다.

station_bikes = [5, 8, 5]
truck_bikes = 2
truck_location = 0
```

그 다음 수요가 발생한다.

```text
demand = [4, 1, 1]
```

수요 처리 후:

```text
0번 대여소: 5대 보유, 수요 4명 -> 모두 대여 성공, 남은 자전거 1대
1번 대여소: 8대 보유, 수요 1명 -> 성공, 남은 자전거 7대
2번 대여소: 5대 보유, 수요 1명 -> 성공, 남은 자전거 4대

unmet_demand = 0
```

reward:

```text
movement_cost = 1
reward = -10 * 0 - 1
       = -1
```

해석:

```text
트럭 이동 비용은 있었지만, 수요 실패를 막았기 때문에 나쁘지 않은 행동이다.
```

### 예시 2: 자전거가 많은 대여소에서 싣기

현재 상태:

```text
station_bikes = [2, 9, 4]
truck_location = 0
truck_bikes = 1
time_step = 13
target_stock = 5
```

DQN이 선택한 행동:

```text
action = 1
```

환경 처리:

```text
1번 대여소 현재 자전거 수 = 9
목표 보유량 = 5
트럭 현재 적재량 = 1
트럭 최대 용량 = 5

1번 대여소는 목표보다 4대 많다.
트럭에는 4대까지 더 실을 수 있다.
따라서 4대를 싣는다.

station_bikes = [2, 5, 4]
truck_bikes = 5
truck_location = 1
```

그 다음 수요가 발생한다.

```text
demand = [3, 1, 1]
```

수요 처리 후:

```text
0번 대여소: 2대 보유, 수요 3명 -> 1명 실패
1번 대여소: 5대 보유, 수요 1명 -> 성공
2번 대여소: 4대 보유, 수요 1명 -> 성공

unmet_demand = 1
```

reward:

```text
movement_cost = 1
reward = -10 * 1 - 1
       = -11
```

해석:

```text
트럭에 자전거를 가득 실은 것은 이후 step에 도움이 될 수 있다.
하지만 이번 step에서는 0번 대여소의 부족을 해결하지 못해 패널티를 받는다.
```

### 예시 3: 잘못된 방문 선택

현재 상태:

```text
station_bikes = [1, 7, 6]
truck_location = 2
truck_bikes = 3
time_step = 8
target_stock = 5
```

아침 시간대라 0번 주거지역 수요가 높다고 가정한다.

DQN이 선택한 행동:

```text
action = 2
```

의미:

```text
트럭이 이미 있는 2번 대여소에 머문다.
```

환경 처리:

```text
2번 대여소 현재 자전거 수 = 6
목표 보유량 = 5
트럭 현재 적재량 = 3

2번 대여소는 목표보다 1대 많다.
트럭에 1대를 싣는다.

station_bikes = [1, 7, 5]
truck_bikes = 4
truck_location = 2
```

그 다음 수요가 발생한다.

```text
demand = [4, 1, 1]
```

수요 처리 후:

```text
0번 대여소: 1대 보유, 수요 4명 -> 3명 실패
1번 대여소: 7대 보유, 수요 1명 -> 성공
2번 대여소: 5대 보유, 수요 1명 -> 성공

unmet_demand = 3
```

reward:

```text
movement_cost = 0
reward = -10 * 3 - 0
       = -30
```

해석:

```text
이동 비용은 없었지만, 수요가 높은 0번 대여소를 방치했기 때문에 큰 패널티를 받는다.
학습이 잘 되면 DQN은 이런 상황에서 0번 대여소를 방문하는 행동의 Q-value를 더 높게 예측하게 된다.
```

## Step 진행 흐름

한 step은 다음 순서로 진행한다.

```text
1. agent가 현재 state를 관찰한다.
2. DQN이 action을 선택한다.
3. 트럭이 선택된 대여소로 이동한다.
4. 환경이 자동 재배치 규칙에 따라 자전거를 싣거나 내린다.
5. 시간대별 수요가 발생한다.
6. 자전거가 부족한 경우 unmet_demand를 계산한다.
7. reward를 계산한다.
8. 다음 state를 반환한다.
```

이 구조에서 에이전트는 "현재 부족한 곳"만 보는 것이 아니라, 시간대별 수요 패턴을 고려해 앞으로 부족해질 가능성이 있는 대여소를 미리 방문하는 정책을 학습하는 것이 목표다.

### Episode log schema

현재 평가 코드는 첫 episode log를 JSON으로 저장할 수 있다.

```bash
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate \
  --policy low-stock \
  --episodes 3 \
  --save-log outputs/low_stock_episode.json
```

reset record:

```text
event
state
info
```

step record:

```text
event
state
action
reward
next_state
terminated
truncated
info
```

`info`에는 분석과 시각화에 필요한 값을 함께 저장한다.

```text
time_step
station_names
previous_truck_location
truck_previous_location
truck_location
truck_bikes
station_bikes
previous_station_bikes
previous_truck_bikes
action
demand
returns
served_demand
unmet_demand
accepted_returns
full_returns
movement_cost
service_success
relocation_delta
rebalance_type
rebalance_station
rebalance_amount
truck_event
truck_event_amount
station_bikes_before_rebalance
station_bikes_after_rebalance
truck_bikes_before_rebalance
truck_bikes_after_rebalance
reward
policy_name
learning_stage
episode_reward_so_far
episode_served_demand_so_far
episode_unmet_demand_so_far
episode_total_demand_so_far
service_rate_so_far
episode_full_returns_so_far
episode_movement_cost_so_far
reward_formula
```

이 log를 사용하면 학습/평가 코드와 렌더링 코드를 분리하면서도, 같은 episode를 콘솔 또는 픽셀/타일맵 시각화로 다시 재생할 수 있다.

pygame 창 기반 replay는 같은 log를 읽어 FrozenLake 스타일 격자 화면으로 보여준다.

```bash
ddareungi-demo
ddareungi-replay-window outputs/low_stock_episode.json --max-steps 10
```

창 replay에서는 `previous_truck_location -> truck_location`으로 트럭 이동을 보간하고, `rebalance_type`, `rebalance_amount`로 자전거 싣기/내리기 이벤트를 표시한다. 오른쪽 패널에는 하루 목표인 `헛걸음 줄이기`, 현재 policy, `DQN 학습 전 기준 정책` label, 누적 점수, 누적 헛걸음, 누적 이동비용을 간결하게 표시한다. 여기서 `헛걸음`은 `unmet_demand`를 발표용으로 바꾼 표현이며, 트럭의 비효율적인 이동은 `누적 이동비용`으로 분리해서 본다. 대여소 card는 `안정`, `부족 주의`, `헛걸음 위험` 상태를 색상, label, 얼굴 표정으로 함께 보여준다. 얼굴 표정은 정책의 감정이 아니라 사용자의 대여 경험 상태를 빠르게 읽기 위한 보조 신호다. episode가 끝나면 `우수`, `양호`, `주의`, `개선 필요` 중 하나로 하루 운영 결과를 보여준다.

현재 V0 replay는 학습 중 화면이 아니라 baseline 정책의 행동을 설명하는 화면이다. DQN을 추가한 뒤에는 replay 화면은 한 episode의 행동을 보여주고, 별도 학습 화면에서 episode reward moving average, epsilon, loss, baseline 대비 개선율을 보여준다.

### Episode 종료 조건

V0는 명시적인 성공 또는 실패 terminal state를 두지 않는다. 한 episode는 하루를 단순화한 24 step으로 고정한다.

Gymnasium 스타일로 구현할 경우 종료 조건은 다음처럼 정의한다.

```text
terminated = False
truncated = time_step >= 24
```

즉, 자전거가 모두 없어지거나 특정 대여소가 비어도 episode를 바로 끝내지 않는다. 그런 상황은 학습해야 할 운영 상태의 일부로 보고, 24 step이 모두 끝났을 때만 episode를 종료한다.

## DQN 학습 구조

V0에서 사용할 DQN은 다음 흐름을 가진다.

```text
state
  -> main Q-network
  -> 각 action의 Q-value 예측
  -> epsilon-greedy로 action 선택
  -> env.step(action)
  -> reward, next_state 저장
  -> replay buffer에서 mini-batch sampling
  -> target network로 target Q-value 계산
  -> main network 업데이트
```

현재 V1 구현은 외부 딥러닝 의존성을 추가하지 않고, 순수 Python으로 작성한 작은 one-hidden-layer MLP Q-network를 사용한다. Python 3.14 환경에서 PyTorch 설치 리스크를 줄이고, toy MDP의 학습 루프를 먼저 검증하기 위한 선택이다. 구조는 DQN의 핵심 요소인 replay buffer, target network, epsilon-greedy exploration, greedy evaluation을 유지한다.

현재 네트워크 구조:

| 항목 | 내용 |
|---|---|
| 입력 | 6차원 state |
| 출력 | 3개 action의 Q-value |
| hidden layer | 기본 32 units, ReLU |
| 구조 | Linear -> ReLU -> Linear |
| 저장 형식 | JSON model file |

V1에서 학습과 평가는 다음 명령으로 분리한다.

```bash
ddareungi-train-dqn --episodes 300 --seed 42 --model-out outputs/models/dqn_v1.json
ddareungi-evaluate --policy dqn --model-path outputs/models/dqn_v1.json --episodes 20 --seed 1000
```

학습 중에는 epsilon-greedy로 탐험하지만, 평가 시에는 저장된 모델을 불러와 greedy action만 사용한다. 따라서 DQN 성능은 학습 중 episode reward가 아니라, baseline과 동일한 seed 묶음에서 별도로 실행한 evaluation 결과로 비교한다.

## Baseline

DQN의 성능을 해석하려면 비교 대상이 필요하다.

V0에서는 최소한 다음 baseline을 둘 수 있다.

| Policy | 설명 |
|---|---|
| Random policy | 매 step마다 무작위 대여소를 선택 |
| Low-stock policy | 현재 자전거 수가 가장 적은 대여소를 선택 |

Random policy는 DQN이 최소한 무작위보다 나은지 확인하기 위한 기준이다. Low-stock policy는 사람이 생각할 수 있는 단순한 휴리스틱과 비교하기 위한 기준이다.

## 단계별 개발 계획

### V0: Toy Environment and Baseline Evaluation

목표는 가장 작은 강화학습 환경을 실행 가능하게 만들고, DQN 학습 전에 baseline 평가 루프를 완성하는 것이다.

| 항목 | 내용 |
|---|---|
| 대여소 | 3개 가상 대여소 |
| 수요 | 직접 만든 시간대별 수요 패턴 |
| action | 방문할 대여소 선택 |
| reward | 미충족 수요 패널티 + 이동 비용 |
| 알고리즘 | Random, Low-stock baseline 우선. DQN은 다음 milestone |
| 비교 | Random policy, Low-stock policy |
| 결과 | reward curve, unmet demand 비교 |

### V1: DQN 학습과 평가

목표는 같은 `ToyDdareungiEnv`에서 DQN을 학습하고, 저장된 모델을 불러와 Random 및 Low-stock baseline과 같은 지표로 비교하는 것이다.

현재 CLI:

```text
ddareungi-train-dqn
ddareungi-evaluate --policy dqn --model-path ...
```

이 단계에서는 환경과 reward를 V0와 동일하게 유지한다. 환경을 바꾸기 전에 DQN 학습, 저장, 로드, greedy 평가가 baseline 비교 루프와 연결되는지 먼저 검증한다.

V1에서 아직 주장하지 않는 것:

- DQN이 항상 baseline보다 우수하다는 주장
- 실제 따릉이 운영 최적화
- Double DQN 또는 Dueling DQN의 우수성
- 재배치 수량까지 학습했다는 주장

### V2: Double DQN 확장

목표는 DQN과 Double DQN을 비교하는 것이다.

| 알고리즘 | 차이점 |
|---|---|
| DQN | target Q 계산 시 target network에서 max action과 value를 모두 선택 |
| Double DQN | main network로 max action을 선택하고 target network로 해당 action의 value를 계산 |

비교 지표:

- episode reward
- unmet demand
- 학습 안정성
- seed별 결과 차이

### V3: Dueling DQN 확장

목표는 DQN, Double DQN, Dueling DQN을 비교하는 것이다.

Dueling DQN은 Q-value를 직접 하나의 흐름으로만 예측하지 않고, state 자체의 가치인 `V(s)`와 action별 이점인 `A(s, a)`를 나누어 추정한다.

```text
Q(s, a) = V(s) + (A(s, a) - mean(A(s, *)))
```

이 구조는 어떤 상태가 전반적으로 좋은 상태인지와, 그 상태에서 어떤 action이 상대적으로 더 좋은지를 분리해 볼 수 있다는 장점이 있다. V0 환경처럼 action 수가 작더라도, DQN 계열 알고리즘의 구조적 차이를 설명하고 비교하기 좋다.

비교 지표:

- episode reward
- unmet demand
- 학습 안정성
- DQN, Double DQN, Dueling DQN reward curve 비교

### V4: 공공데이터 기반 환경 확장

V4에서는 V0의 작은 환경을 유지하되, 일부 값만 실제 서울 공공데이터에서 가져와 현실성을 높인다. 처음부터 서울 전체 대여소를 모두 사용하는 것은 범위가 너무 크기 때문에, 특정 지역의 3개에서 5개 대여소만 선택한다.

사용할 공공데이터 후보:

| 데이터 | 포함 내용 | 프로젝트에서 사용할 값 |
|---|---|---|
| 서울시 공공자전거 대여소 정보 | 대여소 번호, 대여소명, 위도, 경도, 거치대 수 | 실제 대여소 목록, 대여소 위치, 대여소 capacity |
| 서울시 공공자전거 대여이력 정보 | 대여 일시, 대여 대여소, 반납 일시, 반납 대여소 | 시간대별 대여 수요, 시간대별 반납량 |
| 서울시 공공자전거 실시간 대여정보 | 대여소별 현재 대여 가능 자전거 수, 거치율, 위치 | episode 시작 시 초기 자전거 수 |

데이터 반영 방식:

```text
1. 특정 지역의 대여소 3개에서 5개를 선택한다.
2. 대여소 정보 데이터에서 각 대여소의 capacity와 위치를 가져온다.
3. 대여이력 데이터에서 시간대별 대여 건수를 집계한다.
4. 대여이력 데이터에서 시간대별 반납 건수를 집계한다.
5. 집계 결과를 바탕으로 V0의 demand pattern을 실제 패턴으로 교체한다.
6. 실시간 대여정보 API를 사용할 수 있으면 초기 station_bikes 값을 실제 값으로 설정한다.
```

예상 전처리 결과:

```text
station_id, hour, rental_count, return_count
101, 8, 12, 3
101, 9, 9, 5
102, 8, 2, 10
102, 9, 4, 8
```

이 데이터는 환경에서 다음처럼 사용한다.

```text
demand[station_i][hour] = rental_count
arrival[station_i][hour] = return_count
```

V0에서는 수요만 단순하게 사용하지만, V4에서는 반납량도 반영할 수 있다.

```text
다음 자전거 수 =
  현재 자전거 수
  - 실제 대여 수요
  + 실제 반납량
  + 트럭 재배치량
```

이렇게 하면 최종 프로젝트는 완전한 실서비스 최적화는 아니지만, 실제 따릉이 데이터의 시간대별 흐름을 반영한 강화학습 시뮬레이터가 된다.

### Final: 보고서와 발표용 결과 정리

최종 결과물은 다음을 포함한다.

- 프로젝트 배경과 관련 연구 요약
- MDP 정의: Agent, Environment, State, Action, Reward
- DQN 구조 설명
- DQN과 baseline 비교
- DQN, Double DQN, Dueling DQN 비교
- reward curve 및 unmet demand 그래프
- 한계점과 향후 확장 방향

## 현재 결정된 설계

| 항목 | 결정 |
|---|---|
| 첫 알고리즘 | DQN |
| 첫 환경 | 3개 대여소, 트럭 1대 |
| 첫 action | 방문할 대여소 선택 |
| 싣기/내리기 | 환경의 자동 재배치 규칙 |
| 첫 reward | 미충족 수요 패널티 + 이동 비용 + 수요 만족 보상 |
| episode 종료 | 24 step 고정, `truncated=True` |
| V0 수요 패턴 | 시간대별 범위 랜덤 |
| 확장 알고리즘 | Double DQN, Dueling DQN |
| 실제 데이터 | V4 이후 일부 적용 |

## 이후 논의할 내용

- V0 이후 이동 비용을 단순히 `0` 또는 `1`로 유지할지, 대여소 간 거리 행렬로 확장할지
- V0 이후 자동 재배치의 목표 보유량을 모든 대여소에서 `5`로 동일하게 둘지, 대여소별로 다르게 둘지
- V0 이후 reward에 반납 실패나 과잉 재고 패널티를 포함할지
- 최종 보고서에서 실제 따릉이 데이터를 어느 수준까지 반영할지
