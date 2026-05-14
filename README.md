# DQN 기반 따릉이 재배치 MDP

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Status](https://img.shields.io/badge/status-V0%20MDP%20baseline-2E8B57)
![Test](https://img.shields.io/badge/tests-pytest%20passing-4C9A2A)
![RL](https://img.shields.io/badge/RL-DQN%20planned-F28C28)

이 프로젝트는 서울 공공자전거 **따릉이 재배치 문제**를 작은 강화학습 환경으로 단순화하고, DQN이 시간대별 수요 패턴을 이용해 **트럭의 방문 순서 정책**을 학습할 수 있는지 검증하는 기말 프로젝트다.

> [!IMPORTANT]
> 현재 저장소는 **DQN 학습 전 V0 단계**다. V0의 목적은 현실 전체를 재현하는 것이 아니라, `State`, `Action`, `Reward`, `Environment`가 명확한 toy MDP와 baseline 평가 루프를 먼저 검증하는 것이다.

<ins>핵심 질문</ins>

> 작은 MDP에서 DQN이 시간대별 수요 패턴과 현재 재고를 보고, Random 및 Low-stock heuristic보다 평균 헛걸음(`unmet_demand`)을 줄이면서 이동비용을 관리하는 방문 순서 정책을 학습할 수 있는가?

## 왜 따릉이 재배치를 강화학습으로 보는가

따릉이 운영에서는 대여소마다 시간대별 대여/반납 패턴이 다르다. 어떤 대여소는 출근 시간에 빠르게 비고, 다른 대여소는 반납이 몰려 가득 찰 수 있다. 운영 트럭은 현재 재고, 트럭 위치, 시간대 수요를 보고 다음에 어느 대여소를 방문할지 결정해야 한다.

이 문제는 다음 이유로 강화학습의 **순차적 의사결정 문제**로 볼 수 있다.

| 관점 | 따릉이 재배치에서의 의미 |
|---|---|
| 현재 상태가 중요함 | 대여소별 재고, 트럭 위치, 트럭 적재량, 시간대가 다음 결정의 입력이 된다. |
| 행동이 다음 상태를 바꿈 | 트럭이 어느 대여소를 방문하느냐에 따라 재고와 이후 헛걸음이 달라진다. |
| 단기 비용과 장기 성과가 충돌함 | 가까운 곳만 방문하면 이동비용은 줄지만, 수요가 큰 대여소의 헛걸음이 늘 수 있다. |
| 정책 비교가 가능함 | Random, Low-stock heuristic, DQN을 같은 환경과 seed 묶음에서 비교할 수 있다. |

따라서 이 프로젝트는 따릉이 재배치를 Markov Decision Process(MDP)로 축소한다. 즉, 에이전트는 현재 상태 `s`를 보고 행동 `a`를 선택하고, 환경은 다음 상태 `s'`와 보상 `r`을 반환한다.

## 학술적 근거

공공자전거 재배치 문제는 기존 연구에서도 시간/공간 수요 불균형을 줄이기 위한 순차적 의사결정 문제로 다루어진다.

> "empty stations ... lead to severe customer loss"
> Li, Zheng, Yang, **Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach**, KDD 2018

위 관점은 이 프로젝트의 `unmet_demand`, 즉 사용자가 대여소에 왔지만 자전거가 없어 빌리지 못한 **헛걸음** 지표와 직접 연결된다. V0는 실제 Citi Bike나 서울시 전체 데이터를 재현하지 않지만, “빈 대여소가 사용자 손실을 만든다”는 문제의식을 작은 MDP로 옮긴다.

DQN은 Q-learning을 신경망으로 근사해 상태에서 행동가치를 학습하는 방식이다. Mnih et al.은 DQN을 다음과 같이 설명한다.

> "deep Q-network ... can learn successful policies"
> Mnih et al., **Human-level control through deep reinforcement learning**, Nature 2015

이 프로젝트에서 DQN은 이미지 픽셀이 아니라 정규화된 재고/시간/트럭 상태를 입력으로 받고, 가능한 행동 중 다음 방문 대여소를 선택하는 Q-network로 확장될 예정이다.

## V0 MDP 정의

| 요소 | V0 정의 |
|---|---|
| Agent | 다음에 방문할 대여소를 선택하는 정책. 현재는 baseline, V1에서 DQN으로 교체한다. |
| Environment | 대여소 재고, 트럭 이동, 자동 싣기/내리기, 수요/반납, reward 계산을 담당하는 `ToyDdareungiEnv` |
| State | 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 현재 시간 |
| Action | 3개 대여소 중 다음 방문지 선택: `0`, `1`, `2` |
| Transition | 트럭 이동 -> `target_stock=5` 기준 자동 재배치 -> 수요/반납 발생 -> 다음 상태 |
| Reward | `-10 * unmet_demand - movement_cost` |
| Episode | 24 step, 하루 운영을 단순화 |
| 학습 목표 | 헛걸음을 줄이되, 무의미한 이동을 과도하게 늘리지 않는 방문 순서 정책 학습 |

V0에서 가장 중요한 제한은 action space를 작게 유지하는 것이다. DQN은 **몇 대를 싣고 내릴지**를 직접 학습하지 않는다. 싣기/내리기 수량은 환경의 rule-based heuristic이 처리하고, DQN은 **어느 대여소를 먼저 방문할지**를 학습한다.

## State, Action, Reward 해석

### State

V0 state는 “지금 운영 상황이 어떤가?”를 나타낸다.

| State 구성 | 의미 |
|---|---|
| 대여소별 자전거 수 | 각 대여소가 부족/안정/과잉 상태인지 판단하는 핵심 정보 |
| 트럭 위치 | 같은 대여소에 머무를지, 다른 대여소로 이동할지 판단하는 정보 |
| 트럭 적재량 | 트럭이 자전거를 내려줄 수 있는지, 더 실을 여유가 있는지 나타내는 정보 |
| 현재 시간 | 시간대별 수요 패턴을 학습하기 위한 정보 |

### Action

V0 action은 트럭의 다음 방문지다.

```text
0 -> 마포구청역 방문
1 -> 여의도역 방문
2 -> 서울숲입구 방문
```

이 단순화 덕분에 V1에서는 DQN이 “현재 상태에서 어느 대여소를 방문하는 것이 장기적으로 좋은가?”를 먼저 학습할 수 있다.

### Reward

V0 reward는 사용자 헛걸음을 강하게 벌점으로 주고, 트럭 이동에도 작은 비용을 부여한다.

```text
reward = -10 * unmet_demand - movement_cost
```

| Reward 항목 | 의미 |
|---|---|
| `unmet_demand` | 자전거가 부족해서 대여하지 못한 수요. README에서는 **헛걸음**으로 표현한다. |
| `movement_cost` | 트럭이 이전 위치와 다른 대여소로 이동했을 때의 비용 |

이 reward는 “헛걸음을 줄이는 것”을 가장 중요한 목표로 두되, 트럭이 불필요하게 계속 이동하는 정책도 피하도록 만든다.

## 검증 기준

DQN의 성능은 절대 점수가 아니라 **같은 환경, 같은 episode 길이, 같은 seed 묶음**에서 baseline보다 좋아지는지로 평가한다.

| Metric | 의미 | 평가에서의 역할 |
|---|---|---|
| Avg Unmet Demand | 하루 평균 헛걸음 수 | 핵심 성능 지표 |
| Service Rate | 전체 대여 수요 중 처리 성공률 | 사용자 경험 지표 |
| Avg Reward | reward 식으로 계산한 평균 episode 점수 | 종합 비교 지표 |
| Avg Movement Cost | 트럭 이동 횟수/비용 | 운영 효율 지표 |
| Avg Full Returns | 반납 공간 부족으로 처리하지 못한 반납량 | V1 이후 reward 후보 |

평가 프로토콜:

- Random baseline, Low-stock heuristic, DQN을 같은 `ToyDdareungiEnv`에서 비교한다.
- episode 길이는 24 step으로 고정한다.
- 같은 seed 묶음으로 baseline과 DQN을 평가한다.
- DQN 평가는 학습이 끝난 모델을 불러와 탐험 없이 greedy action으로 수행한다.
- 개선 주장은 `Avg Unmet Demand`, `Service Rate`, `Avg Reward`가 함께 좋아질 때만 한다.

## 현재 V0 결과

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

## V0 한계

> [!NOTE]
> V0는 현실 재현보다 **MDP 정의, baseline 평가, replay 시각화**를 먼저 검증하는 단계다.

| 한계 | 현재 처리 | 향후 개선 |
|---|---|---|
| 실제 따릉이 데이터 미사용 | toy demand/return pattern 사용 | 서울 열린데이터 기반 시간대별 수요 구성 |
| 작은 환경 | 대여소 3개, 트럭 1대 | 대여소 수 확장, 클러스터 단위 실험 |
| 단순 action | 다음 방문 대여소만 선택 | 방문지와 싣기/내리기 수량을 함께 선택 |
| 단순 이동비용 | 이동 여부만 비용으로 반영 | 거리/시간 기반 이동비용 |
| 반납 실패 | `full_returns`를 진단 지표로만 기록 | reward에 반납 실패 벌점 추가 |
| 학습 전 단계 | baseline과 replay 구현 | DQN 학습, 저장, 평가, 시각화 |

## Roadmap

| 단계 | 강화학습 관점 | 구현 목표 | 상태 |
|---|---|---|---|
| V0 | MDP가 실행 가능한지 확인 | toy environment, baseline, episode log, replay | 완료 |
| V1 | DQN이 방문 순서 정책을 학습하는지 검증 | DQN train/evaluate, baseline 비교 | 예정 |
| V1.5 | 결과가 seed에 민감한지 확인 | seed 확장 평가, reward ablation | 예정 |
| V2 | DQN 변형 비교 | Double DQN, Dueling DQN 비교 | 예정 |
| V3 | reward 현실성 강화 | 거리 기반 이동비용, 반납 실패 reward 반영 | 예정 |
| V4 | 실제 데이터 연결 | 실제 따릉이 수요 패턴 일부 반영 | 예정 |

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| `ToyDdareungiEnv` | 구현 완료 |
| Random baseline | 구현 완료 |
| Low-stock heuristic baseline | 구현 완료 |
| Episode log 저장 | 구현 완료 |
| `none` / `ansi` / `human` text render | 구현 완료 |
| pygame 창 기반 replay | 구현 완료 |
| 누적 reward / 헛걸음 시각화 | 구현 완료 |
| `pytest` 기반 smoke test | 구현 완료 |
| DQN 학습 | 예정 |
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

설치 없이 바로 실행하려면 다음처럼 `PYTHONPATH`를 지정한다.

```bash
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy random --episodes 5 --seed 42
PYTHONPATH=src python3 -m ddareungi_rl.training.evaluate --policy low-stock --episodes 5 --seed 42
```

## Text Render와 Window Replay

첫 episode를 콘솔 맵으로 확인할 수 있다.

```bash
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode ansi
ddareungi-evaluate --policy low-stock --episodes 1 --render-mode human
```

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

시각화에서 보여주는 정보:

- 대여소별 자전거 수, 수요, 반납
- 트럭 위치와 적재량
- 현재 time step과 하루 진행 bar
- 현재 점수, 누적 점수, 헛걸음 수, 이동비용
- 대여소별 `안정`, `부족 주의`, `헛걸음 위험` 상태
- 대여 경험을 빠르게 읽기 위한 얼굴 표정: 웃음, 무표정, 찡그림, 화남

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
    demo.py
    pixel_replay.py
    pygame_replay.py
tests/
  test_demo.py
  test_pixel_replay.py
  test_pygame_replay.py
  test_toy_ddareungi_env.py
docs/
  01_detailed_design.md
```

자세한 MDP 설계, reward 계산 예시, 단계별 개발 계획은 [docs/01_detailed_design.md](docs/01_detailed_design.md)에 정리되어 있다.

## README 표현 방식

GitHub README에서는 CSS 기반 글자색을 안정적으로 기대하기 어렵다. 대신 이 README는 GitHub가 공식 지원하는 다음 표현을 사용한다.

| 표현 | 사용 위치 | 목적 |
|---|---|---|
| Badge | 문서 상단 | 프로젝트 상태를 색상으로 빠르게 표시 |
| `#`, `##`, `###` heading | 주요 절 제목 | 글자 크기와 구조 강조 |
| `**굵게**` | 핵심 용어 | 중요한 개념 강조 |
| `<ins>밑줄</ins>` | 핵심 질문 | 시선 유도 |
| `> [!IMPORTANT]`, `> [!NOTE]` | V0 상태와 한계 | GitHub alert 색상/아이콘 활용 |
| 표 | MDP, 지표, Roadmap | 비교 구조를 한눈에 표시 |

## 관련 연구

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
- [GitHub Docs: Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
