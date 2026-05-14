# 조원 리뷰 브리프

## 오늘 리뷰 목표

현재 프로젝트는 실제 서울 전체 따릉이 시스템을 바로 재현하기보다, 강화학습 문제로 해석 가능한 최소 toy MDP를 먼저 만든 상태다. 조원 리뷰에서는 화면의 완성도보다 `State`, `Action`, `Reward`, `Environment`, baseline 비교 방식이 타당한지 확인하는 것이 핵심이다.

## 한 문장 요약

따릉이 재배치 트럭이 시간대별 수요와 현재 재고를 보고 다음에 방문할 대여소를 선택해, 자전거 부족으로 인한 헛걸음을 줄이는 정책을 학습하는 프로젝트다.

## 현재 프로젝트 구조

```text
src/ddareungi_rl/
  envs/toy_ddareungi_env.py      Toy MDP 환경
  policies/baselines.py          Random, Low-stock baseline
  agents/dqn.py                  순수 Python toy DQN
  training/train_dqn.py          DQN 학습
  training/evaluate.py           baseline/DQN 평가
  visualization/pygame_replay.py episode log GUI replay
  cli.py                         실습용 메뉴
```

## 현재 MDP 정의

| 항목 | 현재 정의 | 리뷰 포인트 |
|---|---|---|
| State | 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 시간 | 학습에 필요한 정보가 충분한가 |
| Action | 다음에 방문할 대여소 선택 `{0, 1, 2}` | 처음 toy 문제로 적절한 단순화인가 |
| Reward | `-10 * unmet_demand - movement_cost` | 헛걸음 감소 목표와 일치하는가 |
| Environment | 3개 대여소, 트럭 1대, 24 step 하루 운영 | 과제 범위에서 설명 가능한가 |
| Policy | Random, Low-stock, DQN greedy | baseline 비교가 공정한가 |

## 학습 방법

현재 DQN은 PyTorch가 아니라 순수 Python으로 구현한 작은 one-hidden-layer Q-network다. 핵심 구성은 다음과 같다.

- Q-network로 `Q(s, a)` 근사
- replay buffer에 transition 저장
- target network로 Bellman target 계산 안정화
- epsilon-greedy로 exploration 수행
- 학습 후 greedy action으로 held-out seed 평가

현재 단계에서는 고성능 DQN을 주장하기보다, DQN 학습 루프가 toy MDP에 연결되고 baseline과 같은 조건에서 평가될 수 있음을 보이는 것이 목표다.

## 실행 시나리오

```bash
ddareungi
```

메뉴에서 확인할 흐름은 다음과 같다.

| 메뉴 | 용도 |
|---|---|
| 1 | Random/Low-stock baseline 평가 |
| 2 | DQN(Small) 학습 |
| 3 | 저장된 DQN(Small) greedy 평가 |
| 4 | baseline 평가 후 visualization |
| 5 | DQN 평가 후 visualization |

4번/5번 visualization은 창을 닫으면 프로그램도 종료된다.

## 현재까지의 강점

- 강화학습 구성요소가 코드와 README에 명확히 분리되어 있다.
- baseline, DQN 학습, DQN 평가, visualization이 같은 toy MDP 위에서 실행된다.
- episode log 기반 replay라서 학습 코드와 시각화 코드가 분리되어 있다.
- 테스트가 있어 메뉴와 GUI 종료 흐름을 확인할 수 있다.

## 오늘 조원에게 물어볼 질문

1. Reward에서 헛걸음 패널티 `-10`과 이동비용 `-1`의 비율이 직관적인가?
2. action을 “방문 대여소 선택”으로 제한한 것이 V0 toy 문제로 적절한가?
3. DQN 성능 비교는 어떤 지표를 중심으로 보여주는 것이 가장 설득력 있는가?
4. 다음 단계에서 PyTorch DQN으로 넘어갈 때 기존 순수 Python DQN을 유지할지, 교체할지?
5. 실제 따릉이 데이터로 확장하기 전에 reward에 반납 실패도 넣어야 하는가?

## PPT 또는 Word로 만들 수 있는 구성

PPT로 만든다면 6장 구성이 적절하다.

| Slide | 제목 | 핵심 내용 |
|---|---|---|
| 1 | 문제 정의 | 따릉이 재고 불균형과 헛걸음 문제 |
| 2 | MDP 설계 | State, Action, Reward, Environment |
| 3 | 현재 구현 | env, baseline, DQN, evaluation, replay 구조 |
| 4 | 학습 방법 | Q-network, replay buffer, target network |
| 5 | 실행 화면 | 메뉴와 visualization 설명 |
| 6 | 한계와 다음 단계 | PyTorch DQN, multi-seed 평가, 실제 데이터 |

Word 문서로 만든다면 README를 줄인 2~3쪽 리뷰 노트가 적절하다. 현재 `README.md`와 이 브리프를 바탕으로 PPTX나 DOCX를 바로 생성할 수 있다.
