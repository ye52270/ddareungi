# 따릉이 RL Simple

이 저장소는 따릉이 재배치 문제를 **내가 설명할 수 있는 최소 강화학습 프로젝트**로 다시 정리한 버전이다.

목표는 많은 기능을 넣는 것이 아니라, 아래 흐름을 정확히 이해하는 것이다.

```text
환경 만들기 -> baseline 평가 -> DQN 학습 -> baseline과 비교
```

## 문제

대여소마다 시간대별로 자전거가 부족하거나 남을 수 있다. 트럭은 한 번에 한 대여소를 방문해서 자전거를 싣거나 내린다.

이 프로젝트의 질문은 하나다.

> 현재 재고, 트럭 위치, 시간 정보를 보고 다음에 방문할 대여소를 잘 고르면 헛걸음을 줄일 수 있을까?

## MDP 정의

| 요소 | 이 프로젝트의 정의 |
|---|---|
| State | 대여소별 자전거 수, 트럭 위치, 트럭 적재량, 시간 |
| Action | 다음에 방문할 대여소 선택 |
| Reward | `-10 * unmet_demand - movement_cost` |
| Environment | 대여소 3개, 트럭 1대, 하루 24 step |
| Goal | 자전거가 없어 빌리지 못하는 수요를 줄이기 |

## 남긴 파일

```text
src/ddareungi_rl/
  env.py           # MDP 환경
  baselines.py     # random, low-stock, demand-aware
  dqn.py           # PyTorch DQN
  data_profile.py  # 실제 데이터 profile 읽기
  cli.py           # 실행 메뉴
```

시각화, PPT, 여러 DQN 변형, 복잡한 데이터 전처리 코드는 일단 제거했다. 필요하면 나중에 하나씩 다시 붙인다.

## 실행

설치:

```bash
python -m pip install -e ".[dev]"
```

메뉴 실행:

```bash
ddareungi
```

메뉴:

```text
1. Toy baseline 평가
2. Toy DQN 학습/평가
3. Real-profile baseline 평가
4. Real-profile DQN 학습/평가
0. 종료
```

`3`, `4`번은 `outputs/data/magok_3station_profile.json`이 있을 때 실제 따릉이 profile을 사용한다. 파일이 없으면 기본 toy 환경으로 실행된다.

## 지금 버전의 의도

이 버전은 최종 완성본이 아니다. 다시 시작하기 위한 깨끗한 출발점이다.

앞으로는 다음 순서로만 확장한다.

1. 현재 코드 이해
2. reward와 state 설명 정리
3. DQN 학습 결과 확인
4. 실제 데이터 profile을 다시 천천히 연결
5. 필요할 때만 시각화 추가
