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

## 관련 연구와 설계 근거

자전거 공유 시스템의 재배치 문제는 시간대와 지역에 따라 대여/반납 수요가 달라져 일부 대여소는 비고, 일부 대여소는 가득 차는 **spatio-temporal imbalance** 문제로 설명된다. 기존 연구들은 이 문제를 줄이기 위해 운영 차량을 어디로 보낼지 결정하고, 그 결과를 **customer loss**, **unmet demand**, **availability failure**, **운영 비용** 같은 지표로 평가한다.

- Li, Zheng, Yang의 [Dynamic Bike Reposition: A Spatio-Temporal Reinforcement Learning Approach](https://www.kdd.org/kdd2018/accepted-papers/view/dynamic-bike-reposition-a-spatio-temporal-reinforcement-learning-approach)는 자전거 재배치를 장기 customer loss를 줄이는 강화학습 문제로 보고, 실제 Citi Bike 데이터 기반 simulator에서 정책을 평가했다.
- Chen et al.의 [Rebalance Bike-Sharing System With Deep Sequential Learning](https://www.microsoft.com/en-us/research/publication/rebalance-bike-sharing-system-with-deep-sequential-learning/)은 자전거 공유 시스템에서 빈 대여소와 가득 찬 대여소가 사용자 경험을 악화시키며, 과거 문제 instance에서 학습한 지식을 미래 재배치에 활용할 수 있다고 설명한다.
- Yin, Kou, Cai의 [DeepBike](https://www.researchgate.net/publication/378732656_DeepBike_A_Deep_Reinforcement_Learning_Based_Model_for_Large-scale_Online_Bike_Share_Rebalancing)는 DQN이 차량 상태, 대여소 상태, 예측 수요를 입력으로 받아 재배치 action의 장기 가치를 추정하도록 설계했다. 우리 프로젝트가 state에 `expected_demand`를 포함한 이유도 이 흐름과 맞닿아 있다.
- Scarpel et al.의 [Fully Dynamic Rebalancing in Dockless Bike-Sharing Systems via Deep Reinforcement Learning](https://arxiv.org/abs/2605.14501)은 단일 트럭이 실시간으로 이동하며 pick-up/drop-off action을 수행하는 문제를 MDP로 구성하고, availability failure 감소를 핵심 성과로 본다.
- Li et al.의 [Dynamic repositioning in bike-sharing systems with uncertain demand](https://www.sciencedirect.com/science/article/abs/pii/S0305048324000148)는 강화학습은 아니지만, 확률적 수요 아래에서 서비스 품질과 운영 비용의 trade-off를 함께 보는 재배치 문제 설정을 보여준다.

따라서 이 프로젝트도 단순히 DQN을 실행하는 것이 아니라, **대여 실패와 반납 실패를 줄이기 위한 MDP를 정의하고 baseline 대비 개선 여부를 비교하는 것**을 목표로 한다.

## MDP 정의

| 요소 | 이 프로젝트의 정의 |
|---|---|
| State | `station_bikes`, `expected_demand`, `truck_location`, `truck_bikes`, `time_step` |
| Action | 다음 step에서 트럭이 방문할 대여소 index 선택 |
| Reward | `-10 * unmet_demand - 3 * rejected_returns - movement_cost` |
| Environment | 대여소 3개, 트럭 1대, 하루 24 step의 따릉이 재배치 simulator |
| Transition | action 적용 -> 트럭 이동/자동 싣기·내리기 -> 대여 수요 처리 -> 반납 수요 처리 -> 다음 시간으로 이동 |
| Goal | 자전거가 없어 빌리지 못하는 수요와 가득 차서 반납하지 못하는 수요를 줄이기 |

`station_bikes`는 각 대여소에 남아 있는 자전거 수이고, `expected_demand`는 현재 날짜와 시간대에서 예상되는 대여 수요다. `truck_location`은 트럭이 현재 있는 대여소, `truck_bikes`는 트럭에 실린 자전거 수, `time_step`은 하루 24시간 중 현재 시점이다.

`unmet_demand`는 자전거가 부족해서 빌리지 못한 수요이고, `rejected_returns`는 대여소가 가득 차 받아주지 못한 반납이다. Reward는 두 종류의 사용자 실패를 모두 벌점화하고, 트럭이 불필요하게 이동하는 것도 작은 비용으로 반영한다.

DQN observation에는 현재 재고뿐 아니라 현재 시간대의 대여소별 예상 수요도 포함한다. 그래야 DQN이 단순히 “지금 어디가 비었는지”만 보는 것이 아니라 “곧 어느 대여소에서 빌리려 하는지”까지 보고 action을 고를 수 있다.

주의: 이 reward는 단순화 이전 버전의 reward와 다르다. 따라서 예전 결과와 직접 비교하지 않고, 같은 reward 기준에서 baseline과 DQN을 다시 비교한다.

## 실험 출력물

논문형 평가를 위해 이 프로젝트는 다음 산출물을 기준으로 결과를 해석한다.

| 출력물 | 목적 |
|---|---|
| Baseline 비교표/그래프 | no-op, random, low-stock, demand-aware 정책 중 현재 환경에서 어떤 규칙이 좋은지 확인 |
| DQN 학습 곡선 | episode가 진행되며 reward와 unmet demand가 개선되는지 확인 |
| DQN vs baseline 비교 | 학습된 DQN이 단순 규칙 정책보다 나은지 같은 날짜 기준으로 비교 |
| Experiment log | episode 수, learning rate, epsilon decay 같은 parameter 변경 이력을 남겨 재현 가능하게 관리 |

현재 핵심 평가지표는 `avg_reward`, `avg_unmet_demand`, `avg_rejected_returns`, `avg_service_rate`다. 이후 보고서 단계에서는 seed별 평균과 표준편차, action distribution, 대표 episode의 재고 변화도 추가해 DQN이 정말 학습했는지 더 분명히 확인할 수 있다.

## 실험 프로토콜

1. 공공 따릉이 CSV에서 선택 대여소의 날짜/시간별 대여·반납 profile을 만든다.
2. 같은 profile과 같은 평가 날짜를 사용해 baseline 정책을 평가한다.
3. 같은 환경에서 PyTorch DQN을 학습한다.
4. 학습된 DQN을 greedy policy로 평가한다.
5. baseline과 DQN의 reward, 미충족 수요, 반납 실패, 서비스율을 비교한다.

## 남긴 파일

```text
src/ddareungi_rl/
  env.py           # MDP 환경
  config_loader.py # YAML/JSON 설정 로더
  baselines.py     # no-op, random, low-stock, demand-aware
  dqn.py           # PyTorch DQN
  data_profile.py  # 실제 데이터 profile 읽기
  profile_builder.py # 공공데이터 CSV에서 profile 생성
  cli.py           # 실행 메뉴

config/
  default_env.yaml # 환경 크기, 초기 재고 범위, reward 계수, 기본 sample data 경로

sample_data/
  toy_demand_return.json # 시간대별 대여/반납 샘플 범위
```

시각화, PPT, 여러 DQN 변형, 복잡한 데이터 전처리 코드는 일단 제거했다. 필요하면 나중에 하나씩 다시 붙인다.

환경 코드와 실험 조건은 분리했다. `env.py`는 MDP 동작만 담당하고, 대여소 이름/초기 재고 범위/트럭 용량/reward 계수/시간대별 수요 샘플은 `config/`와 `sample_data/`에서 읽는다.

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
1. Baseline 평가
2. DQN 학습/평가
3. 데이터 profile 상태/생성 안내
0. 종료
```

`1`, `2`번은 `outputs/data/magok_3station_daily_profile.json`을 우선 사용한다. 이 파일이 없으면 `outputs/data/magok_3station_profile.json`을 사용한다. 둘 다 없으면 메뉴 `3`번에서 profile 생성 명령을 확인한다. Toy 환경은 사용자 메뉴에서 숨기고, 테스트와 개발용으로만 유지한다.

`1`번 baseline 평가가 끝나면 policy별 평균 reward, 미충족 수요, 반납 실패, 서비스율을 비교하는 그래프가 `outputs/figures/baseline_comparison.png`로 저장된다. 이 그래프는 no-op, random, low-stock, demand-aware 중 어떤 정책이 현재 reward 기준에서 좋은지 빠르게 확인하기 위한 용도다.

`2`번 DQN 학습/평가는 1000 episode를 학습한 뒤 baseline과 같은 30개 날짜에서 평가한다. 학습 중에는 50 episode마다 최근 평균 reward와 미충족 수요를 출력하고, 학습 곡선은 `outputs/figures/dqn_training_curve.png`로 저장된다. 이 학습 곡선에는 `low-stock` baseline의 평균 reward 기준선도 함께 표시된다.

현재 DQN 튜닝 기본값은 초반 탐색을 더 오래 유지하도록 설정했다. 주요 값은 `learning_rate=0.0005`, `epsilon_decay=8000`, `replay_size=10000`, `target_update=200`, `hidden_size=128`이다.

DQN 학습이 끝나면 baseline 정책들과 DQN을 같은 평가 기준으로 비교한 그래프도 `outputs/figures/dqn_vs_baseline_comparison.png`로 저장된다.

DQN을 실행할 때마다 parameter와 결과는 `outputs/experiments/dqn_runs.jsonl`에 한 줄씩 누적 저장된다. 각 기록에는 학습 episode 수, gamma, learning rate, epsilon 설정, replay 설정, observation 크기, 평가 episode 수, 평가 reward와 미충족 수요가 들어간다. 나중에 report에서는 이 파일을 읽어 “어떤 parameter를 바꿨고 결과가 어떻게 달라졌는지” 비교표로 정리하면 된다.

## 실제 데이터 profile 만들기

대용량 공공데이터 CSV는 학습 때마다 직접 읽지 않는다. 먼저 1월~12월 대여이력 CSV를 시간대별 profile JSON으로 전처리한 뒤, 메뉴의 `1`, `2`번에서 사용한다.

예시:

```bash
ddareungi-build-profile \
  --rental-dir "data/서울특별시 공공자전거 대여이력 정보_2025" \
  --master-csv "data/서울시 공공자전거 따릉이 대여소 마스터 정보.csv" \
  --station-keyword "마곡" \
  --station-count 3 \
  --output outputs/data/magok_3station_profile.json
```

이 명령은 `tqdm` 진행률을 보여주면서 CSV를 줄 단위로 읽는다. 결과 JSON에는 선택된 대여소, 시간대별 대여 수요 범위, 시간대별 반납 범위, 사용한 날짜 수와 scale 정보가 저장된다.

공공데이터의 실제 건수는 현재 toy 환경의 대여소 용량보다 훨씬 클 수 있다. 그래서 기본값은 실제 시간대 패턴을 유지하되 `max_sample_high=10` 안에 들어오도록 자동 축소한다. 필요하면 `--scale` 또는 `--max-sample-high`로 조정한다.

날짜별 학습 데이터가 필요하면 `daily` profile을 만든다.

```bash
ddareungi-build-profile \
  --rental-dir "data/서울특별시 공공자전거 대여이력 정보_2025" \
  --station-keyword "마곡" \
  --station-count 3 \
  --profile-kind daily \
  --max-sample-high 10 \
  --output outputs/data/magok_3station_daily_profile.json
```

`hourly` profile은 1년치를 평균적인 24시간 패턴으로 압축한다. 반면 `daily` profile은 `날짜 -> 시간 -> 대여소`별 실제 대여/반납 count를 보존한다.

현재 real-profile 메뉴는 `daily` profile을 읽어 episode마다 실제 날짜 하나를 선택하고, 해당 날짜의 24시간 대여/반납 count를 step에 적용한다. 그래서 baseline과 DQN은 평균 하루가 아니라 날짜별 수요 차이를 겪는다.

단, 공공데이터의 원본 count는 toy 환경의 자전거 보관 용량보다 매우 크다. 그래서 `daily` profile은 기본적으로 전체 날짜/시간/대여소 중 가장 큰 count를 `--max-sample-high` 값에 맞추는 max 정규화를 적용한다. 현재 기본값은 `10`이다. 예를 들어 원본 최대값이 `10`이고 `--max-sample-high 5`이면 같은 시간대 count `[10, 8, 6]`은 `[5, 4, 3]`으로 줄어든다. 이때 모든 값을 같은 배율로 줄이므로 시간대별 상대적인 수요 패턴은 유지된다. 정규화 여부, 원본 최대값, scale 값은 profile JSON의 `metadata.normalization`에 저장된다.

baseline과 DQN 평가는 `daily` profile의 앞 30일만 보지 않고 전체 날짜에 고르게 걸친 30개 날짜를 사용한다. 이렇게 해야 no-op, heuristic, DQN이 여러 계절과 요일 패턴에서 같은 기준으로 비교된다.

## 지금 버전의 의도

이 버전은 최종 완성본이 아니다. 다시 시작하기 위한 깨끗한 출발점이다.

앞으로는 다음 순서로만 확장한다.

1. 현재 코드 이해
2. reward와 state 설명 정리
3. DQN 학습 결과 확인
4. 실제 데이터 profile을 다시 천천히 연결
5. 필요할 때만 시각화 추가
