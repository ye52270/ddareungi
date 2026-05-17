# 실제 따릉이 데이터 EDA 계획

## 목적

이 문서는 공공 따릉이 대여이력 CSV와 대여소 마스터 CSV를 이용해, 현재 `ToyDdareungiEnv`의 toy 수요/반납 패턴을 실제 데이터 기반 패턴으로 확장하기 위한 EDA 계획을 정리한다.

이 단계의 목표는 서울 전체 대여소를 곧바로 강화학습 action space로 사용하는 것이 아니다. 우선 실제 데이터에서 선택한 소수 대여소의 시간대별 대여/반납 패턴을 추출하고, 이를 작은 Gymnasium 환경의 transition model에 반영한다.

즉, 실제 CSV는 state/action/reward 정의를 바꾸는 데이터가 아니라, toy 환경의 수요와 반납 발생 분포를 더 현실적인 값으로 보정하는 근거다. Agent는 실제 미래 수요를 미리 알고 움직이는 것이 아니라, 같은 관측 state에서 다음 방문 station을 선택하고 그 결과로 발생한 unmet demand와 이동비용을 reward로 받는다.

```text
원본 CSV
  -> station-hour 대여/반납 집계
  -> 3개 대여소 profile 생성
  -> ToyDdareungiEnv demand/return pattern에 주입
  -> baseline / DQN / PyTorch DQN 비교
```

## 데이터 위치와 Git 정책

원본 데이터는 로컬 `data/` 폴더에 둔다.

```text
data/
  서울시 공공자전거 따릉이 대여소 마스터 정보.csv
  서울특별시 공공자전거 대여이력 정보_2025/
    서울특별시 공공자전거 대여이력 정보_2512.csv
```

`data/`는 `.gitignore` 대상이다. 원본 CSV, 중간 Parquet, 처리된 대용량 산출물은 git에 올리지 않는다. 문서와 작은 fixture, 코드, 요약 JSON만 필요한 경우에 한해 관리한다.

개인 식별이나 프로젝트 목표에 불필요한 원본 컬럼은 산출물에 남기지 않는다. 현재 목표에는 station ID, 대여/반납 시간대, station 좌표, 집계 count만 필요하다.

## 확인된 원본 스키마

### 대여이력 CSV

12월 파일 기준 확인 결과:

```text
파일: 서울특별시 공공자전거 대여이력 정보_2512.csv
크기: 약 352MB
인코딩: cp949
행 수: 약 1,965,170
```

주요 컬럼:

| 컬럼 | 사용 목적 |
|---|---|
| `대여일시` | 대여 demand의 시간대 추출 |
| `대여 대여소명` | 사람이 읽는 대여소 이름 |
| `대여대여소ID` | 마스터 CSV와 join할 대여소 ID |
| `반납일시` | 반납 return의 시간대 추출 |
| `반납대여소명` | 사람이 읽는 반납 대여소 이름 |
| `반납대여소ID` | 마스터 CSV와 join할 반납 대여소 ID |
| `이용시간(분)` | 이상치 탐지 후보 |
| `이용거리(M)` | 이상치 탐지 후보 |

### 대여소 마스터 CSV

확인 결과:

```text
파일: 서울시 공공자전거 따릉이 대여소 마스터 정보.csv
행 수: 3,418
인코딩: cp949
```

컬럼:

| 컬럼 | 사용 목적 |
|---|---|
| `대여소_ID` | 대여이력의 `대여대여소ID`, `반납대여소ID`와 join |
| `주소1` | 지역/권역 설명 |
| `주소2` | 세부 위치 설명 |
| `위도` | 거리 계산, 시각화 배치 후보 |
| `경도` | 거리 계산, 시각화 배치 후보 |

마스터 CSV에는 대여소명과 거치대 수가 없다. 대여소명은 대여이력 CSV에서 가져오고, capacity는 초기 real-pattern 단계에서는 기존 toy 설정을 유지한다.

## 데이터 품질 메모

- 대여이력의 `대여대여소ID`, `반납대여소ID`는 마스터의 `대여소_ID`와 잘 매칭된다.
- 12월 샘플 기준 station ID join은 정상적으로 동작했다.
- 일부 반납 대여소명 또는 ID에 `\N` 값이 있다. profile 생성 시 제외한다.
- 마스터 3,418개 중 일부 station은 위도/경도가 `0.0`이다. 거리 계산이나 지도 기반 선정에서는 제외한다.
- CSV는 `cp949` 인코딩으로 읽는 것이 안전하다.
- 대여/반납 CSV는 실제로 발생한 거래 기록이다. 자전거가 없어 빌리지 못한 잠재 수요는 직접 기록되지 않을 수 있다.

## 1차 분석 결과

12월 전체 파일을 streaming 방식으로 확인한 결과, 출퇴근 시간대 수요 피크가 뚜렷하다.

예시 시간대 피크:

```text
08시, 18시, 17시, 07시, 19시 대여량이 상대적으로 큼
```

이 결과는 현재 MDP state에 `time_step`을 포함하는 설계를 뒷받침한다.

## 1차 선택 대여소

첫 real-pattern profile은 마곡 권역 3개 대여소로 시작한다. 기준은 활동량, 지리적 근접성, 대여/반납 imbalance의 역할 차이다.

| 역할 | ID | 대여소명 | 12월 대여 | 12월 반납 | 차이 |
|---|---|---|---:|---:|---:|
| 중심 허브 | `ST-2031` | 마곡나루역 2번 출구 | 9,482 | 9,531 | -49 |
| 반납-heavy | `ST-2033` | LG유플러스 마곡사옥 | 3,172 | 3,711 | -539 |
| 대여-heavy | `ST-2049` | 마곡수명산 1-2단지 | 1,704 | 1,244 | +460 |

이 조합은 전체 서울을 대표하기 위한 표본이 아니다. 작은 MDP 안에서 자전거 부족과 과잉이 동시에 발생하도록 만든, 학습 가능한 real-pattern toy system이다.

## Station 선택 기준

1차 profile은 다음 기준으로 station을 고른다.

1. 마스터 CSV와 station ID join이 가능해야 한다.
2. 위도/경도가 `0.0`이 아니어야 한다.
3. 12월 활동량이 충분해야 한다.
4. 세 station이 같은 권역에 있어야 한다.
5. 역할이 달라야 한다.
   - 중심 허브
   - 대여-heavy
   - 반납-heavy
6. 시간대별 대여/반납 패턴이 너무 sparse하지 않아야 한다.

피해야 할 선택:

- 서울 전역에 흩어진 station 조합
- 활동량이 너무 적은 station
- 좌표가 없는 station
- 결측 `\N`이 많은 station
- 한 station만 압도적으로 커서 action이 한쪽으로 collapse하기 쉬운 조합

## EDA 산출물

1차 EDA 파이프라인은 다음 산출물을 목표로 한다.

```text
outputs/data/master_summary.json
outputs/data/december_join_quality.csv
outputs/data/magok_station_candidates.csv
outputs/data/magok_3station_hourly_profile.csv
outputs/data/magok_3station_profile.json
outputs/data/magok_eda_summary.md
```

권장 profile 구조:

```json
{
  "stations": [
    {
      "id": "ST-2031",
      "name": "마곡나루역 2번 출구",
      "lat": 37.566925,
      "lon": 126.827438
    }
  ],
  "demand_by_hour": {
    "0": [1, 0, 0],
    "8": [4, 2, 3]
  },
  "return_by_hour": {
    "0": [0, 1, 0],
    "8": [2, 3, 1]
  }
}
```

최종 구현에서는 평균값 그대로 쓰기보다, 시간대별 평균/분위수에서 작은 정수 범위를 만들고 `ToyDdareungiConfig.demand_pattern`, `ToyDdareungiConfig.return_pattern`에 주입하는 방식을 우선 검토한다.

## MDP 연결 방식

실제 데이터는 현재 MDP의 모든 요소를 바꾸지 않는다. 1차 연결 범위는 transition model이다.

| MDP 요소 | 현재 toy 환경 | real-pattern 1차 확장 |
|---|---|---|
| State | station 재고, 트럭 위치, 트럭 적재량, 시간 | 유지 |
| Action | 다음 방문 대여소 선택 | 유지 |
| Reward | `-10 * unmet_demand - movement_cost` | 유지 |
| Transition | 수동 demand/return pattern | 실제 CSV 기반 시간대별 demand/return profile |
| Horizon | 24 step | 유지 |

이렇게 해야 toy MDP와 real-pattern MDP를 비교할 때, 변화한 부분이 수요/반납 동역학이라는 점이 명확해진다.

학습과 평가는 같은 profile 구조를 사용하되 seed와 episode를 분리한다. 가능하면 profile 생성 기간과 평가 기간도 분리해, 특정 날짜의 패턴을 외운 결과를 일반적인 성능처럼 보이지 않게 한다.

예시 split:

```text
profile 추정: 2025-12-01 ~ 2025-12-21
held-out 검증: 2025-12-22 ~ 2025-12-31
```

## 다음 구현 순서

1. `src/ddareungi_rl/data/inspect_rental_csv.py`
   - 파일 크기, 인코딩, 컬럼명, 샘플 row 확인
2. `src/ddareungi_rl/data/build_station_hour_profile.py`
   - streaming 또는 DuckDB로 station-hour 대여/반납 집계
3. `src/ddareungi_rl/data/profile_loader.py`
   - `magok_3station_profile.json`을 환경 config로 변환
4. baseline 평가
   - Random
   - Low-stock
   - Demand-aware
5. PyTorch DQN 평가
6. README와 결과 표 업데이트

현재 구현된 CLI:

```bash
ddareungi-inspect-rental-csv <대여이력 CSV> --max-rows 1000

ddareungi-build-station-profile <대여이력 CSV> <대여소 마스터 CSV> \
  --start-date 2025-12-01 \
  --end-date 2025-12-21
```

## 안전한 표현

좋은 표현:

> 실제 따릉이 CSV에서 추정한 시간대별 대여/반납 패턴을 toy MDP의 transition model에 반영한다.

피해야 할 표현:

> 실제 서울 전체 따릉이 운영을 최적화했다.

> 전체 대여소를 대상으로 강화학습을 완료했다.

> 실제 데이터에서 DQN의 우수성을 증명했다.

현재 단계는 실제 데이터를 이용해 작은 MDP의 수요 가정을 현실에 가깝게 보정하는 단계다.
