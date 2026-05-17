"""ToyDdareungiEnv 한 step을 수업용으로 풀어 보여주는 walkthrough."""

from __future__ import annotations

from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.training.evaluate import jsonable_observation


def format_observation(observation: object) -> str:
    """observation vector를 소수 둘째 자리 문자열로 변환한다."""
    values = jsonable_observation(observation)
    return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"


def main() -> None:
    """환경 reset부터 action 하나의 transition까지 단계별 설명을 출력한다."""
    env = ToyDdareungiEnv(seed=42)
    observation, info = env.reset(seed=42)
    action = 0

    print("ToyDdareungiEnv Gymnasium walkthrough")
    print("=" * 44)
    print("1) reset()")
    print(f"observation_space={env.observation_space}")
    print(f"action_space={env.action_space}")
    print(f"initial_observation={format_observation(observation)}")
    print(f"initial_station_bikes={info['station_bikes']}")
    print()

    print("2) action 선택")
    print(f"action={action} -> {info['station_names'][action]} 방문")
    print()

    next_observation, reward, terminated, truncated, step_info = env.step(action)

    print("3) step(action) 결과")
    print(f"decision_station_bikes={step_info['decision_station_bikes']}")
    print(
        "rebalance="
        f"{step_info['rebalance_type']} {step_info['rebalance_amount']}대, "
        f"truck_bikes={step_info['truck_bikes_after_rebalance']}"
    )
    print(f"demand={step_info['demand']} served={step_info['served_demand']} unmet={step_info['unmet_demand']}")
    print(f"returns={step_info['returns']} accepted={step_info['accepted_returns']}")
    print(f"reward={reward} = -10 * unmet_demand - movement_cost")
    print(f"next_observation={format_observation(next_observation)}")
    print(f"terminated={terminated}, truncated={truncated}")


if __name__ == "__main__":
    main()
