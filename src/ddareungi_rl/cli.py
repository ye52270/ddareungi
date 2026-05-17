"""따릉이 RL 최소 실험을 실행하는 콘솔 메뉴."""

from __future__ import annotations

from pathlib import Path

from ddareungi_rl.baselines import DemandAwarePolicy, LowStockPolicy, RandomPolicy
from ddareungi_rl.data_profile import load_profile
from ddareungi_rl.dqn import DQNConfig, evaluate_policy, save_model, train_dqn
from ddareungi_rl.env import DdareungiEnv, EnvConfig


DEFAULT_PROFILE_PATH = Path("outputs/data/magok_3station_profile.json")
DEFAULT_MODEL_PATH = Path("outputs/models/simple_dqn.pt")


def make_env(use_profile: bool = False) -> DdareungiEnv:
    """선택에 따라 toy 환경 또는 real-profile 환경을 만든다."""
    config = load_profile(DEFAULT_PROFILE_PATH) if use_profile and DEFAULT_PROFILE_PATH.exists() else EnvConfig()
    return DdareungiEnv(config=config, seed=42)


def run_baselines(use_profile: bool = False) -> None:
    """세 가지 baseline을 같은 환경에서 평가하고 출력한다."""
    env = make_env(use_profile)
    policies = {
        "random": RandomPolicy(seed=42),
        "low-stock": LowStockPolicy(),
        "demand-aware": DemandAwarePolicy(),
    }
    for name, policy in policies.items():
        result = evaluate_policy(env, policy)
        print(f"{name}: {result}")


def run_training(use_profile: bool = False) -> None:
    """DQN을 학습하고 baseline과 비교할 평가 결과를 출력한다."""
    env = make_env(use_profile)
    config = DQNConfig(episodes=100)
    policy, metrics = train_dqn(env, config=config)
    save_model(policy, DEFAULT_MODEL_PATH)
    result = evaluate_policy(env, policy)
    print(f"last_training_metric={metrics[-1]}")
    print(f"dqn_eval={result}")
    print(f"model_saved={DEFAULT_MODEL_PATH}")


def print_menu() -> None:
    """사용자가 선택할 수 있는 최소 메뉴를 출력한다."""
    print()
    print("Ddareungi RL Simple")
    print("1. Toy baseline 평가")
    print("2. Toy DQN 학습/평가")
    print("3. Real-profile baseline 평가")
    print("4. Real-profile DQN 학습/평가")
    print("0. 종료")


def main() -> None:
    """콘솔 메뉴를 반복 실행한다."""
    while True:
        print_menu()
        choice = input("선택: ").strip()
        if choice == "1":
            run_baselines(use_profile=False)
        elif choice == "2":
            run_training(use_profile=False)
        elif choice == "3":
            run_baselines(use_profile=True)
        elif choice == "4":
            run_training(use_profile=True)
        elif choice == "0":
            print("종료합니다.")
            return
        else:
            print("알 수 없는 선택입니다.")


if __name__ == "__main__":
    main()
