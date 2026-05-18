# AGENTS.md

## Project Direction

This repository is now intentionally simplified.

The goal is not to build a large RL platform yet. The goal is to keep a small Ddareungi reinforcement learning project that the student can read, explain, and modify.

Core learning flow:

```text
Environment -> Baselines -> DQN variants -> Evaluation
```

Keep only what directly helps explain:

- State
- Action
- Reward
- Environment transition
- Baseline comparison
- PyTorch DQN learning

## Current Minimal Structure

```text
src/ddareungi_rl/
  env.py           # MDP environment
  config_loader.py # YAML/JSON config loader
  baselines.py     # Random, Low-stock, Demand-aware
  dqn.py           # Backward-compatible DQN API re-export
  algorithms/      # DQN, Double DQN, Dueling DQN implementations
  data_profile.py  # Optional real-data profile loader
  cli.py           # Small menu
```

Keep subpackages limited to algorithm code only. Do not add new layers unless they make the learning flow easier to explain.

## Coding Rules

- Keep code readable over clever.
- Every Python function and method should have a concise docstring.
- Prefer one clear function over several small abstractions when teaching value is higher.
- Do not add PPO, notebooks, PPT generation, or large data pipelines unless the user explicitly asks.
- Keep real data optional. It should only replace demand/return patterns, not change the MDP story.
- Keep comments and documentation in Korean unless code identifiers or standard RL terms are clearer in English.

## RL Rules

- State must stay visible and explainable.
- Action must stay simple: choose the next station to visit.
- Reward must stay explicit: `-10 * unmet_demand - 3 * rejected_returns - movement_cost`.
- DQN performance claims require baseline comparison.
- Training reward is not enough; use evaluation episodes.

## Collaboration Rule

For non-trivial design or code changes, use subagents for design/review when available. Keep their role practical:

- Design review: scope and file structure.
- RL review: MDP validity and claims.
- Main Codex: final implementation.

Skip subagents for tiny fixes, command execution, and git-only requests.

## Definition of Done

A change is done when:

- The user-facing flow remains simple.
- `pytest` passes.
- README still explains the project in a way a student can present.
- No large generated outputs or raw data are committed.
