# AGENTS.md

## Project Context

This repository is a final project for a DQN-based Ddareungi bike repositioning simulator.

The project should first prove that a small reinforcement learning environment works before expanding toward real Seoul bike-sharing data.

Current design direction:

- V0: toy environment with 3 virtual stations and 1 repositioning truck.
- Agent action: choose the next station to visit.
- Loading and unloading: handled automatically by a rule-based environment heuristic.
- Main comparison: DQN vs Random and heuristic baselines.
- Later expansion: Double DQN, Dueling DQN, and selected real Ddareungi data.

## Collaboration Style

For non-trivial coding, design, documentation architecture, RL logic, training/evaluation, visualization, or multi-file project work, always use the Design Agent / Main Codex / Review Agent workflow, even if it is slower. The user does not need to explicitly ask for subagents.

Roles:

- Main Codex coordinates the task, talks with the user, integrates results, and makes final decisions.
- Design Agent proposes implementation design, roadmap, file structure, experiment plan, and documentation updates.
- Review Agent reviews the design for scope risk, RL validity, missing baselines, unclear metrics, implementation complexity, and presentation feasibility.

The Review Agent should not implement code unless explicitly asked. It should focus on risks, inconsistencies, and concrete corrections.

The Design Agent and Review Agent should avoid overlapping ownership. A good default split is:

- Design Agent: "What should we build, and in what order?"
- Review Agent: "What is risky, inconsistent, over-scoped, or unclear?"
- Main Codex: "What final plan or code should actually land in the repository?"

## Project Priorities

1. Keep V0 small and executable first.
2. Prefer a working `ToyDdareungiEnv` over premature V1 complexity.
3. Keep state, action, reward, and episode termination explicit.
4. Compare DQN against Random and at least one heuristic baseline.
5. Track evaluation metrics such as episode reward, unmet demand, movement cost, and baseline improvement.
6. Keep visualization separate from training logic by replaying saved episode logs.
7. Use pixel/tile-map visualization inspired by FrozenLake for presentation clarity.

## Iterative Agent Workflow

For non-trivial project work, use the iterative Design Agent / Main Codex / Review Agent workflow. This applies even when the user does not explicitly request agent-based collaboration.

Default loop:

1. Design Agent proposes the next small improvement, including scope, affected files, and expected output.
2. Main Codex implements the smallest useful change that moves the project forward.
3. Review Agent reviews the result for bugs, scope creep, RL validity, missing evaluation, unclear naming, and consistency with this file.
4. Main Codex applies review feedback unless it is clearly out of scope, incorrect, or conflicts with the user's request.
5. Repeat until the requested task is complete.

Use this loop for:

- New environment features.
- Training or evaluation changes.
- Visualization changes.
- Documentation architecture changes.
- Refactors that affect more than one module.

Skip this workflow only for:

- Simple operational requests such as `git push`, `git status`, running a known command or script, or showing command output.
- Simple repository inspection or read-only orientation.
- Tiny typo fixes.
- Single-line documentation edits.
- Mechanical formatting.
- Other narrow tasks where no implementation or design decision is being made.

The loop should stay practical. Do not create large plans when a small, working patch is better.

## Coding Rules

Keep implementation simple, modular, and easy to explain in a final presentation.

Rules:

- Keep environment logic, agent/training logic, baseline policies, evaluation, and visualization in separate modules.
- Prefer clear Python scripts and modules over notebook-only implementation.
- Keep random seeds configurable for repeatable experiments.
- Avoid adding real-data complexity before the V0 loop works end-to-end.
- Keep all hyperparameters visible in config variables, dataclasses, or command-line arguments.
- Every Python function and method should have a docstring that briefly describes what it does; keep docstrings concise and focused on behavior, inputs, return values, or side effects when those are not obvious.
- Save episode logs in a replayable format before building advanced visualization.
- Add small smoke tests or executable scripts for major features.
- Keep dependencies minimal and document any new dependency when it is introduced.
- Prefer readable, presentation-friendly code over clever abstractions.
- Avoid broad refactors unless they directly support the current milestone.

For reinforcement learning code:

- Make observation and action spaces explicit.
- Normalize observations when helpful, but keep raw values available in `info` or logs.
- Keep reward components visible in `info` so results can be explained.
- Track at least episode reward and unmet demand for every evaluation run.
- Compare learned policies against Random and heuristic baselines before claiming improvement.

For visualization code:

- Build visualization as a replay of saved episode logs.
- Do not couple rendering directly to training.
- Prefer a FrozenLake-like pixel/tile-map style for V0.
- Show station stock, truck location, truck load, time step, reward, and unmet demand.
- Keep the visualization understandable even without reading the source code.

## Definition of Done

A task is done only when the result is usable, checked, and explainable.

For environment work:

- The environment can reset and step through a full episode.
- State, action, reward, termination, and `info` are inspectable.
- At least one simple policy can run on it without crashing.

For baseline work:

- Random policy runs for multiple episodes.
- At least one heuristic policy runs for multiple episodes.
- Results include average reward and average unmet demand.

For DQN work:

- Training can run from a clean command.
- Evaluation can run separately from training.
- Results are compared against baselines using the same environment settings.
- Model saving and loading are supported if evaluation depends on a trained model.

For visualization work:

- Visualization can replay a saved episode log.
- The replay shows the key state changes clearly.
- The visualization is not required for training to run.

For documentation work:

- README remains concise and project-facing.
- Detailed design docs explain the MDP and implementation choices.
- Version names and roadmap terms stay consistent across documents.

## Implementation Guidance

Prefer a simple, testable structure before adding advanced algorithms.

Suggested eventual structure:

```text
src/
  envs/
    toy_ddareungi_env.py
  agents/
    dqn.py
  policies/
    random_policy.py
    heuristic_policy.py
  training/
    train_dqn.py
    evaluate.py
  visualization/
    pixel_replay.py
  utils/
    logging.py
```

Keep the environment independent from the learning algorithm. The environment should expose a clear Gymnasium-style API where practical:

```text
reset() -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

Episode logs should contain enough information for visualization and analysis:

```text
time_step
state
action
reward
next_state
demand
returns
unmet_demand
full_returns
movement_cost
truck_location
truck_bikes
station_bikes
```

## Scope Control

Do not move to real Ddareungi data until the V0 loop works end-to-end:

1. Environment runs.
2. Random baseline runs.
3. Heuristic baseline runs.
4. DQN trains.
5. DQN can be evaluated against baselines.
6. At least one episode can be replayed visually.

Treat the following as optional extensions unless the user asks for them:

- Multiple trucks.
- Weather features.
- Holiday features.
- Action masking.
- Distance-based travel time.
- Full real-data pipeline.

## Documentation Guidance

Keep `README.md` concise and project-facing.

## Documentation Language

프로젝트 문서와 설명 문장은 기본적으로 한글로 작성한다.

적용 대상:

- `README.md`
- `docs/*.md`
- 주석, docstring, 다이어그램, 표, 실험 노트의 설명 문장

다음 항목은 정확성과 검색 가능성을 위해 영어를 그대로 사용할 수 있다.

- 코드 식별자, class/function/module 이름, config key, 파일명
- shell command, CLI option, package 이름, API 이름, library/framework 용어
- 파일 경로, URL, metric 이름, model 이름, DQN / Double DQN 같은 알고리즘 이름
- 논문 제목, 데이터셋 이름, 공식 용어, 외부 reference

한글 설명을 기본으로 하되, 영어 기술 용어는 자연스러운 경우 그대로 둔다. 실행해야 하는 명령어나 코드 식별자를 억지로 번역하지 않는다.

Use `docs/01_detailed_design.md` for detailed MDP explanation, reward examples, step flow, and development roadmap.

If roadmap terminology changes, keep version names consistent between README and detailed docs.
