import optuna
import optuna.importance

storage = "sqlite:///results/fast_run/stage2.db"
studies = optuna.get_all_study_names(storage)

print(f"{'Study':<50} {'Trials':>6}  {'Best α':>7}  {'Best adv':>9}  {'Mean adv':>9}  {'Std':>6}  {'Worst adv':>10}  {'Status'}")
print("-" * 120)

for name in sorted(studies):
    study = optuna.load_study(study_name=name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    running   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING)

    if not completed:
        print(f"{name:<50} {'0':>6}  {'—':>7}  {'—':>9}  {'—':>9}  {'—':>6}  {'—':>10}  "
              f"pruned={pruned} running={running}")
        continue

    values   = [t.value for t in completed]
    best     = max(completed, key=lambda t: t.value)
    worst    = min(completed, key=lambda t: t.value)
    mean_adv = sum(values) / len(values)
    std_adv  = (sum((v - mean_adv)**2 for v in values) / len(values)) ** 0.5

    # convergence: is the best still improving or has it plateaued?
    running_best = []
    for i, t in enumerate(sorted(completed, key=lambda t: t.number)):
        running_best.append(max(t.value, running_best[-1] if running_best else t.value))
    last_improvement = max(
        (i for i, t in enumerate(sorted(completed, key=lambda t: t.number))
         if t.value == best.value),
        default=0
    )
    trials_since_best = len(completed) - 1 - last_improvement
    converged = "✓ converged" if trials_since_best >= 5 else f"↑ improving ({trials_since_best} since best)"

    print(f"{name:<50} {len(completed):>6}  {best.params['alpha']:>7.3f}  "
          f"{best.value:>+9.4f}  {mean_adv:>+9.4f}  {std_adv:>6.4f}  "
          f"{worst.value:>+10.4f}  {converged}  pruned={pruned}")

    # show all alpha trials sorted by alpha value so you can see the shape
    print(f"  {'alpha':>7}  {'adv':>9}")
    for t in sorted(completed, key=lambda t: t.params["alpha"]):
        marker = " ← best" if t.number == best.number else ""
        print(f"  {t.params['alpha']:>7.3f}  {t.value:>+9.4f}{marker}")
    print()