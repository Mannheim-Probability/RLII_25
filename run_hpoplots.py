# plot_best_hpo_curves.py
from __future__ import annotations
import os
import re
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import optuna
import matplotlib.pyplot as plt

try:
    # Journal storage is optional in some installs
    from optuna.storages import JournalStorage

    try:
        # Optuna >=4 recommends JournalFileBackend
        from optuna.storages.journal import JournalFileBackend as _JournalBackend
    except Exception:  # pragma: no cover
        # Older Optuna uses JournalFileStorage
        from optuna.storages import JournalFileStorage as _JournalBackend
except Exception:  # pragma: no cover
    JournalStorage = None
    _JournalBackend = None


def _detect_storage(path: str):
    """
    Return an Optuna storage object or URL suitable for optuna.load_study.
    Supports:
      - SQLite files ('.db', '.sqlite', anything else if prefixed via sqlite URL)
      - Journal log files ('.log') via JournalStorage + JournalFileBackend/JournalFileStorage
    """
    p = os.path.abspath(path)

    # If caller already gave an optuna URL, just pass it through
    if "://" in path:
        return path

    # Journal .log
    if p.endswith(".log"):
        if JournalStorage is None or _JournalBackend is None:
            raise RuntimeError(
                "This looks like an Optuna Journal log, but JournalStorage/backends are not available.\n"
                "Install/upgrade Optuna (e.g., `pip install -U optuna`)."
            )
        return JournalStorage(_JournalBackend(p))

    # Assume SQLite file; optuna wants a URL with four slashes for absolute paths
    return f"sqlite:///{p}" if p.startswith("/") else f"sqlite:///{os.path.abspath(p)}"


def _read_study_name_from_journal(path: str) -> Optional[str]:
    """
    Journal log files typically include the study name on the first JSON line.
    We parse it once to avoid guessing.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        # look for "study_name":"..."
        m = re.search(r'"study_name"\s*:\s*"([^"]+)"', first)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _get_all_study_summaries(storage) -> list:
    """Return a list of StudySummary objects from the given storage, compat across Optuna versions."""
    # Optuna exposes either optuna.get_all_study_summaries or optuna.study.get_all_study_summaries
    fn = getattr(optuna, "get_all_study_summaries", None)
    if fn is None:
        from optuna.study import get_all_study_summaries as fn  # type: ignore
    return fn(storage=storage)


def _resolve_study_name(
    *,
    algo: str,
    path: str,
    storage,
    provided: Optional[str],
) -> str:
    """
    Decide which study_name to use, trying in order:
      1) user-provided mapping entry
      2) parse from journal header (if .log)
      3) if storage contains exactly one study, use it
      4) pick a study whose name contains the algo or file stem
      5) otherwise raise with a helpful message listing available studies
    """
    if provided:
        return provided

    # Try parsing from journal header
    if str(path).endswith(".log"):
        parsed = _read_study_name_from_journal(path)
        if parsed:
            return parsed

    # Inspect storage for existing studies
    try:
        summaries = _get_all_study_summaries(storage)
        names = [s.study_name for s in summaries]
    except Exception:
        names = []

    if len(names) == 1:
        return names[0]

    if names:
        stem = os.path.splitext(os.path.basename(path))[0]
        # Prefer a name that includes the algorithm label or the file stem
        for key in (algo, stem):
            matches = [n for n in names if key and key in n]
            if matches:
                return matches[0]
        # If multiple names exist and no match, raise an informative error
        raise RuntimeError(
            f"Could not determine study_name for '{algo}'. Available studies in storage: {names}\n"
            f"Provide it via study_names['{algo}'] or rename the file to include the study name."
        )

    # Fallback: use file stem
    return os.path.splitext(os.path.basename(path))[0]


def _infer_total_timesteps(path_or_study: str, default: int = 1_000_000) -> int:
    """
    Try to infer total timesteps from a suffix like '*_1e6' or '*_1000000'.
    Falls back to `default` if not found.
    """
    base = os.path.basename(path_or_study)
    # match tokens like 1e6, 2e5, or plain integers
    m = re.search(r"(_|\b)(\d+(?:e[0-9]+)?)\b", base)
    if m:
        token = m.group(2)
        try:
            if "e" in token:
                # e.g., 1e6
                val = float(token)
                # if val is something like 1e6 -> 1000000.0
                return int(val)
            return int(token)
        except Exception:
            pass
    return default


def _best_complete_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        # if nothing complete, still pick the best by value among available
        candidates = [t for t in study.trials if t.value is not None]
        if not candidates:
            raise RuntimeError("No trials with values found.")
        complete = candidates
    # maximize reward
    return max(complete, key=lambda t: t.value if t.value is not None else -np.inf)


def _trial_curve_as_timesteps(
    trial: optuna.trial.FrozenTrial,
    total_timesteps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a trial's intermediate values {step: reward} into (timesteps, rewards).
    We map step indices (1..K) to evenly spaced timesteps up to total_timesteps.
    """
    if not trial.intermediate_values:
        # fall back to a degenerate "curve" with only the final value
        return np.array([total_timesteps], dtype=np.int64), np.array([trial.value], dtype=float)

    steps, rewards = zip(*sorted(trial.intermediate_values.items()))
    steps = np.array(steps, dtype=int)
    rewards = np.array(rewards, dtype=float)

    K = steps.max()
    if K <= 0:
        xs = np.full_like(rewards, total_timesteps, dtype=np.int64)
    else:
        # Evenly spaced: step i -> i/K * total_timesteps
        xs = (steps / K * total_timesteps).astype(np.int64)
    return xs, rewards


def plot_best_learning_curves(
    algo_to_storage_path: Dict[str, str],
    *,
    study_names: Optional[Dict[str, str]] = None,
    total_timesteps_by_algo: Optional[Dict[str, int]] = None,
    title: str = "Best HPO Learning Curves",
    figsize: Tuple[int, int] = (9, 6),
    alpha: float = 0.9,
    marker_every: Optional[int] = None,
) -> Dict[str, dict]:
    """
    For each algorithm:
      - load the study
      - pick the best COMPLETE trial (by max reward)
      - extract intermediate evaluation rewards
      - map evaluation steps -> timesteps
      - plot one curve per algorithm

    Returns a dict with basic metadata per algo (best params, value, n_points).
    """
    meta = {}
    plt.figure(figsize=figsize)

    for algo, path in algo_to_storage_path.items():
        storage = _detect_storage(path)

        # Determine study_name robustly (handles journal + sqlite + multi-study storages)
        sname = _resolve_study_name(
            algo=algo,
            path=path,
            storage=storage,
            provided=(study_names.get(algo) if study_names else None),
        )

        # Load study (retry fallback: if KeyError, list available and show hint)
        try:
            study = optuna.load_study(study_name=sname, storage=storage)
        except KeyError:
            summaries = []
            try:
                summaries = _get_all_study_summaries(storage)
            except Exception:
                pass
            available = [s.study_name for s in summaries] if summaries else []
            raise RuntimeError(
                f"Study '{sname}' not found in storage for algo '{algo}'."
                + (f" Available studies: {available}" if available else "")
                + f"\nHint: pass an explicit name via study_names['{algo}']"
            )

        # Best trial
        best = _best_complete_trial(study)

        # Timesteps upper bound
        total_ts = (
            total_timesteps_by_algo[algo]
            if total_timesteps_by_algo and algo in total_timesteps_by_algo
            else _infer_total_timesteps(sname) or _infer_total_timesteps(path)
        )

        xs, ys = _trial_curve_as_timesteps(best, total_ts)

        # Plot
        kwargs = {}
        if marker_every is not None and marker_every > 0:
            # put a marker every N points to help visibility
            kwargs["markevery"] = max(1, marker_every)

        plt.plot(xs, ys, label=algo, alpha=alpha, **kwargs)

        # collect meta
        meta[algo] = {
            "study_name": sname,
            "storage": str(storage),
            "best_value": float(best.value) if best.value is not None else None,
            "n_intermediate_points": len(best.intermediate_values),
            "best_params": dict(best.params),
        }

    plt.xlabel("timesteps")
    plt.ylabel("mean reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return meta


if __name__ == "__main__":
    # === EDIT THESE PATHS ===
    # Point each algo name to its Optuna storage file (journal .log or sqlite .db)
    # ---- point to your 4 journal logs ----
    algo_paths = {
        "ppo": "/Users/tilmanaach/projects/RLII_25/logs/ppo_lunarlander_nsgaii_1e6.log",
        "ppo_changed_before_normalization": "/Users/tilmanaach/projects/RLII_25/logs/ppo_changed_before_normalization_lunarlander_nsgaii_1e6.log",
        "ppo_mod_advantages": "/Users/tilmanaach/projects/RLII_25/logs/ppo_mod_ad_lunarlander_nsgaii_1e6.log",
        "ppo_mod_sampling": "/Users/tilmanaach/projects/RLII_25/logs/ppo_mod_sampling_lunarlander_nsgaii_1e6.log",
    }

    # ---- study names inside those logs (journal header) ----
    study_names = {
        "ppo": "ppo_lunarlander_nsgaii_1e6",
        "ppo_changed_before_normalization": "ppo_changed_before_normalization_lunarlander_nsgaii_1e6",
        "ppo_mod_advantages": "ppo_mod_ad_lunarlander_nsgaii_1e6",
        "ppo_mod_sampling": "ppo_mod_sampling_lunarlander_nsgaii_1e6",
    }

    # optional: if any algo ran a different budget, set it here; otherwise it infers 1e6 from the name
    total_ts = {
        # "ppo": 1_000_000,
        # "ppo_mod_sampling": 1_000_000,
    }

    info = plot_best_learning_curves(
        algo_paths,
        study_names=study_names,
        total_timesteps_by_algo=total_ts,
        title="Best HPO Learning Curves (one per algorithm)",
        figsize=(10, 6),
        marker_every=1,
    )
    print(info)
    plt.show()
