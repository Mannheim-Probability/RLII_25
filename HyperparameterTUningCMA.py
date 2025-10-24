# pip install "gymnasium[box2d]" stable-baselines3 optuna matplotlib pandas numpy

import gymnasium as gym
import optuna
from optuna.samplers import CmaEsSampler
from rl_zoo3.custom_algos import PPO_changed_before_Normalization
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os

# ============================================================
# Globale Konfiguration
# ============================================================
ENV_ID = "LunarLander-v3"   # Gymnasium-API
SEED = 42

# Trainings-/Eval-/Val-Logik
TRAIN_SEED = 0                              # Trainingsseed je Trial
EVAL_SEEDS = list(range(10, 20))            # Seeds für Evaluation (für alle Trials gleich)
N_EPISODES_PER_SEED = 20
VAL_SEED = 777                              # Start-Seed für Validierungsblöcke
N_VAL_EPISODES = 20

TOTAL_STEPS_TRAIN = int(1e5)

# fixes Evaluations-Gamma (kandidatenunabhängig)
GAMMA_EVAL = 0.99

# Statistik/Tests
ALPHA = 0.05                      # Signifikanzniveau für Dominanztests
PERM_B = 4000                     # Replikate für Bootstrap/Permutation (wir recyceln den Namen)

# === Elite-Validierung über μ-Beste ===
# Schätzwert für λ (CMA-ES-Population) und daraus μ := floor(λ/2)
CMA_LAMBDA_EST = 9
CMA_MU = max(2, CMA_LAMBDA_EST // 2)

# Endlosschleifen-Schutz (Resampling über Seed-Blöcke)
MAX_VAL_BLOCKS = 8

# Präzisionskriterium für CI(Δ) in der Validierung
CI_EPS = 5.0   # gewünschte Halbbreite (anpassen, falls feinere Präzision gewünscht)

# TPE-ähnliches Resampling
HISTORY = []        # Liste {"trial": int, "params": {...}, "eval_mean": float, "model_path": str}
STUDY_REF = None
_LAST_ENQUEUED = None

# ===== Akzeptierter Incumbent (wird für Enqueue genutzt) =====
BEST = None  # {"params":..., "canon":..., "eval_mean":..., "model_path": ...}

# ============================================================
# Hilfsfunktionen (Sampling, Evaluierung, Statistik)
# ============================================================
def _make_single_env(seed):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    return env

def _canonize_params(p):
    """Diskrete aus kontinuierlichen Proxys ableiten und auf Konsistenz runden."""
    p = dict(p)
    batch_size = int(2 ** round(p.pop("log2_batch_size")))
    n_epochs = int(round(p.pop("n_epochs_cont")))
    n_steps = int(p.pop("n_steps_cont"))
    n_steps = max(batch_size, (n_steps // batch_size) * batch_size)
    p.update(batch_size=batch_size, n_epochs=n_epochs, n_steps=n_steps)
    return p

def _eval_discounted_returns(model, seed, n_episodes, gamma):
    """Diskontierte kumulierte Returns pro Episode (deterministische Policy)."""
    rets = []
    for k in range(n_episodes):
        env = gym.make(ENV_ID)
        obs, info = env.reset(seed=seed + k)
        G, t = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            G += (gamma ** t) * float(r)
            t += 1
            if terminated or truncated:
                rets.append(G)
                break
        env.close()
    return np.asarray(rets, dtype=float)

def _eval_multi_seeds(model, seeds, n_episodes_per_seed, gamma):
    """Konkatenation der diskontierten Returns über mehrere Seeds."""
    parts = []
    for s in seeds:
        parts.append(_eval_discounted_returns(model, seed=s, n_episodes=n_episodes_per_seed, gamma=gamma))
    return np.concatenate(parts, axis=0)

def _eval_multi_seeds_both(model, seeds, n_episodes_per_seed, gamma):
    """Gibt (alle Episoden-Returns konkateniniert, seedweise Mittelwerte) zurück."""
    all_returns = []
    seed_means = []
    for s in seeds:
        rets = _eval_discounted_returns(model, seed=s, n_episodes=n_episodes_per_seed, gamma=gamma)
        all_returns.append(rets)
        seed_means.append(float(np.mean(rets)))
    return np.concatenate(all_returns, axis=0), np.asarray(seed_means, dtype=float)

# ========= Hilfsfunktionen für Dominanz & CI-Validierung =========
def _ecdf_vals(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, len(x)+1) / len(x)

def _Dplus(a, b):
    """Einseitige KS-Statistik D^+ für FSD-Verletzung: sup_t(F_A - F_B). <=0 impliziert FSD(A>=B)."""
    xa, _ = _ecdf_vals(a); xb, _ = _ecdf_vals(b)
    x = np.unique(np.concatenate([xa, xb]))
    Fa_x = np.searchsorted(xa, x, side="right") / len(xa)
    Fb_x = np.searchsorted(xb, x, side="right") / len(xb)
    return float(np.max(Fa_x - Fb_x))  # >0 verletzt FSD (A dominiert B NICHT)

def _Splus(a, b):
    """SSD-Verletzungsmaß: sup_t ∫(F_A - F_B) du bis t. <=0 impliziert SSD(A>=B)."""
    xa, _ = _ecdf_vals(a); xb, _ = _ecdf_vals(b)
    x = np.unique(np.concatenate([xa, xb]))
    Fa_x = np.searchsorted(xa, x, side="right") / len(xa)
    Fb_x = np.searchsorted(xb, x, side="right") / len(xb)
    diff = Fa_x - Fb_x
    # rechteckige Riemann-Summen (Schrittweiten = Δx, erster Schritt ignoriert)
    dx = np.diff(np.concatenate([x[:1], x]))
    integ = np.cumsum(diff * dx)
    return float(np.max(integ))  # >0 verletzt SSD

def _pvalue_dom_by_seed_boot(all_returns_by_seed_A, all_returns_by_seed_B, stat_fn, B=PERM_B, rng=None):
    """
    Seed-weise Bootstrap (mit Zurücklegen) für Dominanztest:
    stat_fn: _Dplus (FSD) oder _Splus (SSD).
    """
    if rng is None: rng = np.random.default_rng(0)
    A_concat = np.concatenate(all_returns_by_seed_A)
    B_concat = np.concatenate(all_returns_by_seed_B)
    S_obs = stat_fn(A_concat, B_concat)
    m = len(all_returns_by_seed_A)
    assert m == len(all_returns_by_seed_B) and m > 0
    cnt = 0
    for _ in range(B):
        idx = rng.integers(0, m, size=m)
        A = np.concatenate([all_returns_by_seed_A[i] for i in idx])
        B_ = np.concatenate([all_returns_by_seed_B[i] for i in idx])
        S = stat_fn(A, B_)
        if S > 0.0:  # Verletzung (kein Dominanzbeweis)
            cnt += 1
    p = (cnt + 1) / (B + 1)  # einseitig
    return float(p), float(S_obs)

def _ci_halfwidth_paired_boot(seed_means_i, seed_means_j, B=PERM_B, rng=None):
    """
    CI-Halbbreite für Δ = μ_i - μ_j via gepaarten Bootstrap über Seeds.
    seed_means_*: Liste von Arrays (pro Block), wir konkatenieren sie.
    """
    if rng is None: rng = np.random.default_rng(0)
    a = np.concatenate(seed_means_i, axis=0)
    b = np.concatenate(seed_means_j, axis=0)
    assert len(a) == len(b) and len(a) > 0
    d = a - b
    n = len(d)
    d_obs = float(np.mean(d))
    means = np.empty(B, dtype=float)
    for t in range(B):
        idx = rng.integers(0, n, size=n)
        means[t] = float(np.mean(d[idx]))
    q_lo, q_hi = np.percentile(means, [2.5, 97.5])
    halfwidth = max(abs(d_obs - q_lo), abs(q_hi - d_obs))
    return float(halfwidth), d_obs

# ============================================================
# NEU: Elite-Auswahl per stochastischer Dominanz (statt eval_mean)
# ============================================================
def _select_elite_by_stochastic_dominance(mu, records, alpha=ALPHA):
    """
    Wählt μ-Eliten aus allen bisherigen Trials anhand stochastischer Dominanz.
    Vorgehen:
      1) Für jeden Trial: per-seed Rückgabeverteilungen (EVAL_SEEDS × N_EPISODES_PER_SEED) berechnen.
      2) Dominanzkanten via FSD (D^+, Bootstrap). Falls nötig, SSD-Fallback.
      3) Schichte in Dominanz-Layern (undominierte Fronten). Nimm Layer für Layer, bis μ erreicht.
         Innerhalb eines Layers: Tie-Breaker = eval_mean (absteigend), dann Trial-Nummer.
    Rückgabe: Liste der ausgewählten Record-Dicts (Länge μ), in Auswahlreihenfolge.
    """
    if len(records) == 0:
        return []

    # Modelle laden & per-seed Returns evaluieren (auf GAMMA_EVAL, deterministische Policy)
    models = {rec["trial"]: PPO_changed_before_Normalization.load(rec["model_path"]) for rec in records}
    returns_by_seed = {}
    for rec in records:
        t_id = rec["trial"]
        per_seed_returns = []
        for s in EVAL_SEEDS:
            rets = _eval_discounted_returns(models[t_id], seed=s, n_episodes=N_EPISODES_PER_SEED, gamma=GAMMA_EVAL)
            per_seed_returns.append(rets)
        returns_by_seed[t_id] = per_seed_returns

    ids = [rec["trial"] for rec in records]
    id_to_rec = {rec["trial"]: rec for rec in records}
    rng = np.random.default_rng(0)

    def _dominance_edges(stat_fn):
        edges = {i: set() for i in ids}   # i -> {j} wenn i dominiert j
        for a_id, b_id in itertools.permutations(ids, 2):
            p, S = _pvalue_dom_by_seed_boot(returns_by_seed[a_id], returns_by_seed[b_id],
                                            stat_fn=stat_fn, B=PERM_B, rng=rng)
            if (p < alpha) and (S <= 0.0):
                edges[a_id].add(b_id)
        return edges

    # Versuche FSD, sonst SSD
    edges = _dominance_edges(_Dplus)
    # prüfe, ob es überhaupt undominierte gibt, ansonsten SSD-Fallback
    incoming = {i: 0 for i in ids}
    for a in ids:
        for b in edges[a]:
            incoming[b] += 1
    if all(incoming[i] > 0 for i in ids):
        edges = _dominance_edges(_Splus)

    # Layered (Fronten) Auswahl
    remaining = set(ids)
    selected = []
    while remaining and len(selected) < mu:
        incoming = {i: 0 for i in remaining}
        for a in remaining:
            for b in edges.get(a, set()):
                if b in remaining:
                    incoming[b] += 1
        layer = [i for i in remaining if incoming[i] == 0]
        if not layer:
            # Graph zyklisch/inkomparabel -> alles als letzter Layer betrachten
            layer = list(remaining)
        # Tie-Break innerhalb Layer: eval_mean absteigend, dann Trial-ID
        layer.sort(key=lambda i: (id_to_rec[i]["eval_mean"], -i), reverse=True)
        take = layer[: max(0, mu - len(selected))]
        selected.extend(take)
        remaining -= set(take)
        # zur Sicherheit, entferne zudem die ganze Layer, um nächste Front zu berechnen
        remaining -= set(layer[:0])  # no-op, Platzhalter

    elite = [id_to_rec[i] for i in selected]
    return elite

# ============================================================
# Elite-Validierung mit Dominanz-Auswahl + CI(Δ)-Stoppkriterium
# ============================================================
def _validate_elite_equal_means(elite_records, start_seed=VAL_SEED, block_len=None,
                                n_episodes_per_seed=N_VAL_EPISODES, alpha=ALPHA,
                                max_blocks=MAX_VAL_BLOCKS):
    """
    Angepasst:
      - Auswahl: stochastische Dominanz (FSD, bei Bedarf SSD).
      - Validierung: CI-Halbbreite für Δ = μ_leader - μ_runner ≤ CI_EPS (gepaarter Seed-Bootstrap).
    Rückgabe (Schnittstelle unverändert):
        winner_idx      : Index im elite_records der Gewinner-Policy
        pass_all        : bool, ob CI-Kriterium erfüllt (True) oder Budgetstopp (False)
        min_p           : hier ungenutzt -> None
        blocks_used     : Anzahl seed-Blöcke
        val_means_by_id : Dict trial -> aggregierter Val-Mean
    """
    assert len(elite_records) >= 2
    if block_len is None:
        block_len = len(EVAL_SEEDS)

    # Aggregationen
    agg_seed_means = {rec["trial"]: [] for rec in elite_records}        # Liste [seed_means_block]
    val_means_by_id = {rec["trial"]: 0.0 for rec in elite_records}
    returns_by_seed = {rec["trial"]: [] for rec in elite_records}       # Liste [returns_seed] über alle Blöcke

    # Modelle laden
    models = {rec["trial"]: PPO_changed_before_Normalization.load(rec["model_path"]) for rec in elite_records}

    rng = np.random.default_rng(0)
    blocks_used = 0
    pass_all = False
    min_p = None  # Legacy-Ausgabe

    ids = [rec["trial"] for rec in elite_records]
    id_to_idx = {t_id: i for i, t_id in enumerate(ids)}

    def _undominated_from_edges(edges_dict, node_ids):
        incoming = {i: 0 for i in node_ids}
        for a in node_ids:
            for b in edges_dict.get(a, set()):
                if b in incoming:
                    incoming[b] += 1
        return [i for i in node_ids if incoming[i] == 0]

    while True:
        # --- 1) neuen Seed-Block
        start = start_seed + blocks_used * block_len
        seeds_blk = list(range(start, start + block_len))

        # --- 2) Evaluieren: seedweise Returns und Mittel
        for rec in elite_records:
            t_id = rec["trial"]
            seed_means_blk = []
            for s in seeds_blk:
                rets = _eval_discounted_returns(models[t_id], seed=s, n_episodes=n_episodes_per_seed, gamma=GAMMA_EVAL)
                returns_by_seed[t_id].append(rets)
                seed_means_blk.append(float(np.mean(rets)))
            agg_seed_means[t_id].append(np.asarray(seed_means_blk, dtype=float))

        # --- 3) Dominanzgraph (FSD zuerst, sonst SSD)
        def _dominance_edges(stat_fn):
            edges = {i: set() for i in ids}   # i -> {j} wenn i dominiert j
            for a_id, b_id in itertools.permutations(ids, 2):
                p, S = _pvalue_dom_by_seed_boot(returns_by_seed[a_id], returns_by_seed[b_id],
                                                stat_fn=stat_fn, B=PERM_B, rng=rng)
                if (p < alpha) and (S <= 0.0):
                    edges[a_id].add(b_id)
            return edges

        edges = _dominance_edges(_Dplus)  # FSD
        undominated = _undominated_from_edges(edges, ids)

        if not undominated:
            # Fallback auf SSD
            edges = _dominance_edges(_Splus)
            undominated = _undominated_from_edges(edges, ids)

        if not undominated:
            # Vollständig zyklischer/inkomparabler Graph -> alle als „undominiert“ betrachten
            undominated = ids

        # --- 4) Stopp-/Fortsetzungskriterien: CI(Δ)-Halbbreite
        # Aggregierte Mittel aktualisieren
        for rec in elite_records:
            t_id = rec["trial"]
            a = np.concatenate(agg_seed_means[t_id], axis=0)
            val_means_by_id[t_id] = float(np.mean(a))

        # Leader bestimmen
        if len(undominated) == 1:
            leader_id = undominated[0]
        else:
            leader_id = max(undominated, key=lambda t: val_means_by_id[t])

        # Runner (bester Nicht-Leader)
        others = [t for t in ids if t != leader_id]
        runner_id = max(others, key=lambda t: val_means_by_id[t])

        # CI-Halbbreite über alle bisher gesammelten Seeds
        halfwidth, d_obs = _ci_halfwidth_paired_boot(
            agg_seed_means[leader_id], agg_seed_means[runner_id], B=PERM_B, rng=rng
        )

        blocks_used += 1
        if halfwidth <= CI_EPS:
            pass_all = True
            break
        if blocks_used >= max_blocks:
            # Budget erschöpft -> Abbruch ohne erfülltes Präzisionskriterium
            pass_all = False
            break

    # finaler Gewinner: Leader (wie oben bestimmt)
    winner_id = leader_id
    winner_idx = id_to_idx[winner_id]

    return winner_idx, pass_all, min_p, blocks_used, val_means_by_id

# ============================================================
# Optuna-Objective (unverändert außer Reporting-Entfall)
# ============================================================
def objective(trial: optuna.Trial) -> float:
    global BEST
    # --- kontinuierliche (CMA-ES) Parameter ---
    lr          = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    gamma       = trial.suggest_float("gamma",         0.95, 0.9999)
    gae_lambda  = trial.suggest_float("gae_lambda",    0.80, 0.99)
    clip_range  = trial.suggest_float("clip_range",    0.10, 0.30)
    log2_batch_size = trial.suggest_float("log2_batch_size", 4.0, 8.0)
    n_epochs_cont   = trial.suggest_float("n_epochs_cont", 3.0, 20.0)
    n_steps_cont    = trial.suggest_float("n_steps_cont", 512, 4096, log=True)

    # Diskretisierung
    params = {
        "learning_rate":   lr,
        "gamma":           gamma,
        "gae_lambda":      gae_lambda,
        "clip_range":      clip_range,
        "log2_batch_size": log2_batch_size,
        "n_epochs_cont":   n_epochs_cont,
        "n_steps_cont":    float(n_steps_cont),
    }
    trial.set_user_attr("params_logged", params)

    # --- Training (Seed fest) ---
    train_env = DummyVecEnv([lambda: _make_single_env(TRAIN_SEED)])
    canon = _canonize_params(params)
    model = PPO_changed_before_Normalization(
        "MlpPolicy",
        train_env,
        learning_rate=canon["learning_rate"],
        gamma=canon["gamma"],              # Training: Kandidaten-γ
        gae_lambda=canon["gae_lambda"],
        clip_range=canon["clip_range"],
        n_steps=canon["n_steps"],
        batch_size=canon["batch_size"],
        n_epochs=canon["n_epochs"],
        seed=TRAIN_SEED,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_STEPS_TRAIN, progress_bar=False)
    train_env.close()

    # --- Eval: Seeds×Episoden → seedweise Mittel (für Übersicht) ---
    eval_returns, eval_seed_means = _eval_multi_seeds_both(
        model, seeds=EVAL_SEEDS, n_episodes_per_seed=N_EPISODES_PER_SEED, gamma=GAMMA_EVAL
    )
    eval_mean = float(np.mean(eval_returns))

    # Log: Evaluation
    trial.set_user_attr("eval_mean", eval_mean)
    trial.set_user_attr("n_eval", int(len(eval_returns)))

    # --- Modell speichern (für Elite-Validierung)
    model_path = f"PPO_changed_before_Normalization_trial{trial.number}.zip"
    model.save(model_path)

    # Historie updaten (für Elite-Auswahl)
    HISTORY.append({"trial": trial.number, "params": params, "eval_mean": eval_mean, "model_path": model_path})

    # --- Elite bestimmen: μ-Beste **per stochastischer Dominanz**
    mu = min(CMA_MU, len(HISTORY))
    trial.set_user_attr("elite_mu", int(mu))
    if mu >= 2:
        elite = _select_elite_by_stochastic_dominance(mu, HISTORY, alpha=ALPHA)
    else:
        elite = HISTORY[:1]

    # --- Elite-Validierung (Dominanz + CI-Stopp)
    if len(elite) >= 2:
        winner_idx, pass_all, min_p, blocks_used, val_means_by_id = _validate_elite_equal_means(
            elite, start_seed=VAL_SEED, block_len=len(EVAL_SEEDS),
            n_episodes_per_seed=N_VAL_EPISODES, alpha=ALPHA, max_blocks=MAX_VAL_BLOCKS
        )
        trial.set_user_attr("elite_pass_all", bool(pass_all))
        trial.set_user_attr("elite_min_p", None)  # Legacy-Feld ungenutzt
        trial.set_user_attr("elite_blocks_used", int(blocks_used))
        winner_rec = elite[winner_idx]
        trial.set_user_attr("elite_winner_trial", int(winner_rec["trial"]))

        # Incumbent (BEST) setzen/aktualisieren auf Gewinner
        if BEST is None or BEST.get("params") != winner_rec["params"]:
            canon_w = _canonize_params(winner_rec["params"])
            BEST = {
                "params": winner_rec["params"],
                "canon": canon_w,
                "eval_mean": winner_rec["eval_mean"],
                "model_path": winner_rec["model_path"],
            }
    else:
        if BEST is None:
            BEST = {
                "params": params,
                "canon": canon,
                "eval_mean": eval_mean,
                "model_path": model_path,
            }
            trial.set_user_attr("elite_pass_all", True)
            trial.set_user_attr("elite_min_p", None)
            trial.set_user_attr("elite_blocks_used", 0)
            trial.set_user_attr("elite_winner_trial", int(trial.number))

    # Zielfunktion: Eval-Mean (unverändert)
    return eval_mean

# ============================================================
# CMA-ES mit Startpunkt x0
# ============================================================
sampler = CmaEsSampler(
    seed=SEED,
    sigma0=0.003,
    x0={
        "learning_rate":   3e-4,
        "gamma":           0.999,
        "gae_lambda":      0.98,
        "clip_range":      0.20,
        "log2_batch_size": 5.0,
        "n_epochs_cont":   4.0,
        "n_steps_cont":    1024.0,
    },
)

# ===== Callback: bestes validiertes Set enqueuen (wenn neu) =====A
def _enqueue_best_for_next(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    global BEST, _LAST_ENQUEUED
    if BEST is not None:
        best_params = BEST["params"]
        if trial.user_attrs.get("params_logged") != best_params and best_params != _LAST_ENQUEUED:
            try:
                study.enqueue_trial(best_params)
                _LAST_ENQUEUED = best_params
            except Exception:
                pass

# ===== Reporting-Callback (unverändert) =====
def _report_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    value = trial.value if trial.value is not None else float("nan")
    eval_mean = trial.user_attrs.get("eval_mean")

    elite_mu = trial.user_attrs.get("elite_mu")
    elite_pass_all = trial.user_attrs.get("elite_pass_all")
    elite_min_p = trial.user_attrs.get("elite_min_p")
    elite_blocks_used = trial.user_attrs.get("elite_blocks_used")
    elite_winner_trial = trial.user_attrs.get("elite_winner_trial")

    parts = [
        f"[Trial {trial.number:03d}] value={value:.2f}",
        f"eval_mean={eval_mean:.2f}" if isinstance(eval_mean, (int, float)) else "eval_mean=n/a",
    ]
    if elite_mu is not None:
        parts.append(f"elite_mu={int(elite_mu)}")
    if elite_pass_all is not None:
        parts.append(f"elite_all_equal={bool(elite_pass_all)}")
    if elite_min_p is not None:
        parts.append(f"elite_min_p={elite_min_p}")
    if elite_blocks_used is not None:
        parts.append(f"val_blocks={int(elite_blocks_used)}")
    if elite_winner_trial is not None:
        parts.append(f"elite_winner=Trial{int(elite_winner_trial)}")
    print(" | ".join(parts))

# ============================================================
# Optuna-Study ausführen
# ============================================================
study = optuna.create_study(direction="maximize", sampler=sampler)
STUDY_REF = study
study.optimize(
    objective,
    n_trials=50,
    callbacks=[_enqueue_best_for_next, _report_trial],
    show_progress_bar=True
)

# ============================================================
# Auswertung & Plots
# ============================================================
from optuna.visualization.matplotlib import (
    plot_optimization_history, plot_slice, plot_parallel_coordinate, plot_param_importances
)

# Optuna-Standardplots (tight_layout kann bei manchen Matplotlib-Versionen warnen)
plot_optimization_history(study); plt.tight_layout(); plt.show()
plot_slice(study);                 plt.tight_layout(); plt.show()
plot_parallel_coordinate(study);   plt.tight_layout(); plt.show()
plot_param_importances(study);     plt.tight_layout(); plt.show()

# „Echte“ Zeitreihen je Hyperparameter (Trial-Index auf x-Achse)
df = study.trials_dataframe(attrs=("number","value","state","datetime_start","params"))
df = df[df["state"]=="COMPLETE"].sort_values("number")
params_cols = [c for c in df.columns if c.startswith("params_")]

for p in params_cols:
    y = df[p].to_numpy()
    x = df["number"].to_numpy()
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", label=p.replace("params_",""))
    best = np.maximum.accumulate(df["value"].to_numpy())
    ax2 = plt.gca().twinx()
    ax2.plot(x, best, linestyle="--", alpha=0.6, label="best value so far")
    plt.title(f"Parameter-Verlauf: {p.replace('params_','')}")
    plt.xlabel("Trial"); plt.ylabel("Parameterwert"); ax2.set_ylabel("Objective (best-so-far)")
    if p.endswith("learning_rate"): plt.yscale("log")
    plt.tight_layout(); plt.show()

print("Best value:", study.best_value)
print("Best params:", study.best_params)