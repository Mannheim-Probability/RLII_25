# plot_train_multi.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, X_TIMESTEPS

def smooth(y, w=100):
    if len(y) < w:
        return y, np.arange(len(y))
    k = np.ones(w, dtype=float)/w
    y_s = np.convolve(y, k, mode="valid")
    idx = np.arange(len(y_s)) + (w-1)
    return y_s, idx

def load_xy(run_dir):
    # Erwartet einen Ordner, in dem *monitor.csv liegt.
    # (Bei RL-Zoo: z.B. logs/best_ppo/ppo/LunarLander-v3_1)
    x, y = ts2xy(load_results(run_dir), X_TIMESTEPS)
    return np.asarray(x), np.asarray(y)

def main():
    if (len(sys.argv) - 1) % 2 != 0 or len(sys.argv) < 3:
        print("Usage:\n  python plot_train_multi.py RUN1 LABEL1 [RUN2 LABEL2 ...]")
        sys.exit(1)

    args = sys.argv[1:]
    pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]

    plt.figure(figsize=(8,5))
    for run_dir, label in pairs:
        if not os.path.isdir(run_dir):
            print(f"Warnung: {run_dir} ist kein Verzeichnis – überspringe.")
            continue
        x, y = load_xy(run_dir)
        y_s, idx = smooth(y, w=100)
        x_s = x[idx]
        plt.plot(x_s, y_s, label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (smoothed, w=100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_compare.png", dpi=200)
    print("Gespeichert: train_compare.png")

if __name__ == "__main__":
    main()