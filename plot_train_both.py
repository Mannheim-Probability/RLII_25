# plot_train_both.py
import sys, os, glob, numpy as np, matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy, X_TIMESTEPS

def find_run_dir(path):
    # 1) Wenn im Pfad schon monitor-Dateien liegen, nimm ihn
    if glob.glob(os.path.join(path, "*monitor.csv")):
        return path
    # 2) sonst rekursiv suchen und den jüngsten Run-Ordner wählen
    candidates = glob.glob(os.path.join(path, "**", "*monitor.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Keine *monitor.csv unter {path} gefunden")
    # gruppiere nach Ordner und nimm den mit neuestem Timestamp
    by_dir = {}
    for f in candidates:
        d = os.path.dirname(f)
        by_dir.setdefault(d, []).append(f)
    run_dir = max(by_dir, key=lambda d: max(os.path.getmtime(f) for f in by_dir[d]))
    return run_dir

def load_xy(run_dir):
    x, y = ts2xy(load_results(run_dir), X_TIMESTEPS)
    return np.asarray(x), np.asarray(y)

def smooth(y, w=100):
    if len(y) < w: return y, np.arange(len(y))
    y_s = np.convolve(y, np.ones(w)/w, mode="valid")
    idx = np.arange(w-1, w-1+len(y_s))
    return y_s, idx

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_train_both.py <path1> <label1> <path2> <label2>")
        sys.exit(1)

    p1, l1, p2, l2 = sys.argv[1:5]
    d1, d2 = find_run_dir(p1), find_run_dir(p2)

    x1, y1 = load_xy(d1); y1s, i1 = smooth(y1, w=100); x1s = x1[i1]
    x2, y2 = load_xy(d2); y2s, i2 = smooth(y2, w=100); x2s = x2[i2]

    plt.figure(figsize=(7,4))
    plt.plot(x1s, y1s, label=l1)
    plt.plot(x2s, y2s, label=l2)
    plt.xlabel("Timesteps"); plt.ylabel("Reward"); plt.title("Training Reward")
    plt.legend(); plt.tight_layout(); plt.show()