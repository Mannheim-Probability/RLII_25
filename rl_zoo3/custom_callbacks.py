from stable_baselines3.common.callbacks import BaseCallback

class ExpDecayLRCallback(BaseCallback):
    def __init__(self, lr0=3e-4, final_factor=1e-2, verbose=0):
        super().__init__(verbose)
        self.lr0 = float(lr0)
        self.final_factor = float(final_factor)

    def _on_rollout_end(self) -> None:
        # progress_remaining: 1 -> 0
        pr = float(self.model._current_progress_remaining)
        new_lr = self.lr0 * (self.final_factor ** (1.0 - pr))
        for g in self.model.policy.optimizer.param_groups:
            g["lr"] = new_lr
        if self.verbose:
            print(f"[LR-Decay] lr -> {new_lr:.3g}")

    def _on_step(self) -> bool:
        # Muss existieren, sonst bleibt die Klasse abstrakt
        return True