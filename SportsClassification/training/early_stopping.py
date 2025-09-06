class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        reset_after: int = 0,
        T_mult: float | int = 1,
        threshold: float = 1e-4,
        mode: str = "min",
        threshold_mode: str = "rel",
        *args,
        **kwargs
    ):
        """
        Early stopping to terminate training when a monitored metric stops improving.

        Args:
            patience (int): Number of epochs to wait after the last improvement before stopping.
            threshold (float): Minimum change to qualify as an improvement.
                If `threshold_mode` is "rel", this is a relative change (percentage of best score).
                If `threshold_mode` is "abs", this is an absolute change.
            mode (str): "min" if lower metric is better (e.g., loss), "max" if higher metric is better (e.g., accuracy).
            threshold_mode (str): "rel" for relative threshold, "abs" for absolute threshold.
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.threshold_mode = threshold_mode
        self.best_score: float | None = None
        self.counter = 0
        self.total_counter = 0
        self.should_stop = False
        self.reset_after = reset_after
        self.T_mult = T_mult

    def __call__(self, current_score: float):
        self.total_counter += 1
        if self.best_score is None:
            self.best_score = current_score
            return
        
        if self.reset_after > 0 and self.total_counter >= self.reset_after:
            self.total_counter = 0
            self.counter = 0
            self.best_score = None
            self.reset_after = max(self.reset_after, int(self.reset_after * self.T_mult))
            print("Resetting patience")
            return

        # Determine effective threshold
        threshold = self.threshold
        if self.threshold_mode == "rel":
            threshold *= abs(self.best_score)

        improvement = (
            current_score < self.best_score - threshold
            if self.mode == "min"
            else current_score > self.best_score + threshold
        )

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
