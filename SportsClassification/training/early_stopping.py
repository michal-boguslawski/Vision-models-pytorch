class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
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
        self.should_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None:
            self.best_score = current_score
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
