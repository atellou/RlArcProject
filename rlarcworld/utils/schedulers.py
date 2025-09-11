class BetaScheduler:
    def __init__(self, start: float, end: float, steps: int, internal_step: int = 0):
        """
        Initializes a new BetaScheduler instance.

        Args:
            start (float): The initial value of beta.
            end (float): The final value of beta.
            steps (int): The number of steps to reach the final value of beta.
            internal_step (int, optional): The internal step counter. Defaults to 0.
        """
        self.start = start
        self.end = end
        self.steps = steps
        self.internal_step = internal_step
        self.beta = self.start

    def step(self) -> float:
        """
        Steps the scheduler forward by one step.

        Returns:
            float: The updated value of beta.
        """
        self.internal_step += 1
        # Linear interpolation from start to end over steps, clamped at end
        progress = min(1.0, self.internal_step / max(self.steps, 1))
        self.beta = self.start + (self.end - self.start) * progress
        return self.beta

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the scheduler.

        Returns:
            dict: The state dictionary of the scheduler.
        """
        return {
            "start": self.start,
            "end": self.end,
            "steps": self.steps,
            "internal_step": self.internal_step,
            "beta": self.beta,
        }

    def load(self, state_dict: dict):
        """
        Loads the state dictionary of the scheduler.

        Args:
            state_dict (dict): The state dictionary of the scheduler.
        """
        self.start = state_dict["start"]
        self.end = state_dict["end"]
        self.steps = state_dict["steps"]
        self.internal_step = state_dict["internal_step"]
        self.beta = state_dict["beta"]
