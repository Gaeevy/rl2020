import numpy as np

from utils import Method, WeightingMethod


class Action:
    """Action class for k-armed bandit problem.

    Attributes:
        name: Action name
        _expected_value_mu: value that is used as mu to generate expected
            value with N(mu, 1) distribution
        _action_count: number of times action was taken
        _estimated_value: estimated value of action based on previously yielded rewards
        _weighting_method: method to use for weighting rewards while calculating estimated value.
            Could be either WeightingMethod.AVG or WeightingMethod.CONST
        _weight_coef: coefficient to use for weighting rewards while calculating
            estimated value with help of WeightingMethod.CONST method
        _non_stationary_variance: variance to use for adding non-stationary component to
            expected value of action. If 0, action is stationary.
            Adds N(0, _non_stationary_variance) random value to expected value.
        _ucb_c: coefficient to use for calculating UCB value of action
        _gradient_alpha: alpha coefficient to use for calculating gradient value
            of action preference
        _gradient_h: gradient preference of action

    """
    def __init__(
            self,
            rng: np.random.Generator,
            name: str,
            expected_value_mu: float = 0,
            initial_value: int = 0,
            ucb_c: float = 2,
            gradient_alpha: float = 0.1,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
    ) -> None:
        self._rng = rng
        self.name = name
        self._expected_value_mu = expected_value_mu
        self._expected_value = self._rng.normal(self._expected_value_mu, 1)
        self._action_count = 0
        self._estimated_value = initial_value
        self._weighting_method = weighting_method
        self._weight_coef = weight_coef
        self._non_stationary_variance = non_stationary_variance
        self._ucb_c = ucb_c
        self._gradient_alpha = gradient_alpha
        self._gradient_h = 0

    def act(self) -> float:
        """Yields reward for action. Updates estimated value of action based on weighting method"""
        value = self._rng.normal(self._expected_value, 1)
        self._action_count += 1
        if self._weighting_method == WeightingMethod.CONST:
            self._estimated_value += self._weight_coef * (value - self._estimated_value)
        elif self._weighting_method == WeightingMethod.AVG:
            self._estimated_value += (value - self._estimated_value) / self._action_count
        else:
            raise ValueError(f"Unexpected weighting method: {self._weighting_method}")
        return value

    @property
    def estimated_value(self) -> float:
        """Returns current estimated value of action"""
        return self._estimated_value

    def ucb_value(self, steps_taken) -> float:
        """Calculates UCB value of action based number of total steps taken

        Args:
            steps_taken: number of total steps taken for all actions for current BanditProblem run

        Returns:
            UCB value of action that is used for choosing action to take
        """
        if self._action_count == 0:
            return np.inf
        v = self._estimated_value + self._ucb_c * np.sqrt(np.log(steps_taken) / self._action_count)
        return v

    @property
    def expected_value(self) -> float:
        """Returns q*(a) - expected value of action"""
        return self._expected_value

    @property
    def gradient_h(self) -> float:
        """Returns current gradient preference of action"""
        return self._gradient_h

    def add_non_stationary_variance(self) -> None:
        """Adds non-stationary component to expected value of action"""
        self._expected_value += self._rng.normal(0, self._non_stationary_variance)

    def update_gradient_h(
            self,
            reward: float,
            baseline: float,
            prob: float,
            is_chosen_action: bool
    ) -> None:
        """Updates gradient preference of action.

        Args:
            reward: reward yielded on current step of BanditProblem run
            baseline: baseline value to use for calculating gradient preference
            prob: probability of exactly this action being chosen
            is_chosen_action: whether this action was chosen on current step of BanditProblem run
        """
        if is_chosen_action:
            self._gradient_h += self._gradient_alpha * (reward - baseline) * (1 - prob)
        else:
            self._gradient_h -= self._gradient_alpha * (reward - baseline) * prob

    def __repr__(self) -> str:
        gradient_h_str = f", gradient_h={self._gradient_h}" if self._gradient_h else ""
        return (
            f"Action(name={self.name}, expected_value={self._expected_value},"
            f" estimated_value={self._estimated_value}, action_count={self._action_count}"
            f"{gradient_h_str})"
        )

    def __eq__(self, other) -> bool:
        return self.name == other.name
