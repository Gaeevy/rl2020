import numpy as np

from action import Action
from utils import Method, WeightingMethod


class BanditProblem:
    """k-armed bandit problem class

    Attributes (except ones that already described in Action class):
        _method: method to use for choosing action. Could be either Method.EPS_GREEDY,
            Method.UCB or Method.GRADIENT
        _actions: list of actions to choose from. Each action is an instance of Action class.
            List of actions is generated on BanditProblem initialization
        _best_action: action with highest expected value. Is used for calculating best
            action taken statistics
        _eps: epsilon value to use for choosing action with Method.EPS_GREEDY method
        _num_steps: number of steps to take for current BanditProblem run
        _gradient_with_baseline: whether to use baseline value for calculating gradient preference
        _rewards: list of rewards yielded on each step of BanditProblem run
        _best_action_taken: list of 1s and 0s that indicate whether best action was taken on each
            step of BanditProblem run
        _gradient_probs: list of probabilities of each action being chosen on each step of
            BanditProblem run. Is used for calculating gradient preference of action.
            It is being updated on each step of BanditProblem run
        _steps_taken: current number of steps taken for current BanditProblem run.
            It is being updated on each step of BanditProblem run
        _cur_avg_reward: current average reward yielded on each step of BanditProblem run.
            It is being updated on each step of BanditProblem run
        """

    def __init__(
            self,
            rng: np.random.Generator,
            method: Method = Method.EPS_GREEDY,
            k: int = 10,
            eps: float = 0,
            ucb_c: float = 2,
            gradient_alpha: float = 0.1,
            gradient_with_baseline: bool = True,
            num_steps: int = 1000,
            expected_value_mu: float = 0,
            initial_value: int = 0,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
    ) -> None:
        self._rng = rng
        self._non_stationary_variance = non_stationary_variance
        self._method = method
        self._actions = [
            Action(
                rng=self._rng,
                name=f"Action_{i}",
                expected_value_mu=expected_value_mu,
                initial_value=initial_value,
                ucb_c=ucb_c,
                gradient_alpha=gradient_alpha,
                weighting_method=weighting_method,
                weight_coef=weight_coef,
                non_stationary_variance=self._non_stationary_variance,
            ) for i in range(1, k + 1)
        ]
        self._best_action = max(self._actions, key=lambda b: b.expected_value)
        self._eps = eps
        self._num_steps = num_steps
        self._gradient_with_baseline = gradient_with_baseline
        self._rewards = []
        self._best_action_taken = []
        self._gradient_probs = []
        self._steps_taken = 0
        self._cur_avg_reward = 0

    def run(self) -> None:
        """Runs BanditProblem for specified number of steps.

        On each step of BanditProblem run:
            1. Adds non-stationary component to expected value of each action
            2. Chooses action to take based on method specified on BanditProblem initialization
            3. Runs action.act() method to yield reward and update estimated value of action
            4. Updates average reward yielded on each step of BanditProblem run
            5. Updates gradient preference of action if Method.GRADIENT is used
            6. Updates lists of rewards and best action taken
            """
        for _ in range(self._num_steps):
            if self._non_stationary_variance:
                for action in self._actions:
                    action.add_non_stationary_variance()

            # choose action based on method (eps-greedy, ucb, gradient)
            chosen_action = getattr(self, f"choose_{self._method.value}")()

            _reward = chosen_action.act()
            _best_action_taken = int(chosen_action == self._best_action)
            self._cur_avg_reward += (_reward - self._cur_avg_reward) / (self._steps_taken + 1)

            # update gradient preference of action if Method.GRADIENT is used
            _baseline = self._cur_avg_reward if self._gradient_with_baseline else 0
            if self._method == Method.GRADIENT:
                for action, prob in zip(self._actions, self._gradient_probs):
                    action.update_gradient_h(
                        reward=_reward,
                        baseline=_baseline,
                        prob=prob,
                        is_chosen_action=(action == chosen_action),
                    )

            self._rewards.append(_reward)
            self._best_action_taken.append(_best_action_taken)
            self._steps_taken += 1

    def choose_ucb(self) -> Action:
        """Chooses action to take based on UCB method"""
        chosen_action = max(self._actions, key=lambda a: a.ucb_value(self._steps_taken))
        return chosen_action

    def choose_eps_greedy(self) -> Action:
        """Chooses action to take based on epsilon-greedy method"""
        if self._rng.random() < self._eps:
            chosen_action = self._rng.choice(self._actions)
        else:
            chosen_action = max(self._actions, key=lambda a: a.estimated_value)
        return chosen_action

    def choose_gradient(self) -> Action:
        """Chooses action to take based on gradient method. Updates list of probabilities of
        each action to be choosen on current step of BanditProblem run"""
        h_sum = sum([np.exp(a.gradient_h) for a in self._actions])
        self._gradient_probs = [np.exp(a.gradient_h) / h_sum for a in self._actions]
        chosen_action = self._rng.choice(self._actions, p=self._gradient_probs)
        return chosen_action

    @property
    def rewards(self) -> list[float]:
        """Returns list of rewards yielded during BanditProblem run"""
        return self._rewards

    @property
    def best_action_taken(self) -> list[int]:
        """Returns list of 1s and 0s that indicate whether best action was taken on each step
        of BanditProblem run"""
        return self._best_action_taken
