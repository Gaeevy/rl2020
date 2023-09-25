import numpy as np
import time

from utils import Method, WeightingMethod
from bandit_problem import BanditProblem


class Simulation:
    """Simulation class for running multiple BanditProblem runs with different parameters

    Attributes (Simulation-specific):
        _num_sim: number of BanditProblem runs to make for current Simulation
        _rewards: list of rewards from each BanditProblem run
        _best_action_taken: list of best action taken from each BanditProblem run
        """

    def __init__(
            self,
            rng: np.random.Generator,
            k: int = 10,
            method: Method = Method.EPS_GREEDY,
            eps: float = 0,
            ucb_c: float = 2,
            gradient_alpha: float = 0.1,
            gradient_with_baseline: bool = True,
            expected_value_mu: float = 0,
            initial_value: int = 0,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
            num_steps: int = 1000,
            num_sim: int = 2000,
    ) -> None:
        self._rng = rng
        self._k = k
        self._method = method
        self._eps = eps
        self._ucb_c = ucb_c
        self._gradient_alpha = gradient_alpha
        self._gradient_with_baseline = gradient_with_baseline
        self._expected_value_mu = expected_value_mu
        self._initial_value = initial_value
        self._weighting_method = weighting_method
        self._weight_coef = weight_coef
        self._non_stationary_variance = non_stationary_variance
        self._num_steps = num_steps
        self._num_sim = num_sim
        self._rewards = []
        self._best_action_taken = []

    def run(self) -> None:
        print(f"Starting {self.name}, num_sim={self._num_sim}, num_steps={self._num_steps}.")
        start_time = time.time()
        for _ in range(self._num_sim):
            bandit_problem = BanditProblem(
                rng=self._rng,
                k=self._k,
                method=self._method,
                eps=self._eps,
                ucb_c=self._ucb_c,
                gradient_alpha=self._gradient_alpha,
                gradient_with_baseline=self._gradient_with_baseline,
                expected_value_mu=self._expected_value_mu,
                initial_value=self._initial_value,
                weighting_method=self._weighting_method,
                weight_coef=self._weight_coef,
                non_stationary_variance=self._non_stationary_variance,
                num_steps=self._num_steps)
            bandit_problem.run()
            self._rewards.append(bandit_problem.rewards)
            self._best_action_taken.append(bandit_problem.best_action_taken)
        print(
            f"Finished {self.name}, num_sim={self._num_sim}, num_steps={self._num_steps}. "
            f"Took {time.time() - start_time:.2f} seconds \n"
        )

    @property
    def name(self) -> str:
        """Returns name of current Simulation based on parameters specified on Simulation"""
        non_stationary_str = (
            f", non-stat. var={self._non_stationary_variance}"
            if self._non_stationary_variance else ""
        )
        expected_value_str = (
            f", expected_value_mu={self._expected_value_mu}"
            if self._expected_value_mu else ""
        )
        eps_greed_str = f", eps={self._eps}" if self._method == Method.EPS_GREEDY else ""
        ucb_str = f", ucb_c={self._ucb_c}" if self._method == Method.UCB else ""
        optimistic_str = f", optimistic={self._initial_value}" if self._initial_value else ""
        gradient_str = (
            f", alpha={self._gradient_alpha}, baseline={self._gradient_with_baseline}"
            if self._method == Method.GRADIENT else ""
        )
        weight_str = (
            f", weight={self._weighting_method.value}, weight_coef={self._weight_coef}"
            if self._weighting_method == WeightingMethod.CONST else ""
        )
        return (
            f"Simulation method={self._method}{expected_value_str}{ucb_str}{eps_greed_str}"
            f"{optimistic_str}{gradient_str}{weight_str}{non_stationary_str}"
        )

    @property
    def avg_rewards(self) -> np.ndarray:
        """Returns average reward on each step of all BanditProblem runs for current Simulation"""
        return np.mean(np.array(self._rewards), axis=0)

    @property
    def avg_best_action_taken(self) -> np.ndarray:
        """Returns average best action taken on each step of all BanditProblem runs for current"""
        return np.mean(np.array(self._best_action_taken), axis=0)