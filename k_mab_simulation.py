from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

# random_seed = np.random.randint(0, 1000)
random_seed = 795
rng = np.random.default_rng(random_seed)


class WeightingMethod(Enum):
    AVG = "avg"
    CONST = "const"

class Method(Enum):
    EPS_GREEDY = "eps_greedy"
    UCB = "ucb"


class Action:
    def __init__(
            self,
            name: str,
            initial_value: int = 0,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
            ucb_c: float = 2,
    ) -> None:
        self.name = name
        self._expected_value = rng.normal()
        self._action_count = 0
        self._estimated_value = initial_value
        self._weighting_method = weighting_method
        self._weight_coef = weight_coef
        self._non_stationary_variance = non_stationary_variance
        self.ucb_c = ucb_c

    def act(self) -> float:
        value = rng.normal(self._expected_value, 1)
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
        return self._estimated_value

    def ucb_value(self, steps_taken) -> float:
        if self._action_count == 0:
            return np.inf
        return self._estimated_value + self.ucb_c * np.sqrt(np.log(steps_taken)/self._action_count)

    @property
    def expected_value(self) -> float:
        return self._expected_value

    def add_non_stationary_variance(self) -> None:
        self._expected_value += rng.normal(0, self._non_stationary_variance)

    def __repr__(self) -> str:
        return f"Action(name={self.name}, expected_value={self._expected_value}," \
               f" estimated_value={self._estimated_value}, action_count={self._action_count})"


class BanditProblem:
    def __init__(
            self,
            method: Method = Method.EPS_GREEDY,
            k: int = 10,
            eps: float = 0,
            ucb_c: float = 2,
            num_steps: int = 1000,
            initial_value: int = 0,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
    ) -> None:
        self._non_stationary_variance = non_stationary_variance
        self._method = method
        self._actions = [
            Action(
                name=f"Action_{i}",
                initial_value=initial_value,
                ucb_c=ucb_c,
                weighting_method=weighting_method,
                weight_coef=weight_coef,
                non_stationary_variance=self._non_stationary_variance,
            ) for i in range(1, k + 1)
        ]
        self.best_action = max(self._actions, key=lambda b: b.expected_value)
        self.eps = eps
        self.num_steps = num_steps
        self._rewards = []
        self._best_action_taken = []
        self._steps_taken = 0

    def run(self) -> None:
        for _ in range(self.num_steps):
            if self._non_stationary_variance:
                for action in self._actions:
                    action.add_non_stationary_variance()

            chosen_action = getattr(self, f"choose_{self._method.value}")()

            _best_action_taken = int(chosen_action == self.best_action)
            self._rewards.append(chosen_action.act())
            self._best_action_taken.append(_best_action_taken)
            self._steps_taken += 1

    def choose_ucb(self) -> Action:
        chosen_action = max(self._actions, key=lambda a: a.ucb_value(self._steps_taken))
        return chosen_action

    def choose_eps_greedy(self) -> Action:
        if rng.random() < self.eps:
            chosen_action = rng.choice(self._actions)
        else:
            chosen_action = max(self._actions, key=lambda a: a.estimated_value)
        return chosen_action

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @property
    def best_action_taken(self) -> list[int]:
        return self._best_action_taken


class Simulation:
    def __init__(
            self,
            k: int = 10,
            method: Method = Method.EPS_GREEDY,
            eps: float = 0,
            ucb_c: float = 2,
            initial_value: int = 0,
            weighting_method: WeightingMethod = WeightingMethod.AVG,
            weight_coef: float = 0.1,
            non_stationary_variance: float = 0,
            num_steps: int = 1000,
            num_sim: int = 2000,
    ) -> None:
        self._k = k
        self._method = method
        self._eps = eps
        self._ucb_c = ucb_c
        self._initial_value = initial_value
        self._weighting_method = weighting_method
        self._weight_coef = weight_coef
        self._non_stationary_variance = non_stationary_variance
        self._num_steps = num_steps
        self._num_sim = num_sim
        self._rewards = []
        self._best_action_taken = []

    def run(self) -> None:
        for _ in range(self._num_sim):
            bandit_problem = BanditProblem(
                k=self._k,
                method=self._method,
                eps=self._eps,
                ucb_c=self._ucb_c,
                initial_value=self._initial_value,
                weighting_method=self._weighting_method,
                weight_coef=self._weight_coef,
                non_stationary_variance=self._non_stationary_variance,
                num_steps=self._num_steps)
            bandit_problem.run()
            self._rewards.append(bandit_problem.rewards)
            self._best_action_taken.append(bandit_problem.best_action_taken)

    @property
    def name(self) -> str:
        non_stationary_str = (
            f", non-stat. var={self._non_stationary_variance}"
            if self._non_stationary_variance else ""
        )
        eps_greed_str = f", eps={self._eps}" if self._method == "eps_greedy" else ""
        ucb_str = f", ucb_c={self._ucb_c}" if self._method == "ucb" else ""
        optimistic_str = f", optimistic={self._initial_value}" if self._initial_value else ""
        weight_str = (
            f", weight={self._weighting_method.value}, weight_coef={self._weight_coef}"
            if self._weighting_method == WeightingMethod.CONST else ""
        )
        return (
            f"Simulation method={self._method}{ucb_str}{eps_greed_str}{optimistic_str}"
            f"{weight_str}{non_stationary_str}"
        )

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.mean(np.array(self._rewards), axis=0)

    @property
    def avg_best_action_taken(self) -> np.ndarray:
        return np.mean(np.array(self._best_action_taken), axis=0)


if __name__ == "__main__":
    print(random_seed)

    simulations = []
    rewards = []
    best_action_taken = []

    num_steps = 20

    bandit_problem = BanditProblem(
        k=10,
        method=Method.UCB,
        num_steps=num_steps,
    )
    print("Actions:", bandit_problem._actions)
    print("Best action:", bandit_problem.best_action)
    bandit_problem.run()
    print(bandit_problem.rewards)
    print(bandit_problem.best_action_taken)


    # # eps = 0 optimistic values
    # s = Simulation(
    #     eps=0,
    #     initial_value=5,
    #     num_steps=num_steps,
    # )
    # s.run()
    # simulations.append(s.name)
    # rewards.append(s.avg_rewards)
    # best_action_taken.append(s.avg_best_action_taken)
    #
    # # eps = 0.1
    # s = Simulation(
    #     eps=0.1,
    #     num_steps=num_steps,
    # )
    # s.run()
    # simulations.append(s.name)
    # rewards.append(s.avg_rewards)
    # best_action_taken.append(s.avg_best_action_taken)

    # # ucb
    # s = Simulation(
    #     method=Method.UCB,
    #     num_steps=num_steps,
    # )
    # s.run()
    # simulations.append(s.name)
    # rewards.append(s.avg_rewards)
    # best_action_taken.append(s.avg_best_action_taken)
    #
    # plt.figure(figsize=(10, 10))
    # for s, r, b in zip(simulations, rewards, best_action_taken):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(r, label=s)
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(b, label=s)
    #     plt.legend()

    plt.show()

