import numpy as np
import matplotlib.pyplot as plt

# random_seed = np.random.randint(0, 1000)
random_seed = 795
rng = np.random.default_rng(random_seed)


class Action:
    def __init__(self, name: str) -> None:
        self.name = name
        self._initial_value = rng.normal()
        self._cumulative_value = 0
        self._action_count = 0
        self._estimated_value = 0

    def reset(self) -> None:
        self._cumulative_value = 0
        self._action_count = 0

    def act(self) -> float:
        value = rng.normal(self._initial_value, 1)
        self._cumulative_value += value
        self._action_count += 1
        self._estimated_value = self._cumulative_value / self._action_count
        return value

    @property
    def estimated_value(self) -> float:
        return self._estimated_value

    @property
    def initial_value(self) -> float:
        return self._initial_value

    def __repr__(self) -> str:
        return f"Action(name={self.name}, initial_value={self._initial_value}," \
               f" estimated_value={self._estimated_value}, action_count={self._action_count})"


class BanditProblem:
    def __init__(self, k: int = 10, eps: float = 0, num_steps: int = 1000) -> None:
        self._actions = [Action(f"Action_{i}") for i in range(1, k + 1)]
        self.best_action = max(self._actions, key=lambda b: b.initial_value)
        self.eps = eps
        self.num_steps = num_steps
        self._rewards = []
        self._best_action_taken = []

    def run(self) -> None:
        for _ in range(self.num_steps):
            self.step()

    def step(self) -> None:
        if rng.random() < self.eps:
            chosen_action = rng.choice(self._actions)
        else:
            chosen_action = max(self._actions, key=lambda b: b.estimated_value)
        best_action_taken = int(chosen_action == self.best_action)
        self._rewards.append(chosen_action.act())
        self._best_action_taken.append(best_action_taken)

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @property
    def best_action_taken(self) -> list[int]:
        return self._best_action_taken


class Simulation:
    def __init__(self, k: int = 10, eps: float = 0, num_steps: int = 1000, num_sim: int = 1000) -> None:
        self.k = k
        self.eps = eps
        self.num_steps = num_steps
        self.num_sim = num_sim
        self._rewards = []
        self._best_action_taken = []

    def run(self) -> None:
        for _ in range(self.num_sim):
            bandit_problem = BanditProblem(k=self.k, eps=self.eps, num_steps=self.num_steps)
            bandit_problem.run()
            self._rewards.append(bandit_problem.rewards)
            self._best_action_taken.append(bandit_problem.best_action_taken)

    @property
    def name(self) -> str:
        return f"Simulation eps={eps}"

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
    for eps in (0, 0.01, 0.1, 0.5):
        s = Simulation(k=10, eps=eps, num_steps=1000, num_sim=200)
        s.run()
        simulations.append(s.name)
        rewards.append(s.avg_rewards)
        best_action_taken.append(s.avg_best_action_taken)

    plt.figure(figsize=(10, 10))
    for s, r, b in zip(simulations, rewards, best_action_taken):
        plt.subplot(2, 1, 1)
        plt.plot(r, label=s)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(b, label=s)
        plt.legend()

    plt.show()

