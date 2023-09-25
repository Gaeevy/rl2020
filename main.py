import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation
from utils import Method, WeightingMethod


random_seed = 42
RNG = np.random.default_rng(random_seed)


def plot_simulation_results(
    simulations: list[str],
    rewards: list[np.ndarray],
    best_action_taken: list[np.ndarray],
) -> None:
    """Plot simulation results"""
    plt.figure(figsize=(10, 10))
    for s, r, b in zip(simulations, rewards, best_action_taken):
        plt.subplot(2, 1, 1)
        plt.plot(r, label=s)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.title("Average reward on step")
        plt.subplot(2, 1, 2)
        plt.plot(b, label=s)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Best action taken %")
        plt.title("Best action taken % on step")

    plt.show()


def run_epsilon_greedy_simulation(num_steps: int = 1000, num_sim: int = 2000):
    """Run simulation with epsilon-greedy method for eps values 0, 0.01, and 0.1

    Reproduces the results from Figure 2.2 of Sutton and Barto's book
    """
    _simulations = []
    _rewards = []
    _best_action_taken = []

    for eps in [0, 0.01, 0.1]:
        s = Simulation(
            rng=RNG,
            method=Method.EPS_GREEDY,
            eps=eps,
            num_steps=num_steps,
            num_sim=num_sim,
        )
        s.run()
        _simulations.append(s.name)
        _rewards.append(s.avg_rewards)
        _best_action_taken.append(s.avg_best_action_taken)

    plot_simulation_results(_simulations, _rewards, _best_action_taken)


def run_ucb_simulation(num_steps: int = 1000, num_sim: int = 2000):
    """Run simulation with UCB method for c 2. Comparison with eps-greedy method for eps 0.1

    Reproduces the results from Figure 2.4 of Sutton and Barto's book
    """
    _simulations = []
    _rewards = []
    _best_action_taken = []

    # UCB with c=2
    s = Simulation(
        rng=RNG,
        method=Method.UCB,
        ucb_c=2,
        num_steps=num_steps,
        num_sim=num_sim,
    )
    s.run()
    _simulations.append(s.name)
    _rewards.append(s.avg_rewards)
    _best_action_taken.append(s.avg_best_action_taken)

    # Epsilon-greedy with eps=0.1
    s = Simulation(
        rng=RNG,
        method=Method.EPS_GREEDY,
        eps=0.1,
        num_steps=num_steps,
        num_sim=num_sim,
    )
    s.run()
    _simulations.append(s.name)
    _rewards.append(s.avg_rewards)
    _best_action_taken.append(s.avg_best_action_taken)

    plot_simulation_results(_simulations, _rewards, _best_action_taken)


def run_optimistic_initial_values_simulation(num_steps: int = 1000, num_sim: int = 2000):
    """Run simulation with optimistic initial values for initial value 5. Comparison with
    eps-greedy method for eps 0.1

    Reproduces the results from Figure 2.3 of Sutton and Barto's book
    """
    _simulations = []
    _rewards = []
    _best_action_taken = []

    # Optimistic initial values with initial value 5 and eps=0
    s = Simulation(
        rng=RNG,
        method=Method.EPS_GREEDY,
        eps=0,
        initial_value=5,
        num_steps=num_steps,
        num_sim=num_sim,
    )
    s.run()
    _simulations.append(s.name)
    _rewards.append(s.avg_rewards)
    _best_action_taken.append(s.avg_best_action_taken)

    # Epsilon-greedy with eps=0.1
    s = Simulation(
        rng=RNG,
        method=Method.EPS_GREEDY,
        eps=0.1,
        num_steps=num_steps,
        num_sim=num_sim,
    )
    s.run()
    _simulations.append(s.name)
    _rewards.append(s.avg_rewards)
    _best_action_taken.append(s.avg_best_action_taken)

    plot_simulation_results(_simulations, _rewards, _best_action_taken)


def run_gradient_simulation(num_steps: int = 1000, num_sim: int = 2000):
    """Run simulation with gradient method for alpha values 0.1, 0.4, and with/without baseline

    Reproduces the results from Figure 2.5 of Sutton and Barto's book
    """
    _simulations = []
    _rewards = []
    _best_action_taken = []

    alphas = [0.1, 0.4]
    baselines = [True, False]

    for alpha in alphas:
        for baseline in baselines:
            s = Simulation(
                rng=RNG,
                method=Method.GRADIENT,
                expected_value_mu=4,
                gradient_alpha=alpha,
                gradient_with_baseline=baseline,
                num_steps=num_steps,
                num_sim=num_sim,
            )
            s.run()
            _simulations.append(s.name)
            _rewards.append(s.avg_rewards)
            _best_action_taken.append(s.avg_best_action_taken)

    plot_simulation_results(_simulations, _rewards, _best_action_taken)


if __name__ == "__main__":
    # run_epsilon_greedy_simulation(
    #     num_steps=1000,
    #     num_sim=2000,
    # )

    # run_ucb_simulation(
    #     num_steps=1000,
    #     num_sim=2000,
    # )

    # run_optimistic_initial_values_simulation(
    #     num_steps=1000,
    #     num_sim=2000,
    # )

    run_gradient_simulation(
        num_steps=1000,
        num_sim=2000,
    )

