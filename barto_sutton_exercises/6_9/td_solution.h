#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "Policy.h"
#include "TD.h"
#include "ValueStrategy.h"
#include "WindyGridworld.h"
#include "serialization.h"

static constexpr int N_OF_EPISODES = 6000;
static constexpr double DISCOUNT_RATE = 0.9;  // no discounting
static constexpr double EPSILON = 0.1;
static constexpr double ALPHA = 0.1;  // learning rate

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    auto value_strategy = new TabularValueStrategy<State, Action>();
    value_strategy->initialize(&environment);

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, EPSILON);

    TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, ALPHA);

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl << std::endl;

    auto optimal_policy = policy.optimal();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    save_q_values(*value_strategy, "windygridworld-Q.json");
    serialize_to_json(optimal_policy, "windygridworld-optimal-policy.json");

    return 0;
}
