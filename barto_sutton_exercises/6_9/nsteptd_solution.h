#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DerivedStochasticPolicy.h"
#include "LinearFunctionApproximator.h"
#include "NStepTD.h"
#include "WindyGridworld.h"

static constexpr int N_OF_EPISODES = 1;
static constexpr double DISCOUNT_RATE = 0.9;
static constexpr double EPSILON = 0.1;
static constexpr double ALPHA = 0.1;
static constexpr int N_STEP = 3;

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    DerivedStochasticPolicy<State, Action> policy(EPSILON);

    // --- Feature extractor for LinearFunctionApproximator: one-hot for each state
    std::function<std::vector<double>(const State&)> feature_extractor = [](const State& s) {
        std::vector<double> features(ROW_COUNT * COL_COUNT, 0.0);
        int index = s.first * COL_COUNT + s.second;
        features[index] = 1.0;
        return features;
    };

    // --- Instantiate the linear approximator
    int feature_dim = ROW_COUNT * COL_COUNT;
    auto approximator = new LinearFunctionApproximator<State>(feature_dim, feature_extractor);

    // --- Run n-step TD learning
    NStepTD<State, Action> mdp_solver(environment, &policy, approximator, DISCOUNT_RATE, N_OF_EPISODES, ALPHA, N_STEP);

    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl << std::endl;

    auto optimal_policy = mdp_solver.get_optimal_policy();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    // You won't have Q-values from the value-based NStepTD (unless you approximate Q),
    // so only serialize policy.
    serialize_to_json(optimal_policy, "windygridworld-optimal-policy.json");

    return 0;
}
