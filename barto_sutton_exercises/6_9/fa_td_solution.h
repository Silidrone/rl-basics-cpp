#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "FA_TD.h"
#include "FunctionApproximator.h"
#include "Policy.h"
#include "ValueStrategy.h"
#include "WindyGridworld.h"
#include "serialization.h"

static constexpr int N_OF_EPISODES = 6000;
static constexpr double DISCOUNT_RATE = 0.9;
static constexpr double EPSILON = 0.1;
static constexpr double ALPHA = 0.1;

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    std::function<std::vector<double>(const State&)> feature_extractor = [](const State& s) {
        std::vector<double> features(ROW_COUNT * COL_COUNT, 0.0);
        int index = s.first * COL_COUNT + s.second;
        features[index] = 1.0;
        return features;
    };

    int feature_dim = ROW_COUNT * COL_COUNT;
    auto approximator = new LinearFunctionApproximator<State>(feature_dim, feature_extractor);

    auto value_strategy = new ApproximationValueStrategy<State, Action>();
    value_strategy->initialize(&environment, approximator);

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, EPSILON);

    FA_TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, ALPHA);

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl << std::endl;

    auto optimal_policy = policy.optimal();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    serialize_to_json(optimal_policy, "windygridworld-optimal-policy.json");

    return 0;
}
