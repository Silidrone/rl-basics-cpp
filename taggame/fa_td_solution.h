#pragma once

#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "FA_TD.h"
#include "FunctionApproximator.h"
#include "MDPSolver.h"
#include "Policy.h"
#include "ValueStrategy.h"
#include "m_utils.h"
#include "serialization.h"
#include "taggame/TagGame.h"

constexpr double DISCOUNT_RATE = 1;
static constexpr long double N_OF_EPISODES = 50000;
static constexpr double POLICY_EPSILON = 0.1;
static constexpr double TD_ALPHA = 0.001;
static const std::string WEIGHTS_FILE = "taggame_fa_weights.json";
static const std::string POLICY_FILE = "fa_td_taggame_optimal_policy.json";

inline int taggame_main() {
    TagGame environment;
    environment.initialize();

    std::function<std::vector<double>(const State&, const Action&)> feature_extractor = [](const State& s,
                                                                                           const Action& a) {
        const auto& [predator_pos, prey_pos, distance] = s;
        const auto& [action_x, action_y] = a;

        // Constants for normalization

        std::vector<double> features;

        // Position features - normalized to [0,1] range
        features.push_back(predator_pos.first / MAX_X);
        features.push_back(predator_pos.second / MAX_Y);
        features.push_back(prey_pos.first / MAX_X);
        features.push_back(prey_pos.second / MAX_Y);

        // Distance features - normalized
        features.push_back(distance / MAX_DISTANCE);
        features.push_back(std::abs(predator_pos.first - prey_pos.first) / MAX_X);
        features.push_back(std::abs(predator_pos.second - prey_pos.second) / MAX_Y);

        // Action features - normalized using MAX_VELOCITY
        features.push_back(action_x / MAX_VELOCITY);
        features.push_back(action_y / MAX_VELOCITY);

        // Cross features - normalized
        features.push_back((action_x * (predator_pos.first - prey_pos.first)) / (MAX_X * MAX_VELOCITY));
        features.push_back((action_y * (predator_pos.second - prey_pos.second)) / (MAX_Y * MAX_VELOCITY));

        // Bias term
        features.push_back(1.0);

        return features;
    };

    int feature_dim = 12;
    auto approximator = new LinearFunctionApproximator<State, Action>(feature_dim, feature_extractor);

    auto value_strategy = new ApproximationValueStrategy<State, Action>();
    value_strategy->initialize(&environment, approximator);

    bool weights_loaded = false;
    try {
        weights_loaded = load_approximator(approximator, output_dir + WEIGHTS_FILE);
        if (weights_loaded) {
            std::cout << "Successfully loaded approximator weights from file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load approximator weights: " << e.what() << std::endl;
    }

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, POLICY_EPSILON);
    FA_TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    try {
        std::cout << "Starting policy iteration..." << std::endl;
        double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });
        std::cout << "Policy iteration completed in " << time_taken << " seconds." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during policy iteration: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred during policy iteration." << std::endl;
    }

    try {
        bool saved = save_approximator(approximator, WEIGHTS_FILE);
        if (saved) {
            std::cout << "Successfully saved approximator weights to file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to save approximator weights: " << e.what() << std::endl;
    }

    serialize_to_json(policy.optimal(), POLICY_FILE);
    std::cout << "Optimal policy saved to " << POLICY_FILE << std::endl;

    return 0;
}