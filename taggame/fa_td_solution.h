#pragma once

#include <matplot/matplot.h>

#include <algorithm>
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
        const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = s;
        const auto& [action_x, action_y] = a;

        std::vector<double> features;

        // Raw direction and distance data
        double dx = (my_pos.first - tag_pos.first);
        double dy = (my_pos.second - tag_pos.second);
        double distance = std::sqrt(dx * dx + dy * dy);

        // Normalized direction to tagger (unit vector)
        double dir_magnitude = std::max(0.0001, std::sqrt(dx * dx + dy * dy));  // Avoid division by zero
        double dir_x = dx / dir_magnitude;
        double dir_y = dy / dir_magnitude;

        // Normalized action
        double action_magnitude = std::max(0.0001, std::sqrt(action_x * action_x + action_y * action_y));
        double norm_action_x = action_x / action_magnitude;
        double norm_action_y = action_y / action_magnitude;

        // Moving away from tagger (-1 to 1)
        double moving_away = norm_action_x * dir_x + norm_action_y * dir_y;

        // Speed calculation
        double my_speed = std::sqrt(my_vel.first * my_vel.first + my_vel.second * my_vel.second);
        double tag_speed = std::sqrt(tag_vel.first * tag_vel.first + tag_vel.second * tag_vel.second);

        // All push_backs in a row at the end
        features.push_back(moving_away);
        features.push_back(distance / MAX_DISTANCE);
        features.push_back((my_speed - tag_speed) / MAX_VELOCITY);
        features.push_back(action_magnitude / MAX_VELOCITY);
        features.push_back(my_pos.first / MAX_X);
        features.push_back(my_pos.second / MAX_Y);

        return features;
    };

    int feature_dim = 6;  // Total number of features
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

    // serialize_to_json(policy.optimal(), POLICY_FILE);
    std::cout << "Optimal policy saved to " << POLICY_FILE << std::endl;

    return 0;
}