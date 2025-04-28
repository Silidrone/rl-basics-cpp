#pragma once

#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "MDPSolver.h"
#include "Policy.h"
#include "TD.h"
#include "ValueStrategy.h"
#include "m_utils.h"
#include "serialization.h"
#include "taggame/TagGame.h"

constexpr double DISCOUNT_RATE = 1;
static constexpr long double N_OF_EPISODES = 50000;
static constexpr double POLICY_EPSILON = 0.12;
static constexpr double TD_ALPHA = 0.28;
static const std::string Q_INPUT_FILE = "taggame_q_function.json";
static const std::string POLICY_INPUT_FILE = "taggame_optimal_policy.json";

inline int taggame_main() {
    TagGame environment;

    // Create a TabularValueStrategy
    auto value_strategy = new TabularValueStrategy<State, Action>();
    value_strategy->initialize(&environment);

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, POLICY_EPSILON);
    TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    environment.initialize();
    load_q_values(*value_strategy, output_dir + Q_INPUT_FILE);

    try {
        mdp_solver.policy_iteration();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during policy iteration. Ignoring and proceeding: " << e.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred during policy iteration. Ignoring and proceeding." << std::endl;
    }

    save_q_values(*value_strategy, Q_INPUT_FILE);
    serialize_to_json(policy.optimal(), POLICY_INPUT_FILE);
    return 0;
}
