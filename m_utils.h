#pragma once

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "FunctionApproximator.h"
#include "m_types.h"

static const std::string output_dir = "output/";

double pow(double b, int p);
double next_poisson(double lambda);
double poisson_probability(int k, double lambda);
int random_value(int, int);
double random_value(double, double);

// Benchmark function
template <typename Func>
double benchmark(Func&& func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

// ExtractInnerType definition
template <typename T>
struct ExtractInnerType {
    using type = T;
};

template <typename T>
struct ExtractInnerType<std::vector<T>> {
    using type = T;
};

namespace std {
template <>
struct hash<std::pair<int, int>> {
    size_t operator()(const std::pair<int, int>& action) const {
        size_t seed = 0;

        seed ^= std::hash<int>()(action.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(action.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};
}  // namespace std

template <typename State>
struct StateHash {
    size_t operator()(const State& state) const {
        if constexpr (std::is_same_v<State, std::vector<typename ExtractInnerType<State>::type>>) {
            size_t result = 0;
            for (const auto& val : state) {
                result ^= std::hash<typename ExtractInnerType<State>::type>()(val) + 0x9e3779b9 + (result << 6) +
                          (result >> 2);
            }
            return result;
        } else if constexpr (std::is_same_v<State, std::tuple<int, int, bool>>) {
            const auto& [a, b, c] = state;
            return std::hash<int>()(a) ^ (std::hash<int>()(b) << 1) ^ (std::hash<bool>()(c) << 2);
        } else if constexpr (std::is_same_v<State, std::tuple<std::pair<int, int>, std::pair<int, int>, int>>) {
            const auto& [vec1, vec2, d] = state;
            size_t result = 0;

            result ^= std::hash<int>()(vec1.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(vec1.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            result ^= std::hash<int>()(vec2.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(vec2.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            result ^= std::hash<int>()(d) + 0x9e3779b9 + (result << 6) + (result >> 2);

            return result;
        } else {
            return std::hash<State>()(state);
        }
    }
};

template <typename T>
std::string key_to_string(const T& key);

template <>
inline std::string key_to_string<int>(const int& key) {
    return std::to_string(key);
}

template <>
inline std::string key_to_string<std::vector<int>>(const std::vector<int>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

template <>
inline std::string key_to_string<std::tuple<int, int, bool>>(const std::tuple<int, int, bool>& key) {
    const auto& [a, b, c] = key;
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " + (c ? "true" : "false") + ")";
}

template <>
inline std::string key_to_string<std::pair<std::tuple<int, int, bool>, bool>>(
    const std::pair<std::tuple<int, int, bool>, bool>& key) {
    const auto& [state, action] = key;
    const auto& [a, b, c] = state;
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " + (c ? "true" : "false") + ")," +
           (action ? "hit" : "stick");
}

template <>
inline std::string key_to_string<std::tuple<std::pair<int, int>, std::pair<int, int>, int>>(
    const std::tuple<std::pair<int, int>, std::pair<int, int>, int>& key) {
    const auto& [vec1, vec2, dist] = key;

    auto vec_to_string = [](const std::pair<int, int>& vec) {
        return "(" + std::to_string(static_cast<int>(vec.first)) + ", " + std::to_string(static_cast<int>(vec.second)) +
               ")";
    };

    return "{" + vec_to_string(vec1) + ", " + vec_to_string(vec2) + ", " + std::to_string(dist) + "}";
}

template <>
inline std::string key_to_string<std::pair<int, int>>(const std::pair<int, int>& key) {
    return "(" + std::to_string(key.first) + ", " + std::to_string(key.second) + ")";
}

template <>
inline std::string key_to_string<std::pair<std::pair<int, int>, std::pair<int, int>>>(
    const std::pair<std::pair<int, int>, std::pair<int, int>>& key) {
    return "((" + std::to_string(key.first.first) + ", " + std::to_string(key.first.second) + "), (" +
           std::to_string(key.second.first) + ", " + std::to_string(key.second.second) + "))";
}

template <typename State, typename Action>
struct StateActionPairHash {
    size_t operator()(const std::pair<State, Action>& pair) const {
        return StateHash<State>()(pair.first) ^ (std::hash<Action>()(pair.second) << 1);
    }
};
template <typename Key, typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<Key, Value, Hash>& map, const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        j[key_to_string(key)] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<std::tuple<int, int, bool>, Value, Hash>& map,
                       const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [a, b, c] = key;
        j[key_to_string(a) + "," + key_to_string(b) + "," + (c ? "true" : "false")] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<std::pair<std::tuple<int, int, bool>, bool>, Value, Hash>& map,
                       const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [state, action] = key;
        const auto& [a, b, c] = state;
        j["(" + key_to_string(a) + "," + key_to_string(b) + "," + (c ? "true" : "false") + ")," +
          (action ? "hit" : "stick")] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(
    const std::unordered_map<std::pair<std::tuple<std::pair<int, int>, std::pair<int, int>, int>, std::pair<int, int>>,
                             Value, Hash>& map,
    const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [state, action] = key;
        const auto& [vec1, vec2, integer] = state;

        std::string key_string = "([" + std::to_string(vec1.first) + ", " + std::to_string(vec1.second) + "], " + "[" +
                                 std::to_string(vec2.first) + ", " + std::to_string(vec2.second) + "], " +
                                 std::to_string(integer) + "), " + "(" + std::to_string(action.first) + ", " +
                                 std::to_string(action.second) + ")";

        j[key_string] = value;
    }

    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }

    file << j.dump(4);
    file.close();
}

namespace std {
template <typename T>
std::string to_string(const std::vector<T>& vec);
}

template <typename State, typename Action>
class MDP;

template <typename State, typename Action>
class Policy;

template <typename State, typename Action>
std::vector<std::tuple<State, Action, Reward>> generate_episode(MDP<State, Action>&, Policy<State, Action>*);

template <typename State, typename Action>
std::tuple<Action, Return> greedy_action_approximated(MDP<State, Action>& mdp,
                                                      FunctionApproximator<State>* approximator, const State& s) {
    if (!approximator) {
        throw std::logic_error("FunctionApproximator pointer is null.");
    }

    const auto& actions = mdp.A(s);
    if (actions.empty()) {
        throw std::runtime_error("No available actions from current state.");
    }

    Action best_action = actions[0];
    double best_value = -std::numeric_limits<double>::infinity();

    for (const Action& a : actions) {
        State next_state = mdp.step(s, a).first;
        double value = approximator->predict(next_state);

        if (value > best_value) {
            best_value = value;
            best_action = a;
        }
    }

    return {best_action, best_value};
}

#include "MDPSolver.h"

enum class GreedyStrategy { DETERMINISTIC, STOCHASTIC, FUNCTION_APPROXIMATED };

template <typename State, typename Action>
std::tuple<Action, Return> greedy_action_deterministic(MDPSolver<State, Action>& mdp_solver, const State& s) {
    Return max_return = std::numeric_limits<Return>::lowest();
    Action maximizing_action;

    for (const auto& [state_action, value] : mdp_solver->m_Q) {
        if (state_action.first == s) {
            if (value > max_return) {
                max_return = value;
                maximizing_action = state_action.second;
            }
        }
    }

    return {maximizing_action, max_return};
}

template <typename State, typename Action>
std::tuple<Action, Return> greedy_action_stochastic(MDPSolver<State, Action>& mdp_solver, const State& s) {
    Return max_return = std::numeric_limits<Return>::lowest();
    Action maximizing_action;
    for (Action a : mdp_solver.mdp().A(s)) {
        Return candidate_return = mdp_solver.Q(s, a);
        if (candidate_return > max_return) {
            max_return = candidate_return;
            maximizing_action = a;
        }
    }

    return {maximizing_action, max_return};
}

// TODO: I do not like this. MDPSolver and Policy as well as these "greedy_action" functions need to be refactored.
template <typename State, typename Action>
std::tuple<Action, Return> greedy_action(GreedyStrategy strategy, const State& s, MDPSolver<State, Action>& mdp_solver,
                                         FunctionApproximator<State>* approximator = nullptr) {
    switch (strategy) {
        case GreedyStrategy::DETERMINISTIC:
            return greedy_action_deterministic(mdp_solver, s);

        case GreedyStrategy::STOCHASTIC:
            return greedy_action_stochastic(mdp_solver, s);

        case GreedyStrategy::FUNCTION_APPROXIMATED:
            if (!approximator) {
                throw std::invalid_argument("Approximator required for FUNCTION_APPROXIMATED strategy.");
            }
            return greedy_action_approximated(mdp_solver.mdp(), approximator, s);

        default:
            throw std::invalid_argument("Unknown greedy strategy.");
    }
}

template <typename State, typename Action>
Action random_action(MDPSolver<State, Action>& mdp_solver, const State& s) {
    std::mt19937 generator{std::random_device{}()};
    auto actions = mdp_solver.mdp().A(s);
    if (!actions.empty()) {
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        return actions[dist(generator)];
    }

    std::vector<Action> q_derived_actions;

    for (const auto& [state_action, value] : mdp_solver.m_Q) {
        if (state_action.first == s) {
            q_derived_actions.push_back(state_action.second);
        }
    }

    if (q_derived_actions.empty()) {
        throw std::runtime_error("random_action_fallback: No actions found in Q-table for the given state.");
    }

    std::uniform_int_distribution<int> dist(0, q_derived_actions.size() - 1);
    return q_derived_actions[dist(generator)];
}

template <typename State, typename Action>
std::unordered_map<State, Action, StateHash<State>> get_optimal_policy(MDPSolver<State, Action>& mdp_solver) {
    std::unordered_map<State, Action, StateHash<State>> optimal_policy;

    for (const auto& [state_action, _] : mdp_solver->m_Q) {
        const State& s = state_action.first;
        if (optimal_policy.find(s) == optimal_policy.end()) {
            auto [best_action, _] = greedy_action(s);
            optimal_policy[s] = best_action;
        }
    }

    return optimal_policy;
}