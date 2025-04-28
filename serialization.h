#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "FunctionApproximator.h"
#include "ValueStrategy.h"
#include "m_types.h"
#include "m_utils.h"

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

// Generic serialization function for unordered_map with any key type
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

// Specialized serialization for pair<tuple<int, int, bool>, bool> keys
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

// Specialized serialization for TagGame state-action pair
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

template <typename State, typename Action>
bool save_q_values(TabularValueStrategy<State, Action>& strategy, const std::string& file_path) {
    try {
        serialize_to_json(strategy.get_Q(), file_path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save Q values: " << e.what() << std::endl;
        return false;
    }
}

template <typename State, typename Action>
bool save_v_values(TabularValueStrategy<State, Action>& strategy, const std::string& file_path) {
    try {
        serialize_to_json(strategy.get_v(), file_path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save V values: " << e.what() << std::endl;
        return false;
    }
}

// Generic load function for Q values - must be specialized for different State/Action types
template <typename State, typename Action>
bool load_q_values(TabularValueStrategy<State, Action>& strategy, const std::string& file_path) {
    // Base implementation for simple cases, should be specialized for complex types
    try {
        if (!std::filesystem::exists(file_path)) return false;

        std::ifstream q_input(file_path);
        nlohmann::json q_data;
        q_input >> q_data;

        for (const auto& [key_str, value] : q_data.items()) {
            // For simple cases where we can parse the key directly
            // This should be specialized for complex State/Action types
            std::cerr << "Warning: Using default parser for key: " << key_str << std::endl;
            State state;
            Action action;
            // Default implementation can't parse complex keys
            strategy.set_q(state, action, value);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load Q values: " << e.what() << std::endl;
        return false;
    }
}

// Generic load function for V values - must be specialized for different State types
template <typename State, typename Action>
bool load_v_values(TabularValueStrategy<State, Action>& strategy, const std::string& file_path) {
    // Base implementation for simple cases, should be specialized for complex types
    try {
        if (!std::filesystem::exists(file_path)) return false;

        std::ifstream v_input(file_path);
        nlohmann::json v_data;
        v_input >> v_data;

        for (const auto& [key_str, value] : v_data.items()) {
            // For simple cases where we can parse the key directly
            // This should be specialized for complex State types
            std::cerr << "Warning: Using default parser for key: " << key_str << std::endl;
            State state;
            // Default implementation can't parse complex keys
            strategy.set_v(state, value);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load V values: " << e.what() << std::endl;
        return false;
    }
}

// Specialization for TagGame Q-values
template <>
inline bool load_q_values<std::tuple<std::pair<int, int>, std::pair<int, int>, int>, std::pair<int, int>>(
    TabularValueStrategy<std::tuple<std::pair<int, int>, std::pair<int, int>, int>, std::pair<int, int>>& strategy,
    const std::string& file_path) {
    using State = std::tuple<std::pair<int, int>, std::pair<int, int>, int>;
    using Action = std::pair<int, int>;

    if (!std::filesystem::exists(file_path)) return false;

    try {
        std::ifstream q_input(file_path);
        nlohmann::json q_data;
        q_input >> q_data;

        for (const auto& [state_action_str, value] : q_data.items()) {
            State state;
            Action action;

            sscanf(state_action_str.c_str(), "([%d, %d], [%d, %d], %d), (%d, %d)", &std::get<0>(std::get<0>(state)),
                   &std::get<1>(std::get<0>(state)), &std::get<0>(std::get<1>(state)), &std::get<1>(std::get<1>(state)),
                   &std::get<2>(state), &std::get<0>(action), &std::get<1>(action));

            strategy.set_q(state, action, value);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load Q values: " << e.what() << std::endl;
        return false;
    }
}

template <typename State, typename Action>
bool save_approximator(const FunctionApproximator<State, Action>* approximator, const std::string& file_path) {
    try {
        if (!approximator) {
            std::cerr << "Error: Null approximator pointer" << std::endl;
            return false;
        }

        const auto& weights = approximator->get_weights();

        nlohmann::json j;
        j["weights"] = weights;

        std::ofstream file(output_dir + file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing JSON.");
        }
        file << j.dump(4);
        file.close();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save action-value approximator weights: " << e.what() << std::endl;
        return false;
    }
}

template <typename State, typename Action>
bool load_approximator(FunctionApproximator<State, Action>* approximator, const std::string& file_path) {
    try {
        if (!approximator) {
            std::cerr << "Error: Null approximator pointer" << std::endl;
            return false;
        }

        if (!std::filesystem::exists(file_path)) {
            std::cerr << "File does not exist: " << file_path << std::endl;
            return false;
        }

        std::ifstream input_file(file_path);
        nlohmann::json j;
        input_file >> j;

        std::vector<double> weights = j["weights"].get<std::vector<double>>();

        approximator->set_weights(weights);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load action-value approximator weights: " << e.what() << std::endl;
        return false;
    }
}

// For explicit template specialization for Blackjack's policy
inline void serialize_blackjack_policy(
    const std::unordered_map<std::tuple<int, int, bool>, bool, StateHash<std::tuple<int, int, bool>>>& policy,
    const std::string& filename) {
    nlohmann::json j;
    for (const auto& [state, action] : policy) {
        // For Blackjack: true = "hit", false = "stick"
        j[key_to_string(state)] = action ? "hit" : "stick";
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

// Generic policy serialization for non-boolean actions
template <typename State, typename Action>
typename std::enable_if<!std::is_same<Action, bool>::value && !std::is_same<Action, double>::value, void>::type
serialize_policy_to_json(const std::unordered_map<State, Action, StateHash<State>>& policy,
                         const std::string& filename) {
    nlohmann::json j;
    for (const auto& [state, action] : policy) {
        j[key_to_string(state)] = action;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}
