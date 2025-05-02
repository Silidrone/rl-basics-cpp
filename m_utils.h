#pragma once

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
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

template <typename Func>
double benchmark(Func&& func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

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
        } else if constexpr (std::is_same_v<State, std::tuple<std::pair<int, int>, std::pair<int, int>, std::pair<int, int>, std::pair<int, int>, bool>>) {
            const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = state;
            size_t result = 0;

            // Hash my_pos
            result ^= std::hash<int>()(my_pos.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(my_pos.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            // Hash my_vel
            result ^= std::hash<int>()(my_vel.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(my_vel.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            // Hash tag_pos
            result ^= std::hash<int>()(tag_pos.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(tag_pos.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            // Hash tag_vel
            result ^= std::hash<int>()(tag_vel.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(tag_vel.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            // Hash is_tagged
            result ^= std::hash<bool>()(is_tagged) + 0x9e3779b9 + (result << 6) + (result >> 2);

            return result;
        } else {
            return std::hash<State>()(state);
        }
    }
};

template <typename State, typename Action>
struct StateActionPairHash {
    size_t operator()(const std::pair<State, Action>& pair) const {
        return StateHash<State>()(pair.first) ^ (std::hash<Action>()(pair.second) << 1);
    }
};

namespace std {
template <typename T>
std::string to_string(const std::vector<T>& vec);
}

template <typename State, typename Action>
class MDP;

template <typename State, typename Action>
class Policy;

inline int random_value(int a, int b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b);

    return dis(gen);
}

inline double random_value(double a, double b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    return dis(gen);
}