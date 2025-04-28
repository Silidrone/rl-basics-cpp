#pragma once

#include <m_utils.h>

#include <chrono>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "m_types.h"

template <typename State, typename Action>
class MDP {
   public:
    using Transition = std::tuple<State, Reward, Probability>;
    using Dynamics =
        std::unordered_map<std::pair<State, Action>, std::vector<Transition>, StateActionPairHash<State, Action>>;

   protected:
    std::vector<State> m_S;                                                // State space: S
    std::vector<State> m_T;                                                // Terminal State space: T
    std::unordered_map<State, std::vector<Action>, StateHash<State>> m_A;  // Action space: A
    Dynamics m_dynamics;                                                   // Dynamics P function (if known)
    bool m_is_continuous;

   public:
    virtual ~MDP() = default;

    MDP(bool is_continuous = false) : m_is_continuous(is_continuous) {}

    bool is_continuous() { return m_is_continuous; }

    virtual void initialize() = 0;

    std::vector<State> S() const { return m_S; }
    std::vector<State> T() const { return m_T; }
    std::vector<Action> A(const State& s) const {
        auto it = m_A.find(s);
        if (it != m_A.end()) {
            return it->second;
        }

        throw std::out_of_range("State not found in action map");
    }
    std::unordered_map<State, std::vector<Action>, StateHash<State>> A() const { return m_A; }

    // Return all possible actions independent of state
    virtual std::vector<Action> all_possible_actions() const {
        throw std::logic_error("all_possible_actions is not implemented in this environment.");
    }

    virtual bool is_valid(const State& s, const Action& a) const {
        throw std::logic_error("is_valid is not implemented in this environment.");
    }

    std::vector<Transition> p(const State& s, const Action& a) const { return m_dynamics.at({s, a}); }
    Dynamics dynamics() const { return m_dynamics; }

    virtual State reset() { throw std::logic_error("The reset function is not available in this environment."); }

    virtual std::pair<State, Reward> step(const State& state, const Action& action) {
        throw std::logic_error("The step function is not available in this environment.");
    }

    virtual bool is_terminal(const State& state) {
        if (m_is_continuous) return false;

        return std::find(m_T.begin(), m_T.end(), state) != m_T.end();
    }

    Action random_action(const State& s) const {
        static std::mt19937 generator{std::random_device{}()};
        auto actions = this->A(s);
        if (actions.empty()) {
            throw std::runtime_error("No available actions for the given state");
        }

        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        return actions[dist(generator)];
    }
};
