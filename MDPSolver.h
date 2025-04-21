#pragma once

#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "FunctionApproximator.h"
#include "MDP.h"

template <typename State, typename Action>
class MDPSolver {
   protected:
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDP<State, Action> &m_mdp;
    bool m_strict;

   public:
    MDPSolver() : m_strict(false) {}
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDP<State, Action> &mdp) : m_mdp(mdp) {}

    virtual void initialize() {
        for (const State &s : this->m_mdp.S()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }

        for (const State &s : this->m_mdp.T()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    void set_strict_mode(bool s) { m_strict = s; }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>> get_Q() { return m_Q; }

    Return v(State s) {
        auto it = m_v.find(s);
        if (it == m_v.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error("Error: Invalid state provided for the v-value function.");
        }
        return it->second;
    }

    Return Q(State s, Action a) {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error(
                "Error: Invalid state-action pair provided for the Q-value "
                "function.");
        }

        return it->second;
    }

    std::unordered_map<State, Action, StateHash<State>> get_optimal_policy() {
        std::unordered_map<State, Action, StateHash<State>> optimal_policy;

        for (const auto &[state_action, _] : this->m_Q) {
            const State &s = state_action.first;
            if (optimal_policy.find(s) == optimal_policy.end()) {
                auto [best_action, _] = this->best_action(s);
                optimal_policy[s] = best_action;
            }
        }

        return optimal_policy;
    }

    bool load_Q_from_file(std::string file_path) {
        if (!std::filesystem::exists(file_path)) return false;

        std::ifstream q_input(file_path);
        nlohmann::json q_data;
        q_input >> q_data;

        for (const auto &[state_action_str, value] : q_data.items()) {
            State state;
            Action action;

            sscanf(state_action_str.c_str(), "([%d, %d], [%d, %d], %d), (%d, %d)", &std::get<0>(std::get<0>(state)),
                   &std::get<1>(std::get<0>(state)), &std::get<0>(std::get<1>(state)), &std::get<1>(std::get<1>(state)),
                   &std::get<2>(state), &std::get<0>(action), &std::get<1>(action));

            this->set_q(state, action, value);
        }

        return true;
    }

    MDP<State, Action> &mdp() { return m_mdp; }
};
