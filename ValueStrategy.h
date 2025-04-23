#pragma once
#include <string>

#include "MDP.h"

template <typename State, typename Action>
class ValueStrategy {
   public:
    virtual ~ValueStrategy() = default;
    virtual void initialize(MDP<State, Action>* mdp) = 0;

    virtual std::tuple<Action, Return> get_best_action(const State& s) = 0;
};

template <typename State, typename Action>
class TabularValueStrategy : public ValueStrategy<State, Action> {
   protected:
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q{};  // Action-value Q function
    MDP<State, Action>* m_mdp;
    bool m_strict{false};

   public:
    TabularValueStrategy() : m_mdp(nullptr) {}

    void set_strict_mode(bool strict) { m_strict = strict; }

    void initialize(MDP<State, Action>* mdp) override {
        m_mdp = mdp;

        for (const State& s : m_mdp->S()) {
            m_v[s] = 0;
            for (const Action& a : m_mdp->A(s)) {
                m_Q[{s, a}] = 0;
            }
        }

        for (const State& s : m_mdp->T()) {
            m_v[s] = 0;
            for (const Action& a : m_mdp->A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_mdp) {
            throw std::logic_error("TabularValueStrategy not initialized with an MDP");
        }

        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;

        for (const Action& a : m_mdp->A(s)) {
            Return candidate_return = Q(s, a);
            if (candidate_return > max_return) {
                max_return = candidate_return;
                maximizing_action = a;
            }
        }

        return {maximizing_action, max_return};
    }

    Return v(const State& s) const {
        auto it = m_v.find(s);
        if (it == m_v.end()) {
            if (!m_strict) return 0;
            throw std::runtime_error("Error: Invalid state provided for the v-value function.");
        }
        return it->second;
    }

    Return Q(const State& s, const Action& a) const {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) {
            if (!m_strict) return 0;
            throw std::runtime_error("Error: Invalid state-action pair provided for the Q-value function.");
        }
        return it->second;
    }

    void set_v(const State& s, Return value) { m_v[s] = value; }

    void set_q(const State& s, const Action& a, Return value) { m_Q[{s, a}] = value; }

    std::unordered_map<State, Return, StateHash<State>>& get_v() { return m_v; }

    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>& get_Q() {
        return m_Q;
    }
};
template <typename State, typename Action>
class ApproximationValueStrategy : public ValueStrategy<State, Action> {
   protected:
    FunctionApproximator<State>* m_approximator;
    MDP<State, Action>* m_mdp;

   public:
    ApproximationValueStrategy() : m_approximator(nullptr), m_mdp(nullptr) {}

    void initialize(MDP<State, Action>* mdp, FunctionApproximator<State>* approximator) {
        if (!mdp || !approximator) {
            throw std::invalid_argument("Both MDP and FunctionApproximator must be non-null");
        }
        m_mdp = mdp;
        m_approximator = approximator;
    }

    void initialize(MDP<State, Action>* mdp) override {
        throw std::logic_error("ApproximationValueStrategy requires both MDP and FunctionApproximator");
    }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_approximator || !m_mdp) {
            throw std::logic_error("ApproximationValueStrategy not properly initialized");
        }

        Action best_action;
        double best_value = std::numeric_limits<double>::lowest();

        for (const Action& a : m_mdp->A(s)) {
            auto [next_state, _] = m_mdp->step(s, a);
            double value = m_approximator->predict(next_state);
            if (value > best_value) {
                best_value = value;
                best_action = a;
            }
        }

        return {best_action, best_value};
    }

    FunctionApproximator<State>* get_approximator() const { return m_approximator; }
};