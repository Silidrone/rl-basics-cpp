#pragma once
#include "MDPSolver.h"

template <typename State, typename Action>
class ValueStrategy {
   public:
    virtual ~ValueStrategy() = default;
    virtual void initialize(MDP<State, Action>* mdp) = 0;
    virtual std::tuple<Action, Return> get_best_action(const State& s) = 0;
};

template <typename State, typename Action>
class QTableValueStrategy : public ValueStrategy<State, Action> {
   protected:
    MDPSolver<State, Action>* m_solver;
    MDP<State, Action>* m_mdp;

   public:
    QTableValueStrategy() : m_solver(nullptr), m_mdp(nullptr) {}

    void initialize(MDP<State, Action>* mdp) override { m_mdp = mdp; }

    void initialize_solver(MDPSolver<State, Action>* solver) { m_solver = solver; }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_solver) {
            throw std::logic_error("QTableValueStrategy not initialized with a solver");
        }

        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;

        for (const Action& a : m_mdp->A(s)) {
            Return candidate_return = m_solver->Q(s, a);
            if (candidate_return > max_return) {
                max_return = candidate_return;
                maximizing_action = a;
            }
        }

        return {maximizing_action, max_return};
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

    // Override the base class method to ensure proper initialization
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
};