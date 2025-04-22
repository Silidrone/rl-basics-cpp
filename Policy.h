#pragma once
#include <MDPSolver.h>
#include <ValueStrategy.h>

template <typename State, typename Action>
class Policy {
   protected:
    MDP<State, Action>* m_mdp;
    ValueStrategy<State, Action>* m_value_strategy;

   public:
    Policy() : m_mdp(nullptr), m_value_strategy(nullptr) {}

    virtual ~Policy() = default;

    void initialize(MDP<State, Action>* mdp, ValueStrategy<State, Action>* strategy) {
        if (!strategy) {
            throw std::invalid_argument("MDP cannot be null!");
        }

        if (!strategy) {
            throw std::invalid_argument("Value strategy cannot be null!");
        }

        m_mdp = mdp;
        m_value_strategy = strategy;
    }

    virtual Action sample(const State& s) { return std::get<0>(greedy_action(s)); }

    virtual std::tuple<Action, Return> greedy_action(const State& s) { return m_value_strategy->get_best_action(s); }
};

template <typename State, typename Action>
using DeterministicPolicy = Policy<State, Action>;

template <typename State, typename Action>
class EpsilonGreedyPolicy : public Policy<State, Action> {
   private:
    double m_epsilon;
    std::mt19937 m_generator;

   public:
    EpsilonGreedyPolicy(ValueStrategy<State, Action>* value_strategy, double epsilon)
        : Policy<State, Action>(value_strategy), m_epsilon(epsilon), m_generator(std::random_device{}()) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
    }

    Action sample(const State& s) override {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(m_generator) < m_epsilon) {
            // Random exploration
            auto actions = this->m_mdp->A(s);
            std::uniform_int_distribution<int> action_dist(0, actions.size() - 1);
            return actions[action_dist(m_generator)];
        } else {
            // Greedy exploitation - use base class implementation for greedy action
            return std::get<0>(this->greedy_action(s));
        }
    }
};