#pragma once

#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "MDP.h"

template <typename State, typename Action>
class Policy;

template <typename State, typename Action>
class MDPSolver {
   protected:
    MDP<State, Action>* m_mdp;
    Policy<State, Action>* m_policy;

   public:
    virtual ~MDPSolver() = default;

    MDPSolver(MDP<State, Action>* mdp, Policy<State, Action>* policy) 
        : m_mdp(mdp), m_policy(policy) {
        if (!mdp || !policy) {
            throw std::invalid_argument("MDPSolver requires non-null mdp and policy");
        }
    }

    MDP<State, Action>* mdp() { return m_mdp; }
    Policy<State, Action>* policy() { return m_policy; }

    std::vector<std::tuple<State, Action, Reward>> generate_episode() {
        std::vector<std::tuple<State, Action, Reward>> episode;
        State state = m_mdp->reset();
        bool done = false;

        while (!done) {
            Action action = m_policy->sample(state);
            auto [next_state, reward] = m_mdp->step(state, action);
            episode.emplace_back(state, action, reward);
            state = next_state;
            done = m_mdp->is_terminal(state);
        }

        return episode;
    }
};
