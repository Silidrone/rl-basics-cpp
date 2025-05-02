#pragma once

#include <chrono>
#include <iostream>
#include <limits>

#include "FunctionApproximator.h"
#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action>
class FA_TD : public GPI<State, Action> {
   protected:
    const double step_size;
    ApproximationValueStrategy<State, Action>* m_value_strategy;

   public:
    FA_TD(MDP<State, Action>* mdp_core, Policy<State, Action>* policy,
          ApproximationValueStrategy<State, Action>* value_strategy, const double discount_rate,
          const long double policy_threshold, const double step_size)
        : GPI<State, Action>(mdp_core, policy, discount_rate, policy_threshold),
          m_value_strategy(value_strategy),
          step_size(step_size) {
        policy->initialize(mdp_core, value_strategy);
    };

    void td_main() {
        int i = 0;
        do {  // episode loop
            i++;
            State s = this->m_mdp->reset();
            Action a = this->m_policy->sample(s);
            do {  // step loop
                auto [s_prime, r] = this->m_mdp->step(s, a);
                double q_current = m_value_strategy->get_approximator()->predict(s, a);

                if (this->m_mdp->is_terminal(s_prime)) {
                    double error = r - q_current;
                    m_value_strategy->get_approximator()->update(s, a, error, this->step_size);
                } else {
                    Action a_prime = this->m_policy->sample(s_prime);
                    double q_next = m_value_strategy->get_approximator()->predict(s_prime, a_prime);
                    double error = (r + this->m_discount_rate * q_next) - q_current;
                    m_value_strategy->get_approximator()->update(s, a, error, this->step_size);
                    a = a_prime;
                }

                s = s_prime;
            } while (!this->m_mdp->is_terminal(s));
        } while (i < this->m_policy_threshold);
    }

    void policy_iteration() override { td_main(); }
};