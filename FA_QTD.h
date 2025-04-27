#pragma once

#include <limits>

#include "FunctionApproximator.h"
#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action>
class FA_QTD : public GPI<State, Action> {
   protected:
    const double step_size;
    ActionValueApproximationStrategy<State, Action>* m_value_strategy;

   public:
    FA_QTD(MDP<State, Action>* mdp_core, Policy<State, Action>* policy,
           ActionValueApproximationStrategy<State, Action>* value_strategy, const double discount_rate,
           const long double policy_threshold, const double step_size)
        : GPI<State, Action>(mdp_core, policy, discount_rate, policy_threshold),
          m_value_strategy(value_strategy),
          step_size(step_size) {
        policy->initialize(mdp_core, value_strategy);
    };

    void td_main() {
        int i = 0;
        do {
            i++;
            State s = this->m_mdp->reset();

            do {
                Action a = this->m_policy->sample(s);
                auto [s_prime, r] = this->m_mdp->step(s, a);

                double target;
                if (this->m_mdp->is_terminal(s_prime)) {
                    target = r;
                } else {
                    auto [best_action, _] = this->m_policy->greedy_action(s_prime);
                    target = r + this->m_discount_rate * m_value_strategy->Q(s_prime, best_action);
                }

                m_value_strategy->get_approximator()->update(s, a, target, this->step_size);
                s = s_prime;

            } while (!this->m_mdp->is_terminal(s));
        } while (i < this->m_policy_threshold);
    }

    void policy_iteration() override { td_main(); }
};