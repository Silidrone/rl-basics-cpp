#pragma once

#include <limits>

#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action, typename ValueStrategyType = TabularValueStrategy<State, Action>>
class TD : public GPI<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>> N;
    const double step_size;
    ValueStrategyType* m_value_strategy;

   public:
    TD(MDP<State, Action>* mdp_core, Policy<State, Action>* policy, ValueStrategyType* value_strategy,
       const double discount_rate, const long double policy_threshold, const double step_size)
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
                if (this->m_mdp->is_terminal(s_prime)) {
                    m_value_strategy->set_q(s, a, 
                        m_value_strategy->Q(s, a) + this->step_size * (r - m_value_strategy->Q(s, a)));
                } else {
                    Action a_prime = this->m_policy->sample(s_prime);
                    m_value_strategy->set_q(s, a,
                        m_value_strategy->Q(s, a) +
                        this->step_size * (r + this->m_discount_rate * m_value_strategy->Q(s_prime, a_prime) - m_value_strategy->Q(s, a)));
                    a = a_prime;
                }

                s = s_prime;
            } while (!this->m_mdp->is_terminal(s));
        } while (i < this->m_policy_threshold);  // m_policy_threshold represents the # of episodes before termination
    }

    void policy_iteration() override { td_main(); }
};