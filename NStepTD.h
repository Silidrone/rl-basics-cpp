#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "FunctionApproximator.h"
#include "GPI.h"
#include "Policy.h"

template <typename State, typename Action>
class NStepTD : public GPI<State, Action> {
   private:
    Policy<State, Action>* m_policy;
    FunctionApproximator<State>* approximator;
    const double step_size;
    const int n;

   public:
    NStepTD(MDP<State, Action>& mdp_core, Policy<State, Action>* policy, FunctionApproximator<State>* approximator,
            double discount_rate, long double policy_threshold, double step_size, int n)
        : GPI<State, Action>(mdp_core, discount_rate, policy_threshold),
          m_policy(policy),
          approximator(approximator),
          step_size(step_size),
          n(n) {
        m_policy->initialize(this);
    }

    void td_main() {
        for (int episode = 0; episode < this->m_policy_threshold; ++episode) {
            std::vector<State> states(n + 1);
            std::vector<double> rewards(n + 1);
            int T = std::numeric_limits<int>::max();
            int t = 0;

            states[0] = this->m_mdp.reset();

            while (true) {
                // std::cerr << "t=" << t << " s=" << states[t % (n + 1)].first << "," << states[t % (n + 1)].second << "\n";
                if (t < T) {
                    Action a = m_policy->sample(states[t % (n + 1)]);
                    auto [s_next, r] = this->m_mdp.step(states[t % (n + 1)], a);
                    rewards[(t + 1) % (n + 1)] = r;
                    states[(t + 1) % (n + 1)] = s_next;
                    if (this->m_mdp.is_terminal(s_next)) {
                        T = t + 1;
                    }
                }

                int tau = t - n;
                if (tau >= 0) {
                    double G = 0.0;
                    for (int i = tau + 1; i <= std::min(tau + n, T); ++i) {
                        G += std::pow(this->m_discount_rate, i - tau - 1) * rewards[i % (n + 1)];
                    }

                    if (t < T) {
                        G += std::pow(this->m_discount_rate, n) * approximator->predict(states[t % (n + 1)]);
                    }

                    approximator->update(states[tau % (n + 1)], G, step_size);
                }

                if (tau == T - 1) {
                    break;
                }

                ++t;
            }
        }
    }

    void policy_iteration() override { td_main(); }
};
