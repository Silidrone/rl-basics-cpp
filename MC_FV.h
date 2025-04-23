#pragma once

#include <limits>

#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action, typename ValueStrategyType = TabularValueStrategy<State, Action>>
class MC_FV : public GPI<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>> N;
    std::unordered_map<State, std::vector<Return>, StateHash<State>> m_returns;
    ValueStrategyType* m_value_strategy;
    Return avg_returns(const State& s) {
        if (m_returns.find(s) == m_returns.end()) {
            throw std::runtime_error("State not found in returns.");
        }

        const auto& returns = m_returns[s];

        if (returns.empty()) {
            throw std::runtime_error("No returns available for the given state.");
        }

        double total_sum = 0.0;

        for (const auto& r : returns) {
            total_sum += r;
        }

        return total_sum / static_cast<double>(returns.size());
    }

   public:
    MC_FV(MDP<State, Action>* mdp_core, Policy<State, Action>* policy, ValueStrategyType* value_strategy,
          const double discount_rate, const double number_of_episodes)
        : GPI<State, Action>(mdp_core, policy, discount_rate, number_of_episodes), m_value_strategy(value_strategy) {
        policy->initialize(mdp_core, value_strategy);
    };

    void mc_main(const std::function<void(const State&, const Action&, Return)>& update_fn) {
        int episode_n = 0;
        do {
            episode_n++;
            auto episode = this->generate_episode();

            std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>>
                first_occurence_index_map;
            for (int i = 0; i < episode.size(); i++) {
                auto [s, a, r] = episode[i];
                if (first_occurence_index_map.find({s, a}) == first_occurence_index_map.end()) {
                    first_occurence_index_map[{s, a}] = i;
                }
            }

            Return G = 0;
            for (int reverse_i = episode.size() - 1; reverse_i >= 0; --reverse_i) {
                auto [s, a, r] = episode[reverse_i];
                G = this->m_discount_rate * G + r;
                auto first_visit = first_occurence_index_map[{s, a}] == reverse_i;
                if (first_visit) {
                    update_fn(s, a, G);
                }
            }
        } while (episode_n < this->m_policy_threshold);
    }

    void policy_iteration() override {
        mc_main([this](const State& s, const Action& a, Return G) {
            N[{s, a}]++;
            auto error = G - m_value_strategy->Q(s, a);
            m_value_strategy->set_q(s, a, m_value_strategy->Q(s, a) + error / N[{s, a}]);
        });
    }

    void value_estimation() {
        mc_main([this](const State& s, const Action& a, Return G) {
            m_returns[s].push_back(G);
            m_value_strategy->set_v(s, avg_returns(s));
        });
    }
};