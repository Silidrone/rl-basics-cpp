#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "MC_FV.h"
#include "Policy.h"
#include "ValueStrategy.h"
#include "barto_sutton_exercises/5_1/Blackjack.h"
#include "serialization.h"

static constexpr int N_OF_EPISODES = 500;

inline void plot_v_f(TabularValueStrategy<State, Action>& value_strategy, bool usable_ace_flag) {
    matplot::vector_2d x, y, z;

    for (int player_sum = MIN_PLAYER_SUM; player_sum < MAX_SUM; ++player_sum) {
        matplot::vector_1d x_row, y_row, z_row;

        for (int dealer_face_up = ACE; dealer_face_up <= FACE_CARD; ++dealer_face_up) {
            x_row.push_back(player_sum);
            y_row.push_back(dealer_face_up);

            State state = {player_sum, dealer_face_up, usable_ace_flag};
            z_row.push_back(value_strategy.v(state));
        }

        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    auto fig = matplot::figure();
    matplot::surf(x, y, z);
    matplot::xlabel("Player's Total Sum");
    matplot::ylabel("Dealer's Face-Up Card");
    matplot::zlabel("State-Value V(s)");
    matplot::title(usable_ace_flag ? "V-Function with Usable Ace" : "V-Function without Usable Ace");
    matplot::zlim({LOSS_REWARD, WIN_REWARD});
    matplot::show();
}

inline int blackjack_main() {
    Blackjack environment;
    environment.initialize();

    auto value_strategy = new TabularValueStrategy<State, Action>();
    value_strategy->initialize(&environment);

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, 0.15);

    MC_FV<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES);

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    auto optimal_policy = policy.optimal();

    save_q_values(*value_strategy, "blackjack-optimal-Q.json");
    serialize_blackjack_policy(optimal_policy, "blackjack-optimal-policy.json");
    environment.plot_policy(optimal_policy, false);

    return 0;
}
