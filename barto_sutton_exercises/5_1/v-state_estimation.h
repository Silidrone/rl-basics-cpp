#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "MC_FV.h"
#include "ValueStrategy.h"
#include "barto_sutton_exercises/5_1/Blackjack.h"
#include "serialization.h"

static constexpr int N_OF_EPISODES = 100000;

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

inline void construct_player_policy(DeterministicPolicy<State, Action>& policy) {
    for (int player_sum = MIN_PLAYER_SUM; player_sum < MAX_SUM; ++player_sum) {
        for (int dealer_face_up = ACE; dealer_face_up <= FACE_CARD; ++dealer_face_up) {
            for (bool usable_ace : {false, true}) {
                State state = {player_sum, dealer_face_up, usable_ace};

                bool action = (player_sum < 20);

                policy.set(state, action);
            }
        }
    }
}

inline int blackjack_main() {
    Blackjack environment;
    environment.initialize();

    DeterministicPolicy<State, Action> blackjack_player_policy;

    auto value_strategy = new TabularValueStrategy<State, Action>();
    value_strategy->initialize(&environment);

    MC_FV<State, Action> mdp_solver(&environment, &blackjack_player_policy, value_strategy, DISCOUNT_RATE,
                                    N_OF_EPISODES);

    construct_player_policy(blackjack_player_policy);
    double time_taken = benchmark([&]() { mdp_solver.value_estimation(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    save_v_values(*value_strategy, "blackjack-value-function-estimation.json");
    plot_v_f(*value_strategy, false);

    return 0;
}
