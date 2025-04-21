#pragma once
#include <functional>

#include "FunctionApproximator.h"

template <typename State>
class LinearFunctionApproximator : public FunctionApproximator<State> {
   private:
    std::vector<double> weights;
    std::function<std::vector<double>(const State&)> feature_extractor;

   public:
    LinearFunctionApproximator(int feature_dim, std::function<std::vector<double>(const State&)> fe)
        : weights(feature_dim, 0.0), feature_extractor(fe) {}

    double predict(const State& s) const override {
        auto x = feature_extractor(s);
        double value = 0.0;
        for (size_t i = 0; i < x.size(); ++i) value += weights[i] * x[i];
        return value;
    }

    std::vector<double> gradient(const State& s) const override {
        return feature_extractor(s);  // for linear FA, gradient = features
    }

    void update(const State& s, double target, double step_size) override {
        auto x = feature_extractor(s);
        double prediction = predict(s);
        double delta = target - prediction;
        for (size_t i = 0; i < weights.size(); ++i) weights[i] += step_size * delta * x[i];
    }
};
