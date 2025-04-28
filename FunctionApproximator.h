#pragma once
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

template <typename State, typename Action>
class FunctionApproximator {
   public:
    virtual double predict(const State& s, const Action& a) const = 0;
    virtual std::vector<double> gradient(const State& s, const Action& a) const = 0;
    virtual void update(const State& s, const Action& a, double target, double step_size) = 0;
    virtual const std::vector<double>& get_weights() const = 0;
    virtual void set_weights(const std::vector<double>& new_weights) = 0;
    virtual ~FunctionApproximator() = default;
};

template <typename State, typename Action>
class LinearFunctionApproximator : public FunctionApproximator<State, Action> {
   private:
    std::vector<double> weights;
    std::function<std::vector<double>(const State&, const Action&)> feature_extractor;

   public:
    LinearFunctionApproximator(int feature_dim, std::function<std::vector<double>(const State&, const Action&)> fe)
        : weights(feature_dim, 0.0), feature_extractor(fe) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dis(gen);
        }
    }

    double predict(const State& s, const Action& a) const override {
        auto x = feature_extractor(s, a);
        double value = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            value += weights[i] * x[i];
        }
        return value;
    }

    std::vector<double> gradient(const State& s, const Action& a) const override {
        return feature_extractor(s, a);  // for linear FA, gradient = features
    }

    void update(const State& s, const Action& a, double target, double step_size) override {
        auto x = feature_extractor(s, a);
        double prediction = predict(s, a);
        double delta = target - prediction;
        for (size_t i = 0; i < weights.size(); ++i) weights[i] += step_size * delta * x[i];
    }

    const std::vector<double>& get_weights() const override { return weights; }

    void set_weights(const std::vector<double>& new_weights) override {
        if (weights.size() != new_weights.size()) {
            throw std::invalid_argument("Weight vector size mismatch");
        }
        weights = new_weights;
    }
};