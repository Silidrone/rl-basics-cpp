#pragma once
#include <functional>
#include <vector>

template <typename State>
class FunctionApproximator {
   public:
    virtual double predict(const State& s) const = 0;
    virtual std::vector<double> gradient(const State& s) const = 0;
    virtual void update(const State& s, double target, double step_size) = 0;
    virtual ~FunctionApproximator() = default;
};

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

template <typename State, typename Action>
class ActionValueFunctionApproximator {
   public:
    virtual double predict(const State& s, const Action& a) const = 0;
    virtual std::vector<double> gradient(const State& s, const Action& a) const = 0;
    virtual void update(const State& s, const Action& a, double target, double step_size) = 0;
    virtual ~ActionValueFunctionApproximator() = default;
};

template <typename State, typename Action>
class LinearActionValueFunctionApproximator : public ActionValueFunctionApproximator<State, Action> {
   private:
    std::vector<double> weights;
    std::function<std::vector<double>(const State&, const Action&)> feature_extractor;

   public:
    LinearActionValueFunctionApproximator(int feature_dim,
                                          std::function<std::vector<double>(const State&, const Action&)> fe)
        : weights(feature_dim, 0.0), feature_extractor(fe) {}

    double predict(const State& s, const Action& a) const override {
        auto x = feature_extractor(s, a);
        double value = 0.0;
        for (size_t i = 0; i < x.size(); ++i) value += weights[i] * x[i];
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
};