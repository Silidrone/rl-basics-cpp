#pragma once
#include <vector>

template <typename State>
class FunctionApproximator {
   public:
    virtual double predict(const State& s) const = 0;
    virtual std::vector<double> gradient(const State& s) const = 0;
    virtual void update(const State& s, double target, double step_size) = 0;
    virtual ~FunctionApproximator() = default;
};
