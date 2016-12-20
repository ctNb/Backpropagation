package de.bückemeyer.backpropagation.function;

public class Sigmoid implements DifferentiableFunction{
    public double getVal(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double getDerivativeVal(double x) {
        return x*(1-x);
    }
}
