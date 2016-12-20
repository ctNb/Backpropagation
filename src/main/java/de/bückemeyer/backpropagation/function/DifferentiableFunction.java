package de.bückemeyer.backpropagation.function;

public interface DifferentiableFunction {
    double getVal(double x);
    double getDerivativeVal(double x);
}
