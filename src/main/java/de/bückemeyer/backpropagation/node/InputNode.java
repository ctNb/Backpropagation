package de.bückemeyer.backpropagation.node;

public class InputNode extends Node{
    private double value;

    public InputNode() {
        super(null);
    }

    public double getOutput() {
        return value;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    @Override
    public void trainInputConnection(double learningRate) {
        throw new UnsupportedOperationException("invalid for input node");
    }
}
