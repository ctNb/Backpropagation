package de.bückemeyer.backpropagation.node;

public class BiasNode extends InputNode{

    public BiasNode(double value) {
        setValue(value);
    }

    public double getOutput() {
        return getValue();
    }

    @Override
    public void trainInputConnection(double learningRate) {
        throw new UnsupportedOperationException("not valid for bias node");
    }
}
