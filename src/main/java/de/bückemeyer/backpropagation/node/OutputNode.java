package de.bückemeyer.backpropagation.node;


import de.bückemeyer.backpropagation.function.DifferentiableFunction;

public class OutputNode extends Node {

    private double delta;

    private double target;

    public OutputNode(DifferentiableFunction function) {
        super(function);
    }

    public double getDelta() {
        return delta;
    }

    public void trainInputConnection(double learningRate) {
        double totalDiffToOut = getOutput() - target;
        double derivativeOutput = getFunction().getDerivativeVal(getOutput());
        delta = totalDiffToOut * derivativeOutput;
        for(int i = 0; i < getInputs().size();i++){
            double inputVal = getInputs().get(i).getLeft().getOutput();

            double nWeight = getInputs().get(i).getWeight() - (learningRate * delta * inputVal);

            getInputs().get(i).setWeight(nWeight);
        }
    }

    public double getTarget() {
        return target;
    }

    public void setTarget(double target) {
        this.target = target;
    }

    public double getError(){
        double diff = (target - getOutput());
        return  diff * diff * 0.5;
    }
}
