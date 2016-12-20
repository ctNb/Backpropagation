package de.bückemeyer.backpropagation.node;

import de.bückemeyer.backpropagation.Connection;
import de.bückemeyer.backpropagation.function.DifferentiableFunction;

public class HiddenNode extends CachedNode{

    public HiddenNode(DifferentiableFunction function) {
        super(function);
    }

    public void trainInputConnection(double learningRate) {
        double delta = 0;
        for (Connection con : getOutputs()) {
            delta += con.getOldWeight() * ((OutputNode)con.getRight()).getDelta();
        }
        double derivativeOutput = getFunction().getDerivativeVal(getOutput());
        for(int i = 0; i < getInputs().size();i++){
            double inputVal = getInputs().get(i).getLeft().getOutput();
            double nWeight = getInputs().get(i).getWeight() - (learningRate * delta * derivativeOutput * inputVal);

            getInputs().get(i).setWeight(nWeight);
        }
    }
}
