package de.bückemeyer.backpropagation.node;

import de.bückemeyer.backpropagation.Connection;
import de.bückemeyer.backpropagation.function.DifferentiableFunction;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

public abstract class Node {
    protected Logger logger;

    private List<Connection> inputs;
    private List<Connection> outputs;

    private DifferentiableFunction function;

    public Node(DifferentiableFunction function) {
        logger = LogManager.getLogger(getClass());
        inputs = new ArrayList<Connection>();
        outputs = new ArrayList<Connection>();
        this.function = function;
    }

    public List<Connection> getInputs() {
        return inputs;
    }

    public void setInputs(List<Connection> inputs) {
        this.inputs = inputs;
    }

    public List<Connection> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<Connection> outputs) {
        this.outputs = outputs;
    }

    public DifferentiableFunction getFunction() {
        return function;
    }

    public void setFunction(DifferentiableFunction function) {
        this.function = function;
    }

    public double getOutput(){
        return function.getVal(getNetOutput());
    }

    protected double getNetOutput(){
        double output = 0;
        for (Connection con : getInputs()) {
            output += con.getLeft().getOutput() * con.getWeight();
        }

        return output;
    }

    public abstract void trainInputConnection(double learningRate);
}
