package de.bückemeyer.backpropagation;

import de.bückemeyer.backpropagation.data.DataIterator;
import de.bückemeyer.backpropagation.data.Dataset;
import de.bückemeyer.backpropagation.function.DifferentiableFunction;
import de.bückemeyer.backpropagation.node.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network {

    private List<List<Node>> net;
    private DifferentiableFunction function;
    private double learningRate;

    private Logger logger = LogManager.getLogger(getClass());

    private DataIterator data;

    public Network(DifferentiableFunction function, double learningRate, DataIterator data) {
        net = new ArrayList<List<Node>>();
        this.function = function;
        this.learningRate = learningRate;
        this.data = data;
    }

    public void buildNetwork(int inputNodeCount, int hiddenNodeCount ,int outputNodeCount, boolean useBias, double biasValue){
        net.add(new ArrayList<Node>());
        for(int i = 0; i < inputNodeCount;i++){
            getInputLayer().add(new InputNode());
        }
        addHiddenLayer(hiddenNodeCount, useBias, biasValue);

        net.add(new ArrayList<Node>());
        BiasNode outputBias = null;
        if (useBias)
            outputBias = new BiasNode(1);
        for(int i = 0; i < outputNodeCount; i++){
            OutputNode outputNode = new OutputNode(function);
            if(useBias){
                setConnection(outputBias, outputNode);
            }
            for (Node node : net.get(net.size() - 2)){
                if(!(node instanceof BiasNode))
                setConnection(node, outputNode);
            }
            getOutputLayer().add(outputNode);
        }
    }

    private void addHiddenLayer(int countOfNodes, boolean withBias, double biasDefaultValue){
        List<Node> prevLayer = net.get(net.size()-1);
        net.add(new ArrayList<Node>());
        List<Node> currLayer = net.get(net.size()-1);
        BiasNode biasNode = null;
        if(withBias){
            biasNode = new BiasNode(biasDefaultValue);
            currLayer.add(biasNode);
        }
        for(int b = 0; b < countOfNodes; b++){
            HiddenNode node = new HiddenNode(function);
            currLayer.add(node);
            if(biasNode != null)
                setConnection(biasNode, node);
            for(int i = 0; i < prevLayer.size(); i++){
                setConnection(prevLayer.get(i), node);
            }
        }
    }

    private void setConnection(Node left, Node right){
        Connection con = new Connection(getRandomValue(), left, right);
        left.getOutputs().add(con);
        right.getInputs().add(con);
    }

    private double getRandomValue(){
        Random random = new Random();
        return random.nextDouble();
    }


    private List<Node> getInputLayer(){
        return net.get(0);
    }

    private List<Node> getOutputLayer(){
        return net.get(net.size()-1);
    }

    private List<List<Node>> getHiddenLayers(){
        List<List<Node>> layers = new ArrayList<List<Node>>();
        for(int i = 1; i < net.size()-1;i++){
            layers.add(net.get(i));
        }
        return layers;
    }

    /**
     *
     * @param input must be equal with input layer size
     * @return prediction for the given input
     */
    public double[] forwardPropagate(double... input){
        if(input.length != getInputLayer().size())
            throw new IllegalArgumentException("length of input must be equal with input layer size");

        for(int b = 0; b < getInputLayer().size();b++){
            ((InputNode) getInputLayer().get(b)).setValue(input[b]);
        }

        int i = 0;
        double[] output = new double[getOutputLayer().size()];
        for (Node node : getOutputLayer()) {
            output[i] = node.getOutput();
            i++;
        }
        resetCache();
        return output;
    }

    /**
     * resets the cached output of the CacheNodes
     */
    private void resetCache() {
        for (List<Node> layers : getHiddenLayers()) {
            for (Node node : layers) {
                if(node instanceof CachedNode){
                    ((CachedNode) node).resetCache();
                }
            }
        }
    }

    public void train(int xTimes){
        for (int b = 0; b < xTimes; b++){
            Dataset set = data.getNext();
            if(getInputLayer().size() != set.getInput().length)
                throw new IllegalArgumentException("data input size is incorrect");
            if(getOutputLayer().size() != set.getOutput().length)
                throw new IllegalArgumentException("data output size is incorrect");

            for(int i = 0; i < getInputLayer().size(); i++){
                ((InputNode) getInputLayer().get(i)).setValue(set.getInput()[i]);
            }
            double errTotal = 0;
            //feed forward
            for (int i = 0; i < getOutputLayer().size(); i++) {
                OutputNode outputNode = ((OutputNode) getOutputLayer().get(i));
                double target = set.getOutput()[i];
                outputNode.setTarget(target);
//                double output = getOutputLayer().get(i).getOutput();
//                double diff = (target - output);
                errTotal +=  outputNode.getError();
            }

            logger.debug(b +" - Error value: " + new BigDecimal(errTotal).toPlainString());

            //back propagate
            // train layers
            for (int i = net.size()-1; i > 0 ; i--) {
                for (Node node : net.get(i)) {
                    if(!(node instanceof BiasNode))
                        node.trainInputConnection(learningRate);
                }
            }
            resetCache();
        }
    }

}
