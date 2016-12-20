package de.bückemeyer.backpropagation.node;

import de.bückemeyer.backpropagation.function.DifferentiableFunction;

public abstract class CachedNode extends Node{
    private boolean isOutputCached = false;
    private double cachedOutput;
    public CachedNode(DifferentiableFunction function) {
        super(function);
    }

    @Override
    public double getOutput() {
        if(isOutputCached)
            return cachedOutput;
        cachedOutput = super.getOutput();
        isOutputCached = true;
        return cachedOutput;
    }

    public void resetCache(){
        isOutputCached = false;
    }

}
