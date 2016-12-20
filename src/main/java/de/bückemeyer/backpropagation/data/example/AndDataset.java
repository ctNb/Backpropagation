package de.bückemeyer.backpropagation.data.example;

import de.bückemeyer.backpropagation.data.DataIterator;
import de.bückemeyer.backpropagation.data.Dataset;

public class AndDataset implements DataIterator{
    private int i;

    private double[][] input;
    private double[][] output;

    public AndDataset() {
        i = 0;
        input = new double[][]{{1,1},
                {1,0},
                {0,1},
                {0,0}};
        output = new double[][]{{1},{0},{0},{0}};
    }

    public Dataset getNext() {
        Dataset data = new Dataset(input[i%4], output[i%4]);
        i++;
        return data;
    }
}
