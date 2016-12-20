package de.bückemeyer.backpropagation;

import de.bückemeyer.backpropagation.data.example.AndDataset;
import de.bückemeyer.backpropagation.data.example.XorDataset;
import de.bückemeyer.backpropagation.function.Sigmoid;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.math.BigDecimal;

public class App
{
    private static Logger logger = LogManager.getLogger(App.class);
    public static void main( String[] args ) {
        Network net = new Network(new Sigmoid(), 0.3, new AndDataset());

        net.buildNetwork(2,5,1, true, 1);

        net.train(1000000);
        // try to predict
        predict(net);
    }

    public static void predict(Network net){
        for (int a = 0; a <= 1; a++){
            for (int b = 0; b <= 1; b++){
                double[] output = net.forwardPropagate(a, b);
                String log = a + "\t" + b + " |\t";
                logger.info(log + new BigDecimal(output[0]).toPlainString());
            }
        }
    }
}
