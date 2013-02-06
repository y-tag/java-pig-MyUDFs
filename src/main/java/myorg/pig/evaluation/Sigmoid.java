package myorg.pig.evaluation;

import java.lang.Math;
import java.io.IOException;

import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;

public class Sigmoid extends EvalFunc<Double> {

    public Sigmoid() {
    }

    public Double exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return 0.0;
        }

        Double x = (Double)input.get(0);

        if (x > 0.0) {
            return 1.0 / (1.0 + Math.exp(-x));
        } else {
            double ex = Math.exp(x);
            return ex / (ex + 1.0);
        }

    }
}
