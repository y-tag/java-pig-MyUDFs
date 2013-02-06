package myorg.pig.evaluation;

import java.io.IOException;
import java.util.Random;

import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;

public class MyRandom extends EvalFunc<Double> {
    private Random random = null;

    public MyRandom() {
        this.random = new Random();
    }

    public MyRandom(String seed) {
        this.random = new Random(Long.parseLong(seed));
    }

    public Double exec(Tuple input) throws IOException {
        return random.nextDouble();
    }
}
