package myorg.common;

import java.lang.Math;

public class LossFunction {

    public enum LossType {
        HINGE, SQUAREDHINGE, LOG
    }

    public static LossCalculator getLossCalculator(LossType lossType) {
        switch (lossType) {
            case HINGE:
                return new LossCalculatorHinge();
            case SQUAREDHINGE:
                return new LossCalculatorSquaredHinge();
            case LOG:
                return new LossCalculatorLog();
            default:
                return new LossCalculatorLog();
        }
    }

    public interface LossCalculator {
        float calcLoss(float z);
        float calcDLoss(float z);
    }

    public static class LossCalculatorHinge implements LossCalculator{
        public float calcLoss(float z) {
            float loss = 1.0f - z;
            return loss > 0.0f ? loss : 0.0f;
        }
        public float calcDLoss(float z) {
            float loss = 1.0f - z;
            return loss > 0.0f ? -1.0f : 0.0f;
        }
    }

    public static class LossCalculatorSquaredHinge implements LossCalculator{
        public float calcLoss(float z) {
            float loss = (1.0f - z) * (1.0f - z);
            return loss > 0.0f ? loss : 0.0f;
        }
        public float calcDLoss(float z) {
            float loss = (1.0f - z) * (1.0f - z);
            return loss > 0.0f ? -2 * (1.0f - z) : 0.0f;
        }
    }

    public static class LossCalculatorLog implements LossCalculator{
        public float calcLoss(float z) {
            return (float)(Math.log(1.0 + Math.exp(-z)));
        }
        public float calcDLoss(float z) {
            if (z < 0.0f) {
                return (float)(-1.0 / (1.0 + Math.exp(z)));
            } else {
                double ez = Math.exp(-z);
                return (float)(-ez / (ez + 1.0));
            }
        }
    }

}
