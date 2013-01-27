package myudfs;

import java.lang.Math;
import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;

public class Pegasos extends BinaryOnlineClassifier {
    protected LossType lossType = LossType.LOG;
    protected LossCalculator lossCalculator = new LossCalculatorLog();
    protected float C = 1.0f;
    protected long t = 0;
    protected int k = 0;
    protected float scaleFactor = 1.0f;
    protected float snorm = 0.0f;
    protected List<Integer> labelList = null;
    protected List<Map<String, Float>> featuresList = null;

    enum LossType {
        HINGE, SQUAREDHINGE, LOG
    }

    public Pegasos() {
    }

    public Pegasos(int featureBit, LossType lossType, float C, int k) {
        this(featureBit, FeatureConvert.PARSING, lossType, C, k);
    }

    public Pegasos(int featureBit, FeatureConvert convertType, LossType lossType, float C, int k) {
        super(featureBit, convertType);
        this.lossType = lossType;
        this.lossCalculator = getLossCalculator(lossType);
        this.C  = C;
        this.t  = 0;
        this.k  = k;
        this.scaleFactor = 1.0f;
        this.snorm = 0.0f;
        this.labelList = new ArrayList<Integer>(k);
        this.featuresList = new ArrayList<Map<String, Float>>(k);
    }

    public Pegasos(List<Pegasos> classifierList) {
        if (classifierList.size() == 0) {
            return;
        }

        for (int i = 0; i < classifierList.size(); i++) {
            if (classifierList.get(i).featureBit > this.featureBit) {
                this.featureBit = classifierList.get(i).featureBit;
                this.bitMask    = classifierList.get(i).bitMask;
            }
        }
        this.convertType = classifierList.get(0).convertType;
        this.converter = getStrToIntConverter(this.convertType);
        this.bias = classifierList.get(0).bias;
        this.lossType = classifierList.get(0).lossType;
        this.lossCalculator = getLossCalculator(this.lossType);
        this.C = classifierList.get(0).C;
        this.t  = classifierList.get(0).t;
        this.k  = classifierList.get(0).k;
        this.scaleFactor = 1.0f;

        int featureNum = 1 << this.featureBit;
        this.weightArray = new float[featureNum];

        for (int i = 0; i < classifierList.size(); i++) {
            float sf = classifierList.get(i).scaleFactor;
            for (int j = 0; j < classifierList.get(i).weightArray.length; j++) {
                this.weightArray[j] += classifierList.get(i).weightArray[j] * sf;
            }
        }

        this.scaleFactor = 1.0f;
        this.snorm = 0.0f;
        for (int i = 0; i < featureNum; i++) {
            this.weightArray[i] /= classifierList.size();
            this.snorm += this.weightArray[i] * this.weightArray[i];
        }

    }

    @Override
    public void write(DataOutput out) throws IOException {
        trainWithMiniBatch();
        scaleWeightArray();
        super.write(out);

        WritableUtils.writeEnum(out, this.lossType);
        out.writeFloat(this.C);
        out.writeLong(this.t);
        out.writeInt(this.k);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.lossType = WritableUtils.readEnum(in, LossType.class);
        this.C = in.readFloat();
        this.t = in.readLong();
        this.k = in.readInt();

        this.lossCalculator = getLossCalculator(this.lossType);

        this.scaleFactor = 1.0f;
        this.snorm = 0.0f;

        int featureNum = 1 << this.featureBit;
        for (int i = 0; i < featureNum; i++) {
            this.snorm += this.weightArray[i] * this.weightArray[i];
        }
    }

    @Override
    public void updateWithPredictedValue(Integer label, Map<String, Float> features, float predictedValue) {
        update(label, features);
    }

    @Override
    public void update(Integer label, Map<String, Float> features) {
        labelList.add(label);
        featuresList.add(features);

        if (labelList.size() >= k) {
            trainWithMiniBatch();
        }
    }

    protected void trainWithMiniBatch() {
        int miniBatchSize = labelList.size();
        if (miniBatchSize == 0) {
            return;
        }

        t += 1;

        float[] predictedValues = new float[miniBatchSize];
        for (int i = 0; i < miniBatchSize; i++) {
            predictedValues[i] = predict(featuresList.get(i));
        }

        float decay = 1.0f - 1.0f / t;
        scaleFactor *= decay;
        if (scaleFactor < 1e-5) {
            scaleWeightArray();
        }

        for (int i = 0; i < miniBatchSize; i++) {
            int y = labelList.get(i) > 0 ? +1 : -1;
            Map<String, Float> features = featuresList.get(i);
            float predictedValue = predictedValues[i];
            float loss = lossCalculator.calcLoss(y * predictedValue);

            if (loss > 0.0f) {
                float eta = -(lossCalculator.calcDLoss(y * predictedValue) * C) / (miniBatchSize * t);
                for (String key : features.keySet()) {
                    int   k = converter.convert(key);
                    Float v = features.get(key);

                    float w = weightArray[k & bitMask];
                    snorm -= w * w;
                    w += eta * y * v / scaleFactor;
                    snorm += w * w;
                    weightArray[k & bitMask] = w;
                }

                if (bias > 0.0f) {
                    float w = weightArray[bitMask];
                    snorm -= w * w;
                    w += eta * y * bias / scaleFactor;
                    snorm += w * w;
                    weightArray[bitMask] = w;
                }
            }
        }

        labelList.clear();
        featuresList.clear();

        float b = (float)(Math.sqrt(C / snorm) / scaleFactor);
        if (b < 1.0f) {
            scaleFactor *= b;
        }

    }

    protected void scaleWeightArray() {
        if (scaleFactor != 1.0f) {
            int featureNum = 1 << featureBit;
            snorm = 0.0f;
            for (int i = 0; i < featureNum; i++) {
                weightArray[i] *= scaleFactor;
                snorm += weightArray[i] * weightArray[i];
            }
            scaleFactor = 1.0f;
        }
    }

    private LossCalculator getLossCalculator(LossType lossType) {
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

    interface LossCalculator {
        float calcLoss(float z);
        float calcDLoss(float z);
    }

    public class LossCalculatorHinge implements LossCalculator{
        public float calcLoss(float z) {
            float loss = 1.0f - z;
            return loss > 0.0f ? loss : 0.0f;
        }
        public float calcDLoss(float z) {
            float loss = 1.0f - z;
            return loss > 0.0f ? -1.0f : 0.0f;
        }
    }

    public class LossCalculatorSquaredHinge implements LossCalculator{
        public float calcLoss(float z) {
            float loss = (1.0f - z) * (1.0f - z);
            return loss > 0.0f ? loss : 0.0f;
        }
        public float calcDLoss(float z) {
            float loss = (1.0f - z) * (1.0f - z);
            return loss > 0.0f ? -2 * (1.0f - z) : 0.0f;
        }
    }

    public class LossCalculatorLog implements LossCalculator{
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
