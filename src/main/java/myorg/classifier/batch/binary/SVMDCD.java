package myorg.classifier.batch.binary;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Random;
import java.util.Map;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;

import myorg.common.Convert;
import myorg.common.LossFunction;

public class SVMDCD extends BinaryBatchClassifier {
    protected LossFunction.LossType lossType = LossFunction.LossType.HINGE;
    protected float C = 1.0f;

    protected int numData = 0;
    protected int[] yArray = null;
    protected Map<String, Float>[] xArray = null;

    static final int initDataCapacity = 1 << 10; // 2 ^ 10 = 1K;

    public SVMDCD() {
    }
    
    public SVMDCD(int featureBit, LossFunction.LossType lossType, float C) {
        this(featureBit, Convert.FeatureConvert.PARSING, lossType, C);
    }

    public SVMDCD(int featureBit, Convert.FeatureConvert convertType, LossFunction.LossType lossType, float C) {
        super(featureBit, convertType); this.lossType = lossType;
        this.C  = C;

        this.yArray = new int[initDataCapacity];
        this.xArray = new Map[initDataCapacity];
    }

    public SVMDCD(List<SVMDCD> classifierList) {
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
        this.converter = Convert.getStrToIntConverter(this.convertType);
        this.bias = classifierList.get(0).bias;
        this.lossType = classifierList.get(0).lossType;
        this.C = classifierList.get(0).C;

        this.yArray = new int[initDataCapacity];
        this.xArray = new Map[initDataCapacity];

        int featureNum = 1 << this.featureBit;
        this.weightArray = new float[featureNum];

        for (int i = 0; i < classifierList.size(); i++) {
            for (int j = 0; j < classifierList.get(i).weightArray.length; j++) {
                this.weightArray[j] += classifierList.get(i).weightArray[j];
            }
        }

        for (int i = 0; i < featureNum; i++) {
            this.weightArray[i] /= classifierList.size();
        }
    }

    public void write(DataOutput out) throws IOException {
        super.write(out);

        WritableUtils.writeEnum(out, this.lossType);
        out.writeFloat(this.C);
    }

    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.lossType = WritableUtils.readEnum(in, LossFunction.LossType.class);
        this.C  = in.readFloat();

        this.yArray = new int[initDataCapacity];
        this.xArray = new Map[initDataCapacity];
    }

    @Override
    public void addDatum(Integer label, Map<String, Float> features) {
        int y = label > 0 ? +1 : -1;

        int len = yArray.length;
        if (numData >= len) {
            expandArrays();
        }

        yArray[numData] = y;
        xArray[numData] = features;

        numData++;
    }

    @Override
    public void train() {
        if (numData == 0) {
            return;
        }


        float U = C;
        float D = 0.0f;
        if (lossType == LossFunction.LossType.SQUAREDHINGE) {
            U = Float.MAX_VALUE;
            D = 1.0f / (2 * C);
        }

        int len = yArray.length;
        int[] idxArray = new int[len];
        float[] alphaArray = new float[len];
        float[] qArray = new float[len];

        for (int i = 0; i < numData; i++) {
            float snorm = 0.0f;
            for (Float v : xArray[i].values()) {
                snorm += v * v;
            }

            idxArray[i] = i;
            alphaArray[i] = 0.0f;
            qArray[i] = snorm + D;
        }

        Random random = new Random(1000);

        int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = 0; i < numData; i++) {
                int j = i + random.nextInt() % (numData - i);
            }

            float maxPG = -Float.MAX_VALUE;
            float minPG =  Float.MAX_VALUE;

            for (int i = 0; i < numData; i++) {
                int idx = i;
                Map<String, Float> x = xArray[idx];
                int y = yArray[idx];

                float oldAlpha = alphaArray[idx];
                float G = yArray[idx]*predict(x) - 1.0f + D*oldAlpha;

                float PG = G;
                if (oldAlpha == 0.0f) {
                    PG = Math.min(G, 0.0f);
                } else if (oldAlpha == U) {
                    PG = Math.max(G, 0.0f);
                }

                maxPG = Math.max(PG, maxPG);
                minPG = Math.min(PG, minPG);

                if (Math.abs(PG) > 1.0e-10) {
                    float Q = qArray[idx];
                    float alpha = Math.min(Math.max(oldAlpha - G/Q, 0.0f), U);
                    alphaArray[idx] = alpha;

                    for (String key : x.keySet()) {
                        int   k = converter.convert(key);
                        Float v = x.get(key);
                        weightArray[k & bitMask] += (alpha - oldAlpha) * y * v;
                    }

                    if (bias > 0.0f) {
                        weightArray[bitMask] += (alpha - oldAlpha) * y * bias;
                    }
                }
            }

            if (maxPG - minPG < 1.0e-6) {
                break;
            }
        }

        numData = 0; // clear data
    }

    
    protected void expandArrays() {
        int len = yArray.length;

        int newLen = len * 2; // expand exponentially
        int n = initDataCapacity << 10;
        if (len >= n) {
            newLen += n; // if arrays were expanded 10 times, expand linearly
        }

        int[] newYArray   = new int[newLen];
        Map[] newXArray = new Map[newLen];

        for (int i = 0; i < numData; i++) {
            newYArray[i] = yArray[i];
            newXArray[i] = xArray[i];
        }

        yArray = newYArray;
        xArray = newXArray;
    }

}
