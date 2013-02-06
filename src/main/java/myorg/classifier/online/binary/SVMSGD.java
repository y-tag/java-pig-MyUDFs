package myorg.classifier.online.binary;

import java.lang.Math;
import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;

import myorg.common.Convert;
import myorg.common.LossFunction;

public class SVMSGD extends BinaryOnlineClassifier {
    protected LossFunction.LossType lossType = LossFunction.LossType.LOG;
    protected LossFunction.LossCalculator lossCalculator = LossFunction.getLossCalculator(lossType);
    protected float C = 1.0f;
    protected long t0 = 0;
    protected long t  = 0;
    protected float scaleFactor = 1.0f;

    public SVMSGD() {
    }

    public SVMSGD(int featureBit, LossFunction.LossType lossType, float C, long t0) {
        this(featureBit, Convert.FeatureConvert.PARSING, lossType, C, t0);
    }

    public SVMSGD(int featureBit, Convert.FeatureConvert convertType, LossFunction.LossType lossType, float C, long t0) {
        super(featureBit, convertType);
        this.lossType = lossType;
        this.lossCalculator = LossFunction.getLossCalculator(lossType);
        this.C  = C;
        this.t0 = t0;
        this.t  = 0;
        this.scaleFactor = 1.0f;
    }

    public SVMSGD(List<SVMSGD> classifierList) {
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
        this.lossCalculator = LossFunction.getLossCalculator(this.lossType);
        this.C = classifierList.get(0).C;
        this.t0 = classifierList.get(0).t;
        this.t  = classifierList.get(0).t;
        this.scaleFactor = 1.0f;

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

    @Override
    public void write(DataOutput out) throws IOException {
        scaleWeightArray();
        super.write(out);

        WritableUtils.writeEnum(out, this.lossType);
        out.writeFloat(this.C);
        out.writeLong(this.t0);
        out.writeLong(this.t);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.lossType = WritableUtils.readEnum(in, LossFunction.LossType.class);
        this.C  = in.readFloat();
        this.t0 = in.readLong();
        this.t  = in.readLong();
        this.scaleFactor = 1.0f;

        this.lossCalculator = LossFunction.getLossCalculator(this.lossType);
    }

    @Override
    public void updateWithPredictedValue(Integer label, Map<String, Float> features, float predictedValue) {
        int y = label > 0 ? +1 : -1;
        float loss = lossCalculator.calcLoss(y * predictedValue);

        t += 1;

        scaleFactor *= (1.0f - 1.0f / (t0 + t));
        if (scaleFactor < 1e-5) {
            scaleWeightArray();
        }

        if (loss > 0.0f) {
            float eta = -(lossCalculator.calcDLoss(y * predictedValue) * C) / (t0 + t);
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += eta * y * v / scaleFactor;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += eta * y * bias / scaleFactor;
            }
        }

    }

    protected void scaleWeightArray() {
        if (scaleFactor != 1.0f) {
            int featureNum = 1 << featureBit;
            for (int i = 0; i < featureNum; i++) {
                weightArray[i] *= scaleFactor;
            }
            scaleFactor = 1.0f;
        }
    }

}
