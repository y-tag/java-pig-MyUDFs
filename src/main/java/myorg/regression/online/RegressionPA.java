package myorg.regression.online;

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
import myorg.common.PACommon;

public class RegressionPA extends OnlineRegression {
    protected PACommon.PAType paType = PACommon.PAType.PA;
    protected PACommon.EtaCalculatorPA etaCalculator = PACommon.getEtaCalculator(paType);
    protected float C = 1.0f;
    protected float epsilon = 0.1f;

    public RegressionPA() {
    }

    public RegressionPA(int featureBit, PACommon.PAType paType, float C, float epsilon) {
        this(featureBit, Convert.FeatureConvert.PARSING, paType, C, epsilon);
    }

    public RegressionPA(int featureBit, Convert.FeatureConvert convertType, PACommon.PAType paType, float C, float epsilon) {
        super(featureBit, convertType);
        this.paType = paType;
        this.etaCalculator = PACommon.getEtaCalculator(paType);
        this.C       = C;
        this.epsilon = epsilon;
    }

    public RegressionPA(List<RegressionPA> classifierList) {
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
        this.paType = classifierList.get(0).paType;
        this.etaCalculator = PACommon.getEtaCalculator(this.paType);
        this.C = classifierList.get(0).C;
        this.epsilon = classifierList.get(0).epsilon;

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
        super.write(out);

        WritableUtils.writeEnum(out, this.paType);
        out.writeFloat(this.C);
        out.writeFloat(this.epsilon);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.paType = WritableUtils.readEnum(in, PACommon.PAType.class);
        this.C = in.readFloat();
        this.epsilon = in.readFloat();

        this.etaCalculator = PACommon.getEtaCalculator(this.paType);
    }

    @Override
    public void updateWithPredictedValue(Float target, Map<String, Float> features, float predictedValue) {
        float loss = Math.abs(predictedValue - target) - epsilon;

        if (loss > 0.0f) {
            float squared_norm = 0.0f;
            for (Float v : features.values()) {
                squared_norm += v * v;
            }

            int sign = (target - predictedValue) > 0.0f ? +1 : -1;
            float eta = etaCalculator.calc(loss, squared_norm, this.C);
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += sign * eta * v;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += sign * eta * bias;
            }
        }
    }

    @Override
    public void update(Float target, Map<String, Float> features) {
        float predictedValue = 0.0f;
        float squared_norm = 0.0f;

        for (String key : features.keySet()) {
            int   k = converter.convert(key);
            Float v = features.get(key);

            predictedValue += weightArray[k & bitMask] * v;
            squared_norm += v * v;
        }

        float loss = Math.abs(predictedValue - target) - epsilon;

        if (loss > 0.0) {
            float eta = etaCalculator.calc(loss, squared_norm, this.C);
            int sign = (target - predictedValue) > 0.0f ? +1 : -1;
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += sign * eta * v;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += sign * eta * bias;
            }
        }
    }

}
