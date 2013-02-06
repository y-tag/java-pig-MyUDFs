package myorg.classifier.online.binary;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;

import myorg.common.Convert;

public class Perceptron extends BinaryOnlineClassifier {

    public Perceptron() {
    }

    public Perceptron(int featureBit) {
        super(featureBit);
    }

    public Perceptron(int featureBit, Convert.FeatureConvert convertType) {
        super(featureBit, convertType);
    }

    public Perceptron(List<Perceptron> classifierList) {
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
    public void updateWithPredictedValue(Integer label, Map<String, Float> features, float predictedValue) {
        int y = label > 0 ? +1 : -1;

        if (y * predictedValue <= 0.0) {
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += y * v;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += y * bias;
            }
        }

    }

}
