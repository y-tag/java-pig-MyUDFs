package myorg.classifier.online.multiclass;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import myorg.common.Convert;

public class MulticlassPerceptron extends MulticlassOnlineClassifier {

    public MulticlassPerceptron() {
    }
    
    public MulticlassPerceptron(int featureBit) {
        this(featureBit, Convert.FeatureConvert.PARSING);
    }

    public MulticlassPerceptron(int featureBit, Convert.FeatureConvert convertType) {
        super(featureBit, convertType);
    }

    public MulticlassPerceptron(List<MulticlassPerceptron> classifierList) {
        int listSize = classifierList.size();
        if (listSize == 0) {
            return;
        }

        this.labelMap = new HashMap<String, Integer>();

        for (int i = 0; i < listSize; i++) {
            if (classifierList.get(i).featureBit > this.featureBit) {
                this.featureBit = classifierList.get(i).featureBit;
                this.bitMask    = classifierList.get(i).bitMask;
            }

            for (String l : classifierList.get(i).labelMap.keySet()) {
                if (! this.labelMap.containsKey(l)) {
                    int j = this.labelMap.size();
                    this.labelMap.put(l, j);
                }
            }
        }
        this.convertType = classifierList.get(0).convertType;
        this.converter = Convert.getStrToIntConverter(this.convertType);
        this.bias = classifierList.get(0).bias;

        int sizeGreaterOrEqual = Integer.highestOneBit(this.labelMap.size());
        if (sizeGreaterOrEqual < this.labelMap.size()) {
            sizeGreaterOrEqual <<= 1; // it coubld be negative, but not mind...
        }
        if (sizeGreaterOrEqual < initLabelNum) {
            sizeGreaterOrEqual = initLabelNum;
        }

        int featureNum = 1 << this.featureBit;
        this.weightArrays = new float[sizeGreaterOrEqual][];

        for (int i = 0; i < listSize; i++) {
            for (String l : classifierList.get(i).labelMap.keySet()) {
                int si = classifierList.get(i).labelMap.get(l);
                int di = this.labelMap.get(l);

                if (classifierList.get(i).weightArrays[si] == null) {
                    continue;
                }
                if (this.weightArrays[di] == null) {
                    this.weightArrays[di] = new float[featureNum];
                }

                float[] sWeight = classifierList.get(i).weightArrays[si];
                float[] dWeight = this.weightArrays[di];
                for (int j = 0; j < sWeight.length; j++) {
                    dWeight[j] += sWeight[j];
                }
            }
        }

        for (int i = 0; i < sizeGreaterOrEqual; i++) {
            if (this.weightArrays[i] == null) {
                continue;
            }
            for (int j = 0; j < featureNum; j++) {
                this.weightArrays[i][j] /= listSize;
            }
        }
    }

    @Override
    public void update(String label, Map<String, Float> features) {
        PredictedValues values = calcPredictedValues(label, features);

        if (! labelMap.containsKey(label)) {
            appendAndAllocForLabel(label);
        }

        if (values.maxAnotherLabel != null && values.labelVal - values.maxAnotherVal > 0.0f) {
            return; // not update
        }

        float[] labelWeight   = weightArrays[labelMap.get(label).intValue()];
        float[] anotherWeight = null;
        if (values.maxAnotherLabel != null) {
            anotherWeight = weightArrays[labelMap.get(values.maxAnotherLabel).intValue()];
        }

        for (String key : features.keySet()) {
            int   k = converter.convert(key);
            Float v = features.get(key);
            labelWeight[k & bitMask] += v;
            if (anotherWeight != null) {
                anotherWeight[k & bitMask] -= v;
            }
        }

        if (bias > 0.0f) {
            labelWeight[bitMask] += bias;
            if (anotherWeight != null) {
                anotherWeight[bitMask] -= bias;
            }
        }

    }

}
