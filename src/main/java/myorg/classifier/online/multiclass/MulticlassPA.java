package myorg.classifier.online.multiclass;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.io.WritableUtils;

import myorg.common.Convert;
import myorg.common.PACommon;

public class MulticlassPA extends MulticlassOnlineClassifier {
    protected PACommon.PAType paType = PACommon.PAType.PA;
    protected PACommon.EtaCalculatorPA etaCalculator = PACommon.getEtaCalculator(paType);
    protected float C = 1.0f;

    public MulticlassPA() {
    }
    
    public MulticlassPA(int featureBit, PACommon.PAType paType, float C) {
        this(featureBit, Convert.FeatureConvert.PARSING, paType, C);
    }

    public MulticlassPA(int featureBit, Convert.FeatureConvert convertType, PACommon.PAType paType, float C) {
        super(featureBit, convertType);
        this.paType = paType;
        this.etaCalculator = PACommon.getEtaCalculator(paType);
        this.C    = C;
    }

    public MulticlassPA(List<MulticlassPA> classifierList) {
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
        this.paType = classifierList.get(0).paType;
        this.etaCalculator = PACommon.getEtaCalculator(this.paType);
        this.C = classifierList.get(0).C;

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
    public void write(DataOutput out) throws IOException {
        super.write(out);

        WritableUtils.writeEnum(out, this.paType);
        out.writeFloat(this.C);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.paType = WritableUtils.readEnum(in, PACommon.PAType.class);
        this.C = in.readFloat();

        this.etaCalculator = PACommon.getEtaCalculator(this.paType);
    }

    @Override
    public void update(String label, Map<String, Float> features) {
        PredictedValues values = calcPredictedValues(label, features);

        if (! labelMap.containsKey(label)) {
            appendAndAllocForLabel(label);
        }

        float loss = 1.0f - (values.labelVal - values.maxAnotherVal);

        if (values.maxAnotherLabel != null && loss <= 0.0f) {
            return; // not update
        }

        float[] labelWeight   = weightArrays[labelMap.get(label).intValue()];
        float[] anotherWeight = null;
        if (values.maxAnotherLabel != null) {
            anotherWeight = weightArrays[labelMap.get(values.maxAnotherLabel).intValue()];
        }

        float squared_norm = 0.0f;
        for (Float v : features.values()) {
            squared_norm += v * v;
        }

        float eta = etaCalculator.calc(loss, 2.0f * squared_norm, this.C);
        for (String key : features.keySet()) {
            int   k = converter.convert(key);
            Float v = features.get(key);
            labelWeight[k & bitMask] += eta * v;
            if (anotherWeight != null) {
                anotherWeight[k & bitMask] -= eta * v;
            }
        }

        if (bias > 0.0f) {
            labelWeight[bitMask] += eta * bias;
            if (anotherWeight != null) {
                anotherWeight[bitMask] -= eta * bias;
            }
        }

    }

}
