package myudfs;

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

public class PA extends BinaryOnlineClassifier {
    protected PAType paType = PAType.PA;
    protected EtaCalculatorPA etaCalculator = new EtaCalculatorPA();
    protected float C = 1.0f;

    enum PAType {
        PA, PA1, PA2
    }

    public PA() {
    }

    public PA(int featureBit, PAType paType, float C) {
        this(featureBit, FeatureConvert.PARSING, paType, C);
    }

    public PA(int featureBit, FeatureConvert convertType, PAType paType, float C) {
        super(featureBit, convertType);
        this.paType = paType;
        this.etaCalculator = getEtaCalculator(paType);
        this.C    = C;
    }

    public PA(List<PA> classifierList) {
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
        this.paType = classifierList.get(0).paType;
        this.etaCalculator = getEtaCalculator(this.paType);
        this.C = classifierList.get(0).C;

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
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);

        this.paType = WritableUtils.readEnum(in, PAType.class);
        this.C = in.readFloat();

        this.etaCalculator = getEtaCalculator(this.paType);
    }

    @Override
    public void updateWithPredictedValue(Integer label, Map<String, Float> features, float predictedValue) {
        int y = label > 0 ? +1 : -1;
        float loss = 1.0f - y * predictedValue;

        if (loss > 0.0f) {
            float squared_norm = 0.0f;
            for (Float v : features.values()) {
                squared_norm += v * v;
            }

            float eta = etaCalculator.calc(loss, squared_norm, this.C);
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += eta * y * v;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += eta * y * bias;
            }
        }
    }

    @Override
    public void update(Integer label, Map<String, Float> features) {
        float predictedValue = 0.0f;
        float squared_norm = 0.0f;

        for (String key : features.keySet()) {
            int   k = converter.convert(key);
            Float v = features.get(key);

            predictedValue += weightArray[k & bitMask] * v;
            squared_norm += v * v;
        }

        int y = label > 0 ? +1 : -1;
        float loss = 1.0f - y * predictedValue;

        if (loss > 0.0) {
            float eta = etaCalculator.calc(loss, squared_norm, this.C);
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);
                weightArray[k & bitMask] += eta * y * v;
            }

            if (bias > 0.0f) {
                weightArray[bitMask] += eta * y * bias;
            }
        }
    }

    private EtaCalculatorPA getEtaCalculator(PAType paType) {
        switch (paType) {
            case PA1:
                return new EtaCalculatorPA1();
            case PA2:
                return new EtaCalculatorPA2();
            default:
                return new EtaCalculatorPA();
        }
    }


    public class EtaCalculatorPA {
        public float calc(float loss, float squared_norm, float C) {
            return loss / squared_norm;
        }
    }

    public class EtaCalculatorPA1 extends EtaCalculatorPA {
        @Override
        public float calc(float loss, float squared_norm, float C) {
            float eta = loss / squared_norm;
            if (C < eta) {
                eta = C;
            }
            return eta;
        }
    }

    public class EtaCalculatorPA2 extends EtaCalculatorPA {
        @Override
        public float calc(float loss, float squared_norm, float C) {
            return loss / (squared_norm + (0.5f / C));
        }
    }

}
