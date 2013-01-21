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

import com.google.common.hash.Hashing;

public class BinaryOnlineClassifier implements Writable {
    protected int featureBit  = 0;
    protected int bitMask = 0;
    protected float[] weightArray = null;

    protected FeatureConvert convertType = FeatureConvert.PARSING;
    protected StrToIntConverter converter = new StrToIntConverterWithParsing();

    enum FeatureConvert {
        PARSING, HASHING
    }

    public BinaryOnlineClassifier() {
    }
    
    public BinaryOnlineClassifier(int featureBit) {
        this(featureBit, FeatureConvert.PARSING);
    }

    public BinaryOnlineClassifier(int featureBit, FeatureConvert convertType) {
        this.featureBit = featureBit;

        int featureNum = 1 << featureBit;
        this.bitMask = featureNum - 1;
        this.weightArray = new float[featureNum];

        this.convertType = convertType;
        this.converter = getStrToIntConverter(convertType);
    }

    public BinaryOnlineClassifier(List<BinaryOnlineClassifier> classifierList) {
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
        MapWritable map = new MapWritable();
        for (int i = 0; i < weightArray.length; i++) {
            if (weightArray[i] != 0.0f) {
                map.put(new IntWritable(i), new FloatWritable(weightArray[i]));
            }
        }

        out.writeInt(this.featureBit);
        map.write(out);
        WritableUtils.writeEnum(out, this.convertType);
    }

    public void readFields(DataInput in) throws IOException {
        MapWritable map = new MapWritable();

        this.featureBit = in.readInt();
        map.readFields(in);
        this.convertType = WritableUtils.readEnum(in, FeatureConvert.class);

        int featureNum = 1 << this.featureBit;
        this.bitMask = featureNum - 1;
        this.weightArray = new float[featureNum];

        for (Writable k : map.keySet()) {
            int   i = ((IntWritable)k).get();
            float f = ((FloatWritable)(map.get(k))).get();
            weightArray[i & bitMask] += f;
        }
        
        this.converter = getStrToIntConverter(this.convertType);
    }

    public float predict(Map<String, Float> features) {
        float val = 0.0f;
        if (weightArray != null) {
            for (String key : features.keySet()) {
                int   k = converter.convert(key);
                Float v = features.get(key);

                val += weightArray[k & bitMask] * v;
            }
        }
        return val;
    }

    public void updateWithPredictedValue(Integer label, Map<String, Float> features, float predictedValue) {
    }

    public void update(Integer label, Map<String, Float> features) {
        float predictedValue = predict(features);
        updateWithPredictedValue(label, features, predictedValue);
    }

    protected StrToIntConverter getStrToIntConverter(FeatureConvert convertType) {
        switch (convertType) {
            case PARSING:
                return new StrToIntConverterWithParsing();
            case HASHING:
                return new StrToIntConverterWithHashing();
            default:
                return new StrToIntConverterWithHashing();
        }
    }

    interface StrToIntConverter {
        int convert(String str);
    }

    public class StrToIntConverterWithParsing implements StrToIntConverter {
        public int convert(String str) {
            return Integer.parseInt(str);
        }
    }

    public class StrToIntConverterWithHashing implements StrToIntConverter {
        public int convert(String str) {
            return Hashing.murmur3_32().hashString(str).asInt();
        }
    }

}
