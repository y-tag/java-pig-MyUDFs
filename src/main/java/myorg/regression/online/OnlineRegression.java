package myorg.regression.online;

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

public class OnlineRegression implements Writable {
    protected int featureBit  = 0;
    protected int bitMask = 0;
    protected float[] weightArray = null;
    protected float bias = 0.0f;

    protected Convert.FeatureConvert convertType = Convert.FeatureConvert.PARSING;
    protected Convert.StrToIntConverter converter = Convert.getStrToIntConverter(convertType);

    public OnlineRegression() {
    }
    
    public OnlineRegression(int featureBit) {
        this(featureBit, Convert.FeatureConvert.PARSING);
    }

    public OnlineRegression(int featureBit, Convert.FeatureConvert convertType) {
        this.featureBit = featureBit;

        int featureNum = 1 << featureBit;
        this.bitMask = featureNum - 1;
        this.weightArray = new float[featureNum];
        this.bias = 0.0f;

        this.convertType = convertType;
        this.converter = Convert.getStrToIntConverter(convertType);
    }

    public OnlineRegression(List<OnlineRegression> regressionList) {
        if (regressionList.size() == 0) {
            return;
        }

        for (int i = 0; i < regressionList.size(); i++) {
            if (regressionList.get(i).featureBit > this.featureBit) {
                this.featureBit = regressionList.get(i).featureBit;
                this.bitMask    = regressionList.get(i).bitMask;
            }
        }
        this.convertType = regressionList.get(0).convertType;
        this.converter = Convert.getStrToIntConverter(this.convertType);
        this.bias = regressionList.get(0).bias;

        int featureNum = 1 << this.featureBit;
        this.weightArray = new float[featureNum];

        for (int i = 0; i < regressionList.size(); i++) {
            for (int j = 0; j < regressionList.get(i).weightArray.length; j++) {
                this.weightArray[j] += regressionList.get(i).weightArray[j];
            }
        }

        for (int i = 0; i < featureNum; i++) {
            this.weightArray[i] /= regressionList.size();
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
        out.writeFloat(this.bias);
        WritableUtils.writeEnum(out, this.convertType);
    }

    public void readFields(DataInput in) throws IOException {
        MapWritable map = new MapWritable();

        this.featureBit = in.readInt();
        map.readFields(in);
        this.bias = in.readFloat();
        this.convertType = WritableUtils.readEnum(in, Convert.FeatureConvert.class);

        int featureNum = 1 << this.featureBit;
        this.bitMask = featureNum - 1;
        this.weightArray = new float[featureNum];

        for (Writable k : map.keySet()) {
            int   i = ((IntWritable)k).get();
            float f = ((FloatWritable)(map.get(k))).get();
            weightArray[i & bitMask] += f;
        }
        
        this.converter = Convert.getStrToIntConverter(this.convertType);
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

        if (bias > 0.0f) {
            val += weightArray[bitMask] * bias;
        }
        return val;
    }

    public void updateWithPredictedValue(Float target, Map<String, Float> features, float predictedValue) {
    }

    public void update(Float target, Map<String, Float> features) {
        float predictedValue = predict(features);
        updateWithPredictedValue(target, features, predictedValue);
    }

    @Override
    public String toString() {
        String str = "";
        for (int i = 0; i < weightArray.length; i++) {
            str += i + ":" + weightArray[i];
            if (i < weightArray.length - 1) {
                str += "\n";
            }
        }
        return str;
    }

}
