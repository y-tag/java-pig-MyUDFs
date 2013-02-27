package myorg.classifier.online.multiclass;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;

import myorg.common.Convert;

public class MulticlassOnlineClassifier implements Writable {
    protected int featureBit  = 0;
    protected int bitMask = 0;
    protected Map<String, Integer> labelMap = null;
    protected float[][] weightArrays = null;
    protected float bias = 0.0f;

    static final int initLabelNum = 1 << 4; // 2 ^ 4 = 16

    protected Convert.FeatureConvert convertType = Convert.FeatureConvert.PARSING;
    protected Convert.StrToIntConverter converter = Convert.getStrToIntConverter(convertType);

    public MulticlassOnlineClassifier() {
    }
    
    public MulticlassOnlineClassifier(int featureBit) {
        this(featureBit, Convert.FeatureConvert.PARSING);
    }

    public MulticlassOnlineClassifier(int featureBit, Convert.FeatureConvert convertType) {
        this.featureBit = featureBit;

        int featureNum = 1 << featureBit;
        this.bitMask = featureNum - 1;
        this.labelMap = new HashMap<String, Integer>();
        this.weightArrays = new float[initLabelNum][];
        this.bias = 0.0f;

        this.convertType = convertType;
        this.converter = Convert.getStrToIntConverter(convertType);
    }

    public MulticlassOnlineClassifier(List<MulticlassOnlineClassifier> classifierList) {
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

    public void write(DataOutput out) throws IOException {
        MapWritable map = new MapWritable();

        out.writeInt(this.featureBit);

        map.clear();
        for (String l : this.labelMap.keySet()) {
            map.put(new Text(l), new IntWritable(this.labelMap.get(l).intValue()));
        }
        map.write(out);

        out.writeInt(this.weightArrays.length);
        for (int i = 0; i < this.weightArrays.length; i++) {
            map.clear();
            float[] weight = weightArrays[i];
            if (weight != null) {
                for (int j = 0; j < weight.length; j++) {
                    if (weight[j] != 0.0f) {
                        map.put(new IntWritable(j), new FloatWritable(weight[j]));
                    }
                }
            }
            map.write(out);
        }

        out.writeFloat(this.bias);
        WritableUtils.writeEnum(out, this.convertType);
    }

    public void readFields(DataInput in) throws IOException {
        MapWritable map = new MapWritable();

        this.featureBit = in.readInt();
        int featureNum = 1 << this.featureBit;
        this.bitMask = featureNum - 1;

        map.readFields(in);

        this.labelMap = new HashMap<String, Integer>();
        for (Writable k : map.keySet()) {
            String l = ((Text)k).toString();
            int    v = ((IntWritable)(map.get(k))).get();
            this.labelMap.put(l, v);
        }

        int len = in.readInt();
        this.weightArrays = new float[len][];
        for (int i = 0; i < len; i++) {
            map.clear();
            map.readFields(in);
            if (map.size() == 0) {
                continue;
            }

            float[] weight = new float[featureNum];
            for (Writable k : map.keySet()) {
                int   j = ((IntWritable)k).get();
                float f = ((FloatWritable)(map.get(k))).get();
                weight[j & bitMask] += f;
            }
            this.weightArrays[i] = weight;
        }

        this.bias = in.readFloat();
        this.convertType = WritableUtils.readEnum(in, Convert.FeatureConvert.class);
        this.converter = Convert.getStrToIntConverter(this.convertType);
    }

    public String classify(Map<String, Float> features) {
        String maxLabel = null;
        float maxVal = 0.0f;

        for (String l : labelMap.keySet()) {
            float val = 0.0f;
            int i = labelMap.get(l).intValue();
            if (weightArrays[i] != null) {
                float[] weight = weightArrays[i];
                for (String key : features.keySet()) {
                    int   k = converter.convert(key);
                    Float v = features.get(key);
                    val += weight[k & bitMask] * v;
                }

                if (bias > 0.0f) {
                    val += weight[bitMask] * bias;
                }
            }

            if (maxLabel == null || val > maxVal) {
                maxLabel = l;
                maxVal   = val;
            }
        }

        return maxLabel;
    }

    public Map<String, Float> predict(Map<String, Float> features) {
        Map<String, Float> predictedMap = new HashMap<String, Float>();

        for (String l : labelMap.keySet()) {
            float val = 0.0f;
            int i = labelMap.get(l).intValue();
            if (weightArrays[i] != null) {
                float[] weight = weightArrays[i];
                for (String key : features.keySet()) {
                    int   k = converter.convert(key);
                    Float v = features.get(key);
                    val += weight[k & bitMask] * v;
                }

                if (bias > 0.0f) {
                    val += weight[bitMask] * bias;
                }
            }

            predictedMap.put(l, val);
        }

        return predictedMap;
    }

    public void update(String label, Map<String, Float> features) {
    }

    protected PredictedValues calcPredictedValues(String label, Map<String, Float> features) {
        PredictedValues values = new PredictedValues();
        values.maxAnotherLabel = null;
        values.maxAnotherVal   = 0.0f;
        values.labelVal        = 0.0f;

        for (String l : labelMap.keySet()) {
            float val = 0.0f;
            int i = labelMap.get(l).intValue();
            if (weightArrays[i] != null) {
                float[] weight = weightArrays[i];
                for (String key : features.keySet()) {
                    int   k = converter.convert(key);
                    Float v = features.get(key);
                    val += weight[k & bitMask] * v;
                }

                if (bias > 0.0f) {
                    val += weight[bitMask] * bias;
                }
            }

            if (l.equals(label)) {
                values.labelVal = val;
            } else if (values.maxAnotherLabel == null || val > values.maxAnotherVal) {
                values.maxAnotherLabel = l;
                values.maxAnotherVal   = val;
            }
        }

        return values;
    }

    public class PredictedValues {
        public String maxAnotherLabel;
        public float maxAnotherVal;
        public float labelVal;
    }

    protected void appendAndAllocForLabel(String label) {
        if (! labelMap.containsKey(label)) {
            int i = labelMap.size();
            labelMap.put(label, i);

            if (i >= weightArrays.length) {
                int len = weightArrays.length;
                float[][] newWeightArrays = new float[len * 2][];
                for (int j = 0; j < len; j++) {
                    newWeightArrays[j] = weightArrays[j];
                }
                weightArrays = newWeightArrays;
            }

            int featureNum = 1 << featureBit;
            weightArrays[i] = new float[featureNum];
        }
    }

    @Override
    public String toString() {
        String str = "";
        for (String l : labelMap.keySet()) {
            float val = 0.0f;
            int i = labelMap.get(l).intValue();
            if (weightArrays[i] != null) {
                float[] weight = weightArrays[i];
                for (int j = 0; j < weight.length; j++) {
                    str += l + "\t" + j + ":" + weight[j];
                    if (j < weight.length - 1) {
                        str += "\n";
                    }
                }
            }
        }
        return str;
    }

}
