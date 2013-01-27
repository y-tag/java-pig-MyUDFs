package myudfs;

import java.lang.Exception;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.DataInputStream;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;

import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;

public class ClassifyWithBinaryOnlineClassifierVoting extends EvalFunc<Float> {
    private String modelPath = null;
    private List<BinaryOnlineClassifier> classifierList = null;

    public ClassifyWithBinaryOnlineClassifierVoting(String modelPath) {
        this.modelPath = modelPath;
    }

    public Float exec(Tuple input) throws IOException {
        if (classifierList == null) {
            classifierList =  ModelReader.readModelsFromPath(new Path(this.modelPath), BinaryOnlineClassifier.class);
        }
        
        if (input == null || input.size() == 0) {
            return 0.0f;
        }

        Map<String, Float> features = (Map<String, Float>)input.get(0);

        int posCount = 0;
        int negCount = 0;
        float sumPrediction = 0.0f;
        for (BinaryOnlineClassifier classifier : classifierList) {
            float predictValue = classifier.predict(features);
            sumPrediction += predictValue;
            if (predictValue >= 0.0f) {
                posCount += 1;
            } else {
                negCount += 1;
            }
        }

        sumPrediction = (sumPrediction >= 0.0f) ? 1.0f : -1.0f;
        return (posCount == negCount) ? sumPrediction : ((posCount > negCount) ? 1.0f : -1.0f);
    }

}

