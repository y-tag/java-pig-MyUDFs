package myorg.pig.evaluation;

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

import myorg.common.ModelReader;
import myorg.regression.online.OnlineRegression;

public class PredictWithOnlineRegression extends EvalFunc<Float> {
    private String modelPath = null;
    private OnlineRegression regression = null;

    public PredictWithOnlineRegression(String modelPath) {
        this.modelPath = modelPath;
    }

    public Float exec(Tuple input) throws IOException {
        if (regression == null) {
            List<OnlineRegression> regressionList = ModelReader.readModelsFromPath(new Path(this.modelPath), OnlineRegression.class);
            regression = new OnlineRegression(regressionList);
        }
        
        if (input == null || input.size() == 0) {
            return 0.0f;
        }

        Map<String, Float> features = (Map<String, Float>)input.get(0);

        return regression.predict(features);
    }

}

