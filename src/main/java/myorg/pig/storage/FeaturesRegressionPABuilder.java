package myorg.pig.storage;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import org.apache.pig.StoreFunc;
import org.apache.pig.PigException;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;

import myorg.common.Convert;
import myorg.common.PACommon;
import myorg.common.ModelReader;
import myorg.regression.online.RegressionPA;

public class FeaturesRegressionPABuilder extends StoreFunc {
    protected RecordWriter writer = null;
    private int builderFeatureBit = 20;
    private Convert.FeatureConvert builderConvertType = Convert.FeatureConvert.HASHING;
    private PACommon.PAType builderPAType = PACommon.PAType.PA;
    private float builderC = 1.0f;
    private float builderEpsilon = 0.1f;
    private String modelPath = null;

    public FeaturesRegressionPABuilder() {
    }

    public FeaturesRegressionPABuilder(String modelPath) {
        this.modelPath = modelPath;
    }

    public FeaturesRegressionPABuilder(String featureBit, String convertType, String paType, String C, String epsilon) {
        this.builderFeatureBit = Integer.parseInt(featureBit);
        this.builderC = Float.parseFloat(C);
        this.builderEpsilon = Float.parseFloat(epsilon);

        if (convertType.equals("PARSING")) {
            this.builderConvertType = Convert.FeatureConvert.PARSING;
        } else {
            this.builderConvertType = Convert.FeatureConvert.HASHING;
        }

        if (paType.equals("PA1")) {
            this.builderPAType = PACommon.PAType.PA1;
        } else if (paType.equals("PA2")) {
            this.builderPAType = PACommon.PAType.PA2;
        } else if (paType.equals("PALOG")) {
            this.builderPAType = PACommon.PAType.PALOG;
        } else {
            this.builderPAType = PACommon.PAType.PA;
        }

    }

    @Override
    public void putNext(Tuple f) throws IOException {

        Object field;
        try {
            field = f.get(0);
        } catch (ExecException ee) {
            throw ee;
        }
        Float key = (Float)field;

        try {
            field = f.get(1);
        } catch (ExecException ee) {
            throw ee;
        }
        Map<String, Float> val = (Map<String, Float>)field;

        try {
            writer.write(key, val);
        } catch (InterruptedException e) {
            throw new IOException(e);
        }

    }

    @Override
    public OutputFormat getOutputFormat() {
        return new FeaturesRegressionPAOutputFormat();
    }

    @Override
    public void prepareToWrite(RecordWriter writer) {
        this.writer = writer;
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(RegressionPA.class);

        SequenceFileOutputFormat.setOutputPath(job, new Path(location));
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    }

    public class FeaturesRegressionPAOutputFormat extends FileOutputFormat<Float, Map<String, Float>> {
        private SequenceFileOutputFormat<NullWritable, RegressionPA> outputFormat = null;

        public FeaturesRegressionPAOutputFormat() {
            outputFormat = new SequenceFileOutputFormat<NullWritable, RegressionPA>();
        }

        @Override
        public void checkOutputSpecs(JobContext context) throws IOException {
            outputFormat.checkOutputSpecs(context);
        }

        @Override
        public OutputCommitter getOutputCommitter(TaskAttemptContext context) throws IOException {
            return outputFormat.getOutputCommitter(context);
        }

        @Override
        public RecordWriter<Float, Map<String, Float>> getRecordWriter(
                TaskAttemptContext context) throws IOException, InterruptedException {
            return new FeaturesRegressionPARecordWriter(outputFormat.getRecordWriter(context), builderFeatureBit, builderConvertType, builderPAType, builderC, builderEpsilon, modelPath);
        }

    }

    public class FeaturesRegressionPARecordWriter extends RecordWriter<Float, Map<String, Float>> {

        private RecordWriter writer     = null;
        private RegressionPA regression = null;

        public FeaturesRegressionPARecordWriter(RecordWriter<NullWritable, RegressionPA> writer, int featureBit, Convert.FeatureConvert convertType, PACommon.PAType paType, float C, float epsilon, String modelPath) {
            this.writer     = writer;

            if (modelPath == null) {
                this.regression = new RegressionPA(featureBit, convertType, paType, C, epsilon);
            } else {
                try {
                    List<RegressionPA> classifierList = ModelReader.readModelsFromPath(new Path(modelPath), RegressionPA.class);
                    this.regression = new RegressionPA(classifierList);
                } catch(Exception e) {
                    this.regression = new RegressionPA(featureBit, convertType, paType, C, epsilon);
                }
            }
        }

        @Override
        public void close(TaskAttemptContext context) throws IOException, InterruptedException {
            writer.write(NullWritable.get(), regression);
            writer.close(context);
        }

        @Override
        public void write(Float k, Map<String, Float> v) throws IOException, InterruptedException {
            regression.update(k, v);
        }

    }

}

