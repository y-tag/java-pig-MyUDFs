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
import myorg.common.ModelReader;
import myorg.classifier.online.multiclass.MulticlassPerceptron;

public class FeaturesMulticlassPerceptronBuilder extends StoreFunc {
    protected RecordWriter writer = null;
    private int builderFeatureBit = 20;
    private Convert.FeatureConvert builderConvertType = Convert.FeatureConvert.HASHING;
    private String modelPath = null;

    public FeaturesMulticlassPerceptronBuilder() {
    }

    public FeaturesMulticlassPerceptronBuilder(String modelPath) {
        this.modelPath = modelPath;
    }

    public FeaturesMulticlassPerceptronBuilder(String featureBit, String convertType) {
        this.builderFeatureBit = Integer.parseInt(featureBit);

        if (convertType.equals("PARSING")) {
            this.builderConvertType = Convert.FeatureConvert.PARSING;
        } else {
            this.builderConvertType = Convert.FeatureConvert.HASHING;
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
        String key = (String)field;

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
        return new FeaturesMulticlassPerceptronOutputFormat();
    }

    @Override
    public void prepareToWrite(RecordWriter writer) {
        this.writer = writer;
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(MulticlassPerceptron.class);

        SequenceFileOutputFormat.setOutputPath(job, new Path(location));
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    }

    public class FeaturesMulticlassPerceptronOutputFormat extends FileOutputFormat<String, Map<String, Float>> {
        private SequenceFileOutputFormat<NullWritable, MulticlassPerceptron> outputFormat = null;

        public FeaturesMulticlassPerceptronOutputFormat() {
            outputFormat = new SequenceFileOutputFormat<NullWritable, MulticlassPerceptron>();
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
        public RecordWriter<String, Map<String, Float>> getRecordWriter(
                TaskAttemptContext context) throws IOException, InterruptedException {
            return new FeaturesMulticlassPerceptronRecordWriter(outputFormat.getRecordWriter(context), builderFeatureBit, builderConvertType, modelPath);
        }

    }

    public class FeaturesMulticlassPerceptronRecordWriter extends RecordWriter<String, Map<String, Float>> {

        private RecordWriter writer = null;
        private MulticlassPerceptron classifier = null;

        public FeaturesMulticlassPerceptronRecordWriter(RecordWriter<NullWritable, MulticlassPerceptron> writer, int featureBit, Convert.FeatureConvert convertType, String modelPath) {
            this.writer = writer;

            if (modelPath == null) {
                this.classifier = new MulticlassPerceptron(featureBit, convertType);
            } else {
                try {
                    List<MulticlassPerceptron> classifierList = ModelReader.readModelsFromPath(new Path(modelPath), MulticlassPerceptron.class);
                    this.classifier = new MulticlassPerceptron(classifierList);
                } catch(Exception e) {
                    this.classifier = new MulticlassPerceptron(featureBit, convertType);
                }
            }
        }

        @Override
        public void close(TaskAttemptContext context) throws IOException, InterruptedException {
            writer.write(NullWritable.get(), classifier);
            writer.close(context);
        }

        @Override
        public void write(String k, Map<String, Float> v) throws IOException, InterruptedException {
            classifier.update(k, v);
        }

    }

}

