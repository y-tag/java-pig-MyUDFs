package myudfs;

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

public class FeaturesPABuilder extends StoreFunc {
    protected RecordWriter writer = null;
    private int builderFeatureBit = 20;
    private BinaryOnlineClassifier.FeatureConvert builderConvertType =
        BinaryOnlineClassifier.FeatureConvert.HASHING;
    private PA.PAType builderPAType = PA.PAType.PA;
    private float builderC = 1.0f;
    private String modelPath = null;

    public FeaturesPABuilder() {
    }

    public FeaturesPABuilder(String modelPath) {
        this.modelPath = modelPath;
    }

    public FeaturesPABuilder(String featureBit, String convertType, String paType, String C) {
        this.builderFeatureBit = Integer.parseInt(featureBit);
        this.builderC = Float.parseFloat(C);

        if (convertType.equals("PARSING")) {
            this.builderConvertType = BinaryOnlineClassifier.FeatureConvert.PARSING;
        } else {
            this.builderConvertType = BinaryOnlineClassifier.FeatureConvert.HASHING;
        }

        if (paType.equals("PA1")) {
            this.builderPAType = PA.PAType.PA1;
        } else if (paType.equals("PA2")) {
            this.builderPAType = PA.PAType.PA2;
        } else {
            this.builderPAType = PA.PAType.PA;
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
        Integer key = (Integer)field;

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
        return new FeaturesPAClassifierOutputFormat();
    }

    @Override
    public void prepareToWrite(RecordWriter writer) {
        this.writer = writer;
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(PA.class);

        SequenceFileOutputFormat.setOutputPath(job, new Path(location));
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    }

    public class FeaturesPAClassifierOutputFormat extends FileOutputFormat<Integer, Map<String, Float>> {
        private SequenceFileOutputFormat<NullWritable, PA> outputFormat = null;

        public FeaturesPAClassifierOutputFormat() {
            outputFormat = new SequenceFileOutputFormat<NullWritable, PA>();
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
        public RecordWriter<Integer, Map<String, Float>> getRecordWriter(
                TaskAttemptContext context) throws IOException, InterruptedException {
            return new FeaturesPAClassifierRecordWriter(outputFormat.getRecordWriter(context), builderFeatureBit, builderConvertType, builderPAType, builderC, modelPath);
        }

    }

    public class FeaturesPAClassifierRecordWriter extends RecordWriter<Integer, Map<String, Float>> {

        private RecordWriter writer = null;
        private PA classifier       = null;

        public FeaturesPAClassifierRecordWriter(RecordWriter<NullWritable, PA> writer, int featureBit, BinaryOnlineClassifier.FeatureConvert convertType, PA.PAType paType, float C, String modelPath) {
            this.writer     = writer;

            if (modelPath == null) {
                this.classifier = new PA(featureBit, convertType, paType, C);
            } else {
                try {
                    List<PA> classifierList = ModelReader.readModelsFromPath(new Path(modelPath), PA.class);
                    this.classifier = new PA(classifierList);
                } catch(Exception e) {
                    this.classifier = new PA(featureBit, convertType, paType, C);
                }
            }
        }

        @Override
        public void close(TaskAttemptContext context) throws IOException, InterruptedException {
            writer.write(NullWritable.get(), classifier);
            writer.close(context);
        }

        @Override
        public void write(Integer k, Map<String, Float> v) throws IOException, InterruptedException {
            classifier.update(k, v);
        }

    }

}

