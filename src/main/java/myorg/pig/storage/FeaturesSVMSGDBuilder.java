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
import myorg.common.LossFunction;
import myorg.common.ModelReader;
import myorg.classifier.online.binary.SVMSGD;

public class FeaturesSVMSGDBuilder extends StoreFunc {
    protected RecordWriter writer = null;
    private int builderFeatureBit = 20;
    private Convert.FeatureConvert builderConvertType = Convert.FeatureConvert.HASHING;
    private LossFunction.LossType builderLossType = LossFunction.LossType.LOG;
    private float builderC  = 1.0f;
    private long  builderT0 = 1;
    private String modelPath = null;

    public FeaturesSVMSGDBuilder() {
    }

    public FeaturesSVMSGDBuilder(String modelPath) {
        this.modelPath = modelPath;
    }

    public FeaturesSVMSGDBuilder(String featureBit, String convertType, String lossType, String C, String t0) {
        this.builderFeatureBit = Integer.parseInt(featureBit);
        this.builderC  = Float.parseFloat(C);
        this.builderT0 = Long.parseLong(t0);

        if (convertType.equals("PARSING")) {
            this.builderConvertType = Convert.FeatureConvert.PARSING;
        } else {
            this.builderConvertType = Convert.FeatureConvert.HASHING;
        }

        if (lossType.equals("HINGE")) {
            this.builderLossType = LossFunction.LossType.HINGE;
        } else if (lossType.equals("SQUAREDHINGE")) {
            this.builderLossType = LossFunction.LossType.SQUAREDHINGE;
        } else {
            this.builderLossType = LossFunction.LossType.LOG;
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
        return new FeaturesSVMSGDOutputFormat();
    }

    @Override
    public void prepareToWrite(RecordWriter writer) {
        this.writer = writer;
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(SVMSGD.class);

        SequenceFileOutputFormat.setOutputPath(job, new Path(location));
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    }

    public class FeaturesSVMSGDOutputFormat extends FileOutputFormat<Integer, Map<String, Float>> {
        private SequenceFileOutputFormat<NullWritable, SVMSGD> outputFormat = null;

        public FeaturesSVMSGDOutputFormat() {
            outputFormat = new SequenceFileOutputFormat<NullWritable, SVMSGD>();
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
            return new FeaturesSVMSGDRecordWriter(outputFormat.getRecordWriter(context), builderFeatureBit, builderConvertType, builderLossType, builderC, builderT0, modelPath);
        }

    }

    public class FeaturesSVMSGDRecordWriter extends RecordWriter<Integer, Map<String, Float>> {

        private RecordWriter writer = null;
        private SVMSGD classifier       = null;

        public FeaturesSVMSGDRecordWriter(RecordWriter<NullWritable, SVMSGD> writer, int featureBit, Convert.FeatureConvert convertType, LossFunction.LossType lossType, float C, long t0, String modelPath) {
            this.writer     = writer;

            if (modelPath == null) {
                this.classifier = new SVMSGD(featureBit, convertType, lossType, C, t0);
            } else {
                try {
                    List<SVMSGD> classifierList = ModelReader.readModelsFromPath(new Path(modelPath), SVMSGD.class);
                    this.classifier = new SVMSGD(classifierList);
                } catch(Exception e) {
                    this.classifier = new SVMSGD(featureBit, convertType, lossType, C, t0);
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

