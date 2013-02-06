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
import myorg.classifier.online.multiclass.MulticlassPA;

public class FeaturesMulticlassPABuilder extends StoreFunc {
    protected RecordWriter writer = null;
    private int builderFeatureBit = 20;
    private Convert.FeatureConvert builderConvertType = Convert.FeatureConvert.HASHING;
    private PACommon.PAType builderPAType = PACommon.PAType.PA;
    private float builderC = 1.0f;
    private String modelPath = null;

    public FeaturesMulticlassPABuilder() {
    }

    public FeaturesMulticlassPABuilder(String modelPath) {
        this.modelPath = modelPath;
    }

    public FeaturesMulticlassPABuilder(String featureBit, String convertType, String paType, String C) {
        this.builderFeatureBit = Integer.parseInt(featureBit);
        this.builderC = Float.parseFloat(C);

        if (convertType.equals("PARSING")) {
            this.builderConvertType = Convert.FeatureConvert.PARSING;
        } else {
            this.builderConvertType = Convert.FeatureConvert.HASHING;
        }

        if (paType.equals("PA1")) {
            this.builderPAType = PACommon.PAType.PA1;
        } else if (paType.equals("PA2")) {
            this.builderPAType = PACommon.PAType.PA2;
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
        return new FeaturesMulticlassPAOutputFormat();
    }

    @Override
    public void prepareToWrite(RecordWriter writer) {
        this.writer = writer;
    }

    @Override
    public void setStoreLocation(String location, Job job) throws IOException {
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(MulticlassPA.class);

        SequenceFileOutputFormat.setOutputPath(job, new Path(location));
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    }

    public class FeaturesMulticlassPAOutputFormat extends FileOutputFormat<String, Map<String, Float>> {
        private SequenceFileOutputFormat<NullWritable, MulticlassPA> outputFormat = null;

        public FeaturesMulticlassPAOutputFormat() {
            outputFormat = new SequenceFileOutputFormat<NullWritable, MulticlassPA>();
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
            return new FeaturesMulticlassPARecordWriter(outputFormat.getRecordWriter(context), builderFeatureBit, builderConvertType, builderPAType, builderC, modelPath);
        }

    }

    public class FeaturesMulticlassPARecordWriter extends RecordWriter<String, Map<String, Float>> {

        private RecordWriter writer = null;
        private MulticlassPA classifier = null;

        public FeaturesMulticlassPARecordWriter(RecordWriter<NullWritable, MulticlassPA> writer, int featureBit, Convert.FeatureConvert convertType, PACommon.PAType paType, float C, String modelPath) {
            this.writer = writer;

            if (modelPath == null) {
                this.classifier = new MulticlassPA(featureBit, convertType, paType, C);
            } else {
                try {
                    List<MulticlassPA> classifierList = ModelReader.readModelsFromPath(new Path(modelPath), MulticlassPA.class);
                    this.classifier = new MulticlassPA(classifierList);
                } catch(Exception e) {
                    this.classifier = new MulticlassPA(featureBit, convertType, paType, C);
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

