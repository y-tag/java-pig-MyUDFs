package myudfs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import org.apache.pig.LoadFunc;
import org.apache.pig.PigException;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.backend.hadoop.executionengine.mapReduceLayer.PigSplit;
import org.apache.pig.data.DataByteArray;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

public class SVMLightLoader extends LoadFunc {
    protected RecordReader in = null;
    private ArrayList<Object> mProtoTuple = null;
    private TupleFactory mTupleFactory = TupleFactory.getInstance();

    public SVMLightLoader() {
    }

    @Override
    public Tuple getNext() throws IOException {
        try {
            boolean nextExist = in.nextKeyValue();
            if (! nextExist) {
                return null;
            }
            String key = (String)in.getCurrentKey();
            Map<String, Float> val = (HashMap<String, Float>)in.getCurrentValue();

            if (mProtoTuple == null) {
                mProtoTuple = new ArrayList<Object>();
            }
            mProtoTuple.add(key);
            mProtoTuple.add(val);

            Tuple t =  mTupleFactory.newTupleNoCopy(mProtoTuple);
            mProtoTuple = null;
            return t;
        } catch (InterruptedException e) {
            int errCode = 6018;
            String errMsg = "Error while reading input";
            throw new ExecException(errMsg, errCode,
                    PigException.REMOTE_ENVIRONMENT, e);
        }

    }

    @Override
    public InputFormat getInputFormat() {
        return new SVMLightInputFormat();
    }

    @Override
    public void prepareToRead(RecordReader reader, PigSplit split) {
        in = reader;
    }

    @Override
    public void setLocation(String location, Job job)
            throws IOException {
        FileInputFormat.setInputPaths(job, location);
    }

    public class SVMLightInputFormat extends FileInputFormat<String, HashMap<String, Float>> {

        public SVMLightInputFormat() {
        }

        @SuppressWarnings("unchecked")
        @Override
        public RecordReader createRecordReader(InputSplit split,
                TaskAttemptContext context) throws IOException, InterruptedException {
            return new SVMLightRecordReader((new TextInputFormat()).createRecordReader(split, context));
        } 

        @Override
        protected boolean isSplitable(JobContext context, Path filename) {
            CompressionCodec codec = 
                new CompressionCodecFactory(context.getConfiguration()).getCodec(filename);
            return (codec == null);
        }

    }

    public class SVMLightRecordReader extends RecordReader<String, HashMap<String, Float>> {
        RecordReader<LongWritable,Text> textReader = null;
        String currentKey = null;
        HashMap<String, Float> currentVal = null;

        public SVMLightRecordReader(RecordReader<LongWritable,Text> textReader) {
            this.textReader = textReader;
        }

        @Override
        public void initialize(InputSplit genericSplit, TaskAttemptContext context)
        throws IOException, InterruptedException {
            textReader.initialize(genericSplit, context);
        } 

        @Override
        public void close() throws IOException {
            textReader.close();
        }

        @Override
        public String getCurrentKey() throws IOException, InterruptedException {
            return currentKey;
        }

        @Override
        public HashMap<String, Float> getCurrentValue() throws IOException, InterruptedException {            
            return currentVal;
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return textReader.getProgress();
        }


        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            try {
                boolean nextExists = textReader.nextKeyValue();
                if (! nextExists) {
                    return false;
                }

                Text value = (Text) textReader.getCurrentValue();
                StringTokenizer st = new StringTokenizer(value.toString(), " ");

                if (st.hasMoreTokens()) {
                    currentKey = st.nextToken();
                }

                currentVal = new HashMap<String, Float>();
                while (st.hasMoreTokens()) {
                    String[] tokens = st.nextToken().split(":");
                    if (tokens.length < 2) {
                        continue;
                    }
                    String key = tokens[0];
                    Float  val = Float.parseFloat(tokens[1]);
                    currentVal.put(key, val);
                }

            } catch (InterruptedException e) {
                int errCode = 6018;
                String errMsg = "Error while reading input";
                throw new ExecException(errMsg, errCode,
                        PigException.REMOTE_ENVIRONMENT, e);
            }

            return true;
        }
    }

}
