package myorg.common;

import java.lang.Exception;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;

import org.apache.pig.impl.util.UDFContext;

public class ModelReader {

    public ModelReader() {
    }

    public static <T extends Writable> List<T> readModelsFromPath(Path modelPath, Class<T> targetClass) throws IOException, FileNotFoundException {
        FileSystem fs = FileSystem.get(UDFContext.getUDFContext().getJobConf());
        List<Path> fileList = getFileList(fs, modelPath);

        List<T> classifierList = new ArrayList<T>();
        for (int i = 0; i < fileList.size(); i++) {
            classifierList.add(readModelFromFile(fileList.get(i), targetClass));
        }

        return classifierList;
    }

    public static <T extends Writable> T readModelFromFile(Path modelPath, Class<T> targetClass) throws IOException, FileNotFoundException {
        Configuration conf = UDFContext.getUDFContext().getJobConf();
        FileSystem fs = FileSystem.get(conf);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, modelPath, conf);

        Class<?> keyClass = reader.getKeyClass();
        Class<?> valClass = reader.getValueClass();

        NullWritable key = NullWritable.get();
        T val = null;

        try {
            val = (T)valClass.newInstance();
            reader.next(key, val);
        } catch (Exception e) {
        } finally {
            reader.close();
        }

        return val;
    }

    static public List<Path> getFileList(FileSystem fs, Path path) throws IOException {
        List<Path> fileList = new ArrayList<Path>();
        if (fs.exists(path)) {
            if (fs.getFileStatus(path).isDir()) {
                FileStatus[] status = fs.listStatus(path);
                for (int i = 0; i < status.length; i++) {
                    fileList.addAll(getFileList(fs, status[i].getPath()));
                }
            } else if (path.getName().startsWith("part")) {
                fileList.add(path);
            }
        }

        return fileList;
    }

}
