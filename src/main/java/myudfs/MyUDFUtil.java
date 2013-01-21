package myudfs;

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;

public class MyUDFUtil {

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
