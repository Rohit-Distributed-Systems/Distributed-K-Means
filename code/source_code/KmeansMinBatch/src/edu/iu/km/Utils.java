package edu.iu.km;

import java.io.File;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class Utils {

	public static void setupData(FileSystem fs, String localCatDirStr, String localDocDir, Path categoriesDir,
			Path docDir) throws IllegalArgumentException, IOException {

		if (fs.exists(categoriesDir)) {
			fs.delete(categoriesDir, true);
		}
		fs.mkdirs(categoriesDir);

		// Transfer category files
		File dir = new File(localCatDirStr);
		for (File file : dir.listFiles()) {
			fs.copyFromLocalFile(new Path(file.getPath()), categoriesDir);
		}

		if (fs.exists(docDir)) {
			fs.delete(docDir, true);
		}
		fs.mkdirs(docDir);

		dir = new File(localDocDir);
		for (File file : dir.listFiles()) {
			fs.copyFromLocalFile(new Path(file.getPath()), categoriesDir);
		}

	}

	static void generateInitialCentroids(int numCentroids, int vectorSize, Configuration configuration, Path workingDir,
			FileSystem fs) throws IOException {
		StringBuilder centroidsData = new StringBuilder();
		for (int i = 0; i < numCentroids; i++) {
			for (int dimension = 0; dimension < vectorSize; dimension++) {
				centroidsData.append(Math.random() * KMeansConstants.POINT_GENERATION_RANGE);
				centroidsData.append(' ');
			}

			centroidsData.append(System.lineSeparator());
		}
		if (fs.exists(workingDir)) {
			fs.delete(workingDir, true);
		}
		FSDataOutputStream outHDFS = fs.create(new Path(workingDir, "centroidData"), true);
		outHDFS.write(centroidsData.toString().getBytes());
		outHDFS.flush();
		outHDFS.close();
	}

}