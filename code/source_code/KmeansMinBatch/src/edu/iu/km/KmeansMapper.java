package edu.iu.km;

import it.unimi.dsi.fastutil.ints.IntIterator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.CollectiveMapper;
import org.apache.hadoop.mapreduce.Mapper;

import edu.iu.fileformat.Document;
import edu.iu.harp.example.DoubleArrPlus;
import edu.iu.harp.example.IntArrPlus;
import edu.iu.harp.partition.Partition;
import edu.iu.harp.partition.PartitionCombiner;
import edu.iu.harp.partition.PartitionStatus;
import edu.iu.harp.partition.Table;
import edu.iu.harp.resource.DoubleArray;
import edu.iu.harp.resource.IntArray;

public class KmeansMapper extends CollectiveMapper<String, String, Object, Object> {
	private double batchSizeInPercent;
	private int numOfCentroids;
	private int numMapTasks;
	private int iterations;
	private String workingDir;
	private String outputDir;
	private String localDirOutput;
	double classificationAccuracy;

	private static final boolean DEBUG = true;

	protected void setup(Mapper<String, String, Object, Object>.Context context)
			throws IOException, InterruptedException

	{
		Configuration conf = context.getConfiguration();
		outputDir = conf.get("outputFilePath");
		localDirOutput = conf.get("localDirOutput");
		iterations = conf.getInt("iterations", 1);
		batchSizeInPercent = conf.getDouble("batchSizeInPercent", 1);
		numOfCentroids = conf.getInt("numOfCentroids", 1);
		workingDir = conf.get("workingDir");
		numMapTasks = conf.getInt("numMapTasks", 1);

	}

	protected void mapCollective(KeyValReader reader, Mapper<String, String, Object, Object>.Context context)
			throws IOException, InterruptedException {
		List<String> docFiles = new LinkedList<>();
		while (reader.nextKeyValue()) {
			String key = reader.getCurrentKey();
			// values are filenames
			String value = reader.getCurrentValue();
			docFiles.add(value);
		}
		Configuration conf = context.getConfiguration();
		try {
			runKmeans(docFiles, conf, context);
		} catch (Exception e) {
			System.err.println("Exception occured while running kmeans");
			e.printStackTrace();
		}
	}

	private void runKmeans(List<String> fileNames, Configuration conf,
			Mapper<String, String, Object, Object>.Context context) throws Exception {
		System.out.println("kmeans Mini Batch starts here");
		final long mapperStart = System.nanoTime();
		Table<Document> centroidTable = new Table<Document>(0, new CentroidCombiner());

		final long documentLoadStart = System.nanoTime();
		final ConcurrentMap<Integer, Document> documents = loadDocuments(fileNames.get(0), conf);
		final long documentLoadEnd = System.nanoTime();
		final long documentLoadTime = TimeUnit.NANOSECONDS.toMillis(documentLoadEnd - documentLoadStart);
		final long documentsCount = documents.size();
		System.out.println(documentsCount + " documents loaded in " + documentLoadTime + "ms");

		int batchSize = (int) ((documentsCount * batchSizeInPercent / 100) / numMapTasks);

		if (isMaster()) {
			populateCentroidTable(centroidTable, numOfCentroids, documents);
		}
		final boolean success = broadcast("KMeans", "centroidBcast", centroidTable, getMasterID(), false);
		if (DEBUG) {
			StringBuilder sb = new StringBuilder("Worker ID: " + getSelfID() + "\n");
			for (Partition<Document> parition : centroidTable.getPartitions()) {
				sb.append("partition:" + parition.id() + "\n");
			}
			System.out.println(sb.toString());
		}
		ConcurrentMap<Integer, Document> centroidIdToCentroidDocs = centroidTable.getPartitions().stream()
				.map(Partition::get).collect(Collectors.toConcurrentMap(k -> k.getId(), v -> v));

		List<Document> originalCentroids = centroidIdToCentroidDocs.values().stream()
				.map(centroid -> new Document(centroid)).collect(Collectors.toList());

		final long initialSSEStart = System.nanoTime();
		final double initialSSE = calculateSSE(true, documents.values(), centroidIdToCentroidDocs.values());
		final long initialSSECalculationTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - initialSSEStart);
		System.out.println("Calculated initial SSE in " + initialSSECalculationTime + "ms");
		if (isMaster()) {
			System.out.println("initial SSE:" + initialSSE);
		}

		// Mini-batch k-means starts here
		final long miniBatchStart = System.nanoTime();
		final Collection<Document> newCentroids = runMiniBatch(centroidTable, documents, centroidIdToCentroidDocs,
				batchSize, iterations);

		final long miniBatchProcessingTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - miniBatchStart);
		System.out.println("Time spent in processing Mini-batch k-means" + miniBatchProcessingTime + "ms");

		if (DEBUG) {
			System.out.println("New centroids:");
			System.out.println(newCentroids);
		}

		final long finalSSEStart = System.nanoTime();
		final double finalSSE = calculateSSE(false, documents.values(), newCentroids);
		final long finalSSECalculationTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - finalSSEStart);
		System.out.println("Calculated Final SSE in " + finalSSECalculationTime + "ms");
		System.out.println("final SSE:" + finalSSE);

		System.out.println("Starting classification accuracy test");
		final long classificationAccuracyStart = System.nanoTime();
		final List<Integer> centroidRepresentatorDocumentID = findCentroidRepresentatorDocumentID(newCentroids,
				documents.values());

		final ConcurrentMap<Integer, Integer> docIdTocentroidRepresentatorId = associateDocumentWithNearestCentroidRepresentation(
				documents, centroidRepresentatorDocumentID);

		documents.clear();

		final long categoryFileLoadStart = System.nanoTime();
		ConcurrentMap<Integer, Set<String>> categories = loadCategories(getCategoriesFilePath(workingDir), conf);
		final long categoryFileLoadTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - categoryFileLoadStart);
		System.out.println("Time spent in loading categories file:" + categoryFileLoadTime + "ms");
		final long docsCategorizedCorrectly = countDocumentsCategorizedCorrectly(categories,
				docIdTocentroidRepresentatorId);
		getAggregateAccuracyCounts((int) docsCategorizedCorrectly, docIdTocentroidRepresentatorId.size());
		final long classificationAccuracyCalculationTime = TimeUnit.NANOSECONDS
				.toMillis(System.nanoTime() - classificationAccuracyStart);
		System.out.println("Finished classification accuracy test in:" + classificationAccuracyCalculationTime + "ms");
		context.progress();
		final long timeSpentInMapper = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - mapperStart);
		System.out.println("Total time spent in mapper excluding output write:" + timeSpentInMapper + "ms");
		if (isMaster()) {
			writeResults(classificationAccuracy, conf, outputDir, localDirOutput, finalSSE, initialSSE, newCentroids,
					originalCentroids, iterations, batchSize, batchSizeInPercent, documentsCount, numOfCentroids,
					numMapTasks, documentLoadTime, initialSSECalculationTime, miniBatchProcessingTime,
					finalSSECalculationTime, classificationAccuracyCalculationTime, timeSpentInMapper,
					categoryFileLoadTime);
		}
		centroidTable.release();
	}

	private ConcurrentMap<Integer, Set<String>> loadCategories(Path categoryPath, Configuration conf)
			throws IOException {
		final Function<Matcher, String> baseCategoryExtractor = m -> m.group("baseCategory");
		final Function<Matcher, Integer> docIDExtractor = m -> Integer.parseInt(m.group("docid"));
		Pattern pattern = Pattern.compile("^(?<category>(?<baseCategory>\\w{1})(\\w+))\\s(?<docid>\\d+)\\s1$");
		ConcurrentMap<Integer, Set<String>> categories = null;
		try (FileSystem fs = FileSystem.get(conf);
				FSDataInputStream in = fs.open(categoryPath);
				BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {
			categories = reader.lines().map(s -> {
				Matcher matcher = pattern.matcher(s);
				matcher.find();
				return matcher;
			}).collect(Collectors.groupingByConcurrent(docIDExtractor,
					Collectors.mapping(baseCategoryExtractor, Collectors.toSet())));
		}
		return categories;
	}

	private ConcurrentMap<Integer, Document> loadDocuments(String path, Configuration conf) throws IOException {
		Path docPath = new Path(path);
		ConcurrentMap<Integer, Document> documents = null;
		try (FileSystem fs = FileSystem.get(conf);
				FSDataInputStream in = fs.open(docPath);
				BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {
			documents = reader.lines().map(line -> line.trim().split("\\s+"))
					.map(splitLine -> new Document(Integer.parseInt(splitLine[0]),
							Arrays.stream(splitLine).skip(1).map(splitToken -> splitToken.trim().split(":"))
									.collect(Collectors.toConcurrentMap(d -> Integer.parseInt(d[0]),
											d -> Double.parseDouble(d[1])))))
					.collect(Collectors.toConcurrentMap(k -> k.getId(), v -> v));
		}

		return documents;
	}

	private void writeResults(double classificationAccuracy, Configuration conf, String hdfsOutputDir,
			String localDirOutput, double finalSSE, double initialSSE, Collection<Document> newCentroids,
			List<Document> originalCentroids, int iterations, int batchSize, double batchSizeInPercent,
			long documentsCount, int numOfCentroids, int numMapTasks, final long documentLoadTime,
			final long initialSSECalculationTime, final long miniBatchProcessingTime,
			final long finalSSECalculationTime, final long classificationAccuracyCalculationTime,
			final long timeSpentInMapper, final long categoryFileLoadTime) throws IOException {
		StringBuilder output = new StringBuilder();
		String newLine = System.getProperty("line.separator");
		output.append("Please See Results Below : ").append(newLine).append("Initial SSE:").append(initialSSE)
				.append(newLine).append("Final SSE:").append(finalSSE).append(newLine)
				.append("Classification Accuracy:").append(classificationAccuracy).append(newLine)
				.append("Initial Centroids:").append(originalCentroids).append(newLine).append("Final Centroids:")
				.append(newCentroids).append(newLine);
		String content = output.toString();
		writeOutput(content, hdfsOutputDir, conf);
		writeOutputToLocalDir(content, localDirOutput);
	}

	private void writeOutput(String content, String hadoopOutputDir, Configuration conf) throws IOException {
		FileSystem fs = FileSystem.get(conf);
		System.out.println("Writing O/P");
		Path output = new Path(hadoopOutputDir, "result");
		FSDataOutputStream outHDFS = fs.create(output, true);
		outHDFS.write(content.toString().getBytes());
		outHDFS.flush();
		outHDFS.close();
		System.out.println("Output Written!");
	}

	private void writeOutputToLocalDir(String content, String localDirOutput) throws IOException {
		System.out.println("Writing output file");
		java.nio.file.Path outputDir = Paths.get(localDirOutput);
		if (Files.notExists(outputDir)) {
			Files.createDirectory(outputDir);
		}
		LocalDateTime now = LocalDateTime.now();
		String outputFileName = "output-"
				+ now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH));
		java.nio.file.Path outputFile = Paths.get(localDirOutput, outputFileName);
		Files.write(outputFile, content.getBytes());
		System.out.println("Completed writing output file");
	}

	private long countDocumentsCategorizedCorrectly(final ConcurrentMap<Integer, Set<String>> categories,
			final ConcurrentMap<Integer, Integer> docIdTocentroidRepresentatorId) {
		System.out.println("Starting to count the documents categorized correctly on the mapper");
		final long categorizedCorrectly = docIdTocentroidRepresentatorId.entrySet().stream().filter(
				e -> categories.get(e.getKey()).stream().anyMatch(cat -> categories.get(e.getValue()).contains(cat)))
				.count();
		System.out.println("Finished counting the documents categorized correctly on the mapper");
		System.out.println("Docs categorized correctly on the mapper:" + categorizedCorrectly + " out of:"
				+ docIdTocentroidRepresentatorId.size());
		return categorizedCorrectly;

	}

	private Path getCategoriesFilePath(String workingDir) {
		final Path dataDir = new Path(workingDir, "data");
		final Path categoryDir = new Path(dataDir, "categories");
		return new Path(categoryDir, KMeansConstants.CATEGORY_FILE_NAME);
	}

	private void getAggregateAccuracyCounts(int docsCategorizedCorrectly, int docsCount) {
		final Table<IntArray> categoryAccuracy = new Table<IntArray>(1, new IntArrPlus());
		categoryAccuracy.addPartition(getCorrectlyCategorizedPartition(docsCategorizedCorrectly));
		categoryAccuracy.addPartition(getTotalDocsPartition(docsCount));
		reduce("KMeans", "reduce-categorization-accuracy", categoryAccuracy, 0);
		System.out.println("Aggregation complete at Master");
		if (isMaster()) {
			double totalDocsCategorizedCorrectly = categoryAccuracy.getPartition(0).get().get()[0];
			double totalDocs = categoryAccuracy.getPartition(1).get().get()[0];
			classificationAccuracy = (totalDocsCategorizedCorrectly / totalDocs);
			System.out.println("Category Accuracy:" + classificationAccuracy);
		}
		categoryAccuracy.release();
	}

	public ConcurrentMap<Integer, Integer> associateDocumentWithNearestCentroidRepresentation(
			ConcurrentMap<Integer, Document> documents, List<Integer> centroidRepresentatorDocumentID) {
		ConcurrentMap<Integer, Integer> docIdTocentroidRepresentatorId = documents.values().stream()
				.collect(Collectors.toConcurrentMap(k -> k.getId(), v -> {
					CentroidComparisonWrapper min = new CentroidComparisonWrapper();
					centroidRepresentatorDocumentID.stream().map(id -> documents.get(id)).forEach(other -> {
						min.assignMin(1 - v.calculateCosineSimilarity(other), other.getId());
					});
					return min.getClosestCentroidID();
				}));
		return docIdTocentroidRepresentatorId;
	}

	public List<Integer> findCentroidRepresentatorDocumentID(Collection<Document> newCentroids,
			Collection<Document> documents) {
		System.out.println("Finding centroids");
		List<Integer> representativeDocumentID = newCentroids.stream().map(document -> {
			CentroidComparisonWrapper min = new CentroidComparisonWrapper();
			documents.stream().forEach(other -> {
				min.assignMin(1 - document.calculateCosineSimilarity(other), other.getId());
			});
			return min.getClosestCentroidID();
		}).collect(Collectors.toList());
		System.out.println("Finished finding centroid representators");
		if (DEBUG) {
			System.out.println("Documents representing centroids");
			System.out.println(representativeDocumentID);
		}
		return representativeDocumentID;
	}

	public double calculateSSE(boolean initialSSE, Collection<Document> documents, Collection<Document> centroids) {
		final ConcurrentMap<Integer, CentroidSimilarityAvg> centroidIDToSimilarityAvg = new ConcurrentHashMap<>();
		// initialize the map
		centroids.stream().map(centroid -> centroid.getId())
				.forEach(k -> centroidIDToSimilarityAvg.put(k, new CentroidSimilarityAvg()));
		// holds the information about the association with closest centroid
		List<DocInfo> docInfo = documents.stream().map(doc -> {
			CentroidComparisonWrapper min = new CentroidComparisonWrapper();
			centroids.stream().forEach(other -> {
				min.assignMin(1 - doc.calculateCosineSimilarity(other), other.getId());
			});
			centroidIDToSimilarityAvg.get(min.getClosestCentroidID()).add(min.getClosestCentroidSimilarity());
			return new DocInfo(min.getClosestCentroidID(), doc.getId(), min.getClosestCentroidSimilarity());
		}).collect(Collectors.toList());
		ConcurrentMap<Integer, Double> centroidIdToSSEAvg = synchrnonizeAndUpdateCentroidSSEAvg(initialSSE,
				centroidIDToSimilarityAvg);
		ConcurrentMap<Integer, DoubleAdder> centroidSSE = new ConcurrentHashMap<>();
		// initialize the map
		centroids.stream().map(centroid -> centroid.getId()).forEach(k -> centroidSSE.put(k, new DoubleAdder()));
		docInfo.stream().forEach(documentInfo -> {
			centroidSSE.get(documentInfo.getClosestCentroidID())
					.add(Math.pow(documentInfo.getDistanceFromClosestCentroid()
							- centroidIdToSSEAvg.get(documentInfo.getClosestCentroidID()), 2));
		});
		double nodeLevelSSE = centroidSSE.values().stream().mapToDouble(adder -> adder.doubleValue()).sum();
		double totalSSE = addSSEFromAlltheMappers(nodeLevelSSE);
		return totalSSE;
	}

	private double addSSEFromAlltheMappers(double sseVal) {
		Table<DoubleArray> summer = new Table<DoubleArray>(3, new DoubleArrPlus());
		DoubleArray sseValPartition = DoubleArray.create(1, false);
		double[] backingArray = sseValPartition.get();
		backingArray[0] = sseVal;
		summer.addPartition(new Partition<DoubleArray>(0, sseValPartition));
		allreduce("KMeans", "reduce-addSSEFromAlltheMappers", summer);
		double totalSSE = summer.getPartition(0).get().get()[0];
		summer.release();
		return totalSSE;
	}

	private ConcurrentMap<Integer, Double> synchrnonizeAndUpdateCentroidSSEAvg(boolean intial,
			ConcurrentMap<Integer, CentroidSimilarityAvg> centroidIDToSimilarityAvg) {
		Table<DoubleArray> categoryAccuracySynchronizer = new Table<DoubleArray>(2, new DoubleArrPlus());
		List<Partition<DoubleArray>> partitions = centroidIDToSimilarityAvg.entrySet().stream().map(entry -> {
			return getSSEAvgPartition(entry.getKey(), entry.getValue().getSum(), entry.getValue().getCount());
		}).collect(Collectors.toList());
		partitions.stream().forEach(partition -> categoryAccuracySynchronizer.addPartition(partition));
		allreduce("KMeans", "reduce-synchrnonizeCentroidSSEAvg" + intial, categoryAccuracySynchronizer);
		ConcurrentMap<Integer, Double> centroidAvgSSE = categoryAccuracySynchronizer.getPartitions().stream()
				.collect(Collectors.toConcurrentMap(partition -> partition.id(), partition -> {
					double updatedSum = partition.get().get()[0];
					double updatedCount = partition.get().get()[1];
					return updatedSum / updatedCount;
				}));
		categoryAccuracySynchronizer.release();
		return centroidAvgSSE;
	}

	private Partition<DoubleArray> getSSEAvgPartition(int centroidID, double sum, int count) {
		DoubleArray sseAvgPartition = DoubleArray.create(2, false);
		double[] backingArray = sseAvgPartition.get();
		backingArray[0] = sum;
		backingArray[1] = count;
		return new Partition<DoubleArray>(centroidID, sseAvgPartition);
	}

	private Partition<IntArray> getCorrectlyCategorizedPartition(int docsCategorizedCorrectly) {
		IntArray categoryAccuracyArray = IntArray.create(1, false);
		categoryAccuracyArray.get()[0] = docsCategorizedCorrectly;
		return new Partition<IntArray>(0, categoryAccuracyArray);
	}

	private Partition<IntArray> getTotalDocsPartition(int totalDocs) {
		IntArray totalDocsArr = IntArray.create(1, false);
		totalDocsArr.get()[0] = totalDocs;
		return new Partition<IntArray>(1, totalDocsArr);
	}

	private Collection<Document> runMiniBatch(Table<Document> centroidTable, ConcurrentMap<Integer, Document> documents,
			ConcurrentMap<Integer, Document> centroids, int batchSize, int iterations) {
		System.out.println("Mini-batch k-means started");
		IntStream.range(0, iterations).peek(i -> {

		}).forEach(i -> {

			List<Document> batch = getBatch(documents, batchSize);

			ConcurrentMap<Integer, Integer> centroidCache = buildCentroidCache(batch, centroids.values());

			List<Integer> removePartition = new LinkedList<>();
			IntIterator partitionIDIterator = centroidTable.getPartitionIDs().iterator();
			while (partitionIDIterator.hasNext()) {
				Integer parttionID = partitionIDIterator.next();
				if (!((parttionID % getNumWorkers()) == getSelfID())) {
					removePartition.add(parttionID);
				}
			}
			removePartition.stream().forEach(parttionID -> {
				centroids.remove(centroidTable.getPartition(parttionID).get().getId());
				centroidTable.removePartition(parttionID);
			});

			IntStream.range(0, getNumWorkers()).peek(it -> {

			}).forEach(j -> {
				batch.stream().forEach(document -> {
					Document centroid = centroids.get(centroidCache.get(document.getId()));
					if (centroid == null) {
						return;
					} else {
						int count = centroid.getPointsAssociatedCount().incrementAndGet();
						double n = (double) 1 / count;
						BiFunction<Integer, Double, Double> reMappingFunction = (featureID,
								featureVal) -> featureVal * (1 - n) + n * document.getFeatureValue(featureID);
						centroid.updateFeatureValues(reMappingFunction);
					}
				});

				rotate("KMeans", "roatateCentroids-" + "i-" + j, centroidTable, null);

				centroids.clear();
				centroidTable.getPartitions().stream().map(Partition::get)
						.forEach(centroid -> centroids.put(centroid.getId(), centroid));
			});

			allgather("KMMapper", "allGather-" + i, centroidTable);
			centroidTable.getPartitions().stream().map(Partition::get).forEach(centroid -> {
				if (!centroids.containsKey(centroid.getId())) {
					centroids.put(centroid.getId(), centroid);
				}
			});

		});
		System.out.println("Done....");
		return centroids.values();
	}

	private List<Document> getBatch(Map<Integer, Document> documents, int batchSize) {
		List<Document> docList = new ArrayList<>(documents.values());
		return IntStream.range(0, batchSize).mapToObj(in -> getRandomDocument(documents)).collect(Collectors.toList());
	}

	private ConcurrentMap<Integer, Integer> buildCentroidCache(List<Document> batch, Collection<Document> centroids) {
		return batch.stream().distinct().collect(Collectors.toConcurrentMap(document -> document.getId(), document -> {
			CentroidComparisonWrapper min = new CentroidComparisonWrapper();
			centroids.stream().forEach(d -> {
				min.assignMin(1 - d.calculateCosineSimilarity(document), d.getId());
			});
			return min.getClosestCentroidID();
		}));
	}

	private Document getRandomDocument(Map<Integer, Document> documents) {
		Document random = null;
		try {
			Field table = ConcurrentHashMap.class.getDeclaredField("table");
			table.setAccessible(true);
			Entry<Integer, Document>[] entries = (Entry<Integer, Document>[]) table.get(documents);
			while (random == null) {
				int index = (int) Math.floor(documents.size() * Math.random());
				while (index >= entries.length) {
					index = (int) Math.floor(documents.size() * Math.random());
				}
				if (entries[index] != null) {
					random = (Document) entries[index].getValue();
				}
			}
		} catch (NoSuchFieldException | IllegalAccessException ex) {
			throw new RuntimeException(ex);
		}
		return new Document(random);
	}

	private void populateCentroidTable(Table<Document> cenTable, int numOfCentroids,
			ConcurrentMap<Integer, Document> documents) {
		System.out.println("Centroid creation started");
		ConcurrentMap<Integer, Document> centroids = getRandomCentroids(numOfCentroids, documents);
		int[] i = new int[1];
		centroids.values().stream().forEach(doc -> {
			cenTable.addPartition(new Partition<Document>(i[0]++, doc));
		});
		System.out.println("centroid creation done!");
	}

	private ConcurrentMap<Integer, Document> getRandomCentroids(int centroids, Map<Integer, Document> documents) {
		return IntStream.range(0, centroids).mapToObj(in -> getRandomDocument(documents))
				.collect(Collectors.toConcurrentMap(k -> k.getId(), v -> v));
	}

	public static class CentroidSimilarityAvg {
		private DoubleAdder sum;
		private LongAdder count;

		public CentroidSimilarityAvg() {
			sum = new DoubleAdder();
			count = new LongAdder();
		}

		public void add(double val) {
			sum.add(val);
			count.increment();
		}

		public double getSum() {
			return sum.doubleValue();
		}

		public int getCount() {
			return count.intValue();
		}

	}

	public static class DocInfo {
		private int closestCentroidID;
		private int docID;
		private double distanceFromClosestCentroid;

		public DocInfo(int closestCentroidID, int docID, double distanceFromClosestCentroid) {
			super();
			this.closestCentroidID = closestCentroidID;
			this.docID = docID;
			this.distanceFromClosestCentroid = distanceFromClosestCentroid;
		}

		public int getClosestCentroidID() {
			return closestCentroidID;
		}

		public void setClosestCentroidID(int closestCentroidID) {
			this.closestCentroidID = closestCentroidID;
		}

		public int getDocID() {
			return docID;
		}

		public void setDocID(int docID) {
			this.docID = docID;
		}

		public double getDistanceFromClosestCentroid() {
			return distanceFromClosestCentroid;
		}

		public void setDistanceFromClosestCentroid(double distanceFromClosestCentroid) {
			this.distanceFromClosestCentroid = distanceFromClosestCentroid;
		}

	}

	public static class CentroidComparisonWrapper {
		private volatile double distance;
		private volatile int documentID;

		public CentroidComparisonWrapper() {
			distance = Double.MAX_VALUE;
			documentID = Integer.MIN_VALUE;
		}

		public synchronized void assignMin(double otherDistance, int otherID) {
			if (otherDistance < distance) {
				distance = otherDistance;
				documentID = otherID;
			}
		}

		public int getClosestCentroidID() {
			return documentID;
		}

		public double getClosestCentroidSimilarity() {
			return distance;
		}

	}
}

class CentroidCombiner extends PartitionCombiner<Document> {
	public PartitionStatus combine(Document curPar, Document newPar) {
		return PartitionStatus.COMBINED;
	}
}