# Group15_Minibatch_KMeans_TermProject
Distributed minibatch K-Means for Document clustering
 <br /> <br />

Requirements: <br />
1. Java 8 <br />
2. Ant <br />
3. Hadoop 2.6 <br />
4. HARP <br />
5. Increase Xms if you're faced with memory overflow <br /> <br />


Input:  <br />
Place the input files (unzipped) in the data in the input_files directory within code. <br />
Input file links: <br />
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz <br />
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz <br />
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz <br />
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz <br />


Run: <br />
Command to run the program: <br />
hadoop jar harp3-app-hadoop-2.6.0.jar edu.iu.km.KmeansMapCollective <batch size in %> <number of centroids> <number of iterations> <working directory> <input directory> <output directory> <br />
Eg: hadoop jar harp3-app-hadoop-2.6.0.jar edu.iu.km.KmeansMapCollective 20 10 15 ~/Harp3-Project-master/harp3-app/src ~/Harp3-Project-master/harp3-app/src/input_data/data ~/Harp3-Project-master/harp3-app/src/output_data <br /> <br />


Output: <br />
In the folder output_files (path specified in the command line argument) <br />
