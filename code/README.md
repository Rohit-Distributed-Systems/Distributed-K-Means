# Group15_Minibatch_KMeans_TermProject
Distributed minibatch K-Means for Document clustering


Requirements:
1. Java 8 <br />
2. Ant <br />
3. Hadoop 2.6 <br />
4. HARP <br />
5. Increase Xms if you're faced with memory overflow <br />


Input: 
Place the input files (unzipped) in the data in the input_files directory within code.
Input file links:
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz
http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz


Run:
Command to run the program:
hadoop jar harp3-app-hadoop-2.6.0.jar edu.iu.km.KmeansMapCollective <batch size in %> <number of centroids> <number of iterations> <working directory> <input directory> <output directory>
Eg: hadoop jar harp3-app-hadoop-2.6.0.jar edu.iu.km.KmeansMapCollective 20 10 15 ~/Harp3-Project-master/harp3-app/src ~/Harp3-Project-master/harp3-app/src/input_data/data ~/Harp3-Project-master/harp3-app/src/output_data


Output:
In the folder output_files (path specified in the command line argument)
