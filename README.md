# fuzzyCMeans
## [Apache Spark Fuzzy CMeans Implementation](https://en.wikipedia.org/wiki/Fuzzy_clustering)

The name of the branch matches the Spark version

In order to use it on a Spark cluster the following steps are required:
- checkout the source code
- build it using `sbt package` optionally you can also build the `test` jar by using `sbt test:package`. The `test` jar is required to run the `org.apache.spark.mllib.clustering.FuzzyCMeansClusterSuite` test cases.
- the build process will result in the creation of the following jar files:
 - ```./target/scala-2.10/fuzzycmeans_2.10-1.5.2.jar```
 - ```./target/scala-2.10/fuzzycmeans_2.10-1.5.2-tests.jar```
