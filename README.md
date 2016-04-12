# fuzzyCMeans
## [Apache Spark Fuzzy CMeans Implementation](https://en.wikipedia.org/wiki/Fuzzy_clustering)

The name of the branch matches the Spark version

In order to use it on a Spark cluster the following steps are required:
- checkout the source code
- build it using `sbt package` optionally you can also build the `test` jar by using `sbt test:package`. The `test` jar is required to run the `org.apache.spark.mllib.clustering.FuzzyCMeansClusterSuite` test cases.
- the build process will result in the creation of the following jar files:
 - ```./target/scala-2.10/fuzzycmeans_2.10-<version>.jar```
 - ```./target/scala-2.10/fuzzycmeans_2.10-<version>-tests.jar```
- The built jars can be added to the Spark cluster using `--jars` argument, for example, ```./spark-shell --jars /pathTo/fuzzyCMeans/target/scala-2.10/fuzzycmeans_2.10-<version>.jar, pathTo/fuzzyCMeans/target/scala-2.10/fuzzycmeans_2.10-<version>-tests.jar```

###Usage example in Scala

```java
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),
      Vectors.dense(9.0, 0.0),
      Vectors.dense(9.0, 0.2),
      Vectors.dense(9.2, 0.0)
    )
    val rdd = sc.parallelize(points, 3).cache()

    for (initMode <- Seq(RANDOM, K_MEANS_PARALLEL)) {

      (1 to 4).map(_ * 2) foreach { fuzzifier =>

        val model = FuzzyCMeans.train(rdd, k = 2, maxIterations = 10, runs = 10, initMode,
          seed = 26031979L, m = fuzzifier)

        val fuzzyPredicts = model.fuzzyPredict(rdd).collect()
        
        rdd.collect() zip fuzzyPredicts foreach { fuzzyPredict =>
          println(s" Point ${fuzzyPredict._1}")
          fuzzyPredict._2 foreach{clusterAndProbability =>
            println(s"Probability to belong to cluster ${clusterAndProbability._1} " +
              s"is ${"%.2f".format(clusterAndProbability._2)}")
          }
        }
      }
    }


```
