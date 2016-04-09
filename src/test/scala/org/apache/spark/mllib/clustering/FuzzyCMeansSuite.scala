/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.Utils

/**
 * Created by acflorea on 05/04/16.
 */
class FuzzyCMeansSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("two clusters") {
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),
      Vectors.dense(9.0, 0.0),
      Vectors.dense(9.0, 0.2),
      Vectors.dense(9.2, 0.0)
    )
    val rdd = sc.parallelize(points, 3)

    for (initMode <- Seq(KMeans.RANDOM, KMeans.K_MEANS_PARALLEL)) {

      (1 to 4).map(_ * 2) foreach { fuzzifier =>

        val model = FuzzyCMeans.train(rdd, k = 2, maxIterations = 10, runs = 10, initMode,
          seed = 26031979L, m = fuzzifier)

        assert(model.m === fuzzifier)

        val fuzzyPredicts = model.fuzzyPredict(rdd).collect()

        assert(fuzzyPredicts(0).maxBy(_._2)._1 === fuzzyPredicts(1).maxBy(_._2)._1)
        assert(fuzzyPredicts(0).maxBy(_._2)._1 === fuzzyPredicts(2).maxBy(_._2)._1)
        assert(fuzzyPredicts(3).maxBy(_._2)._1 === fuzzyPredicts(4).maxBy(_._2)._1)
        assert(fuzzyPredicts(3).maxBy(_._2)._1 === fuzzyPredicts(5).maxBy(_._2)._1)
        assert(fuzzyPredicts(0).maxBy(_._2)._1 != fuzzyPredicts(3).maxBy(_._2)._1)

      }
    }
  }

  test("model save/load") {
    val tempDir = Utils.createTempDir()
    val path = tempDir.toURI.toString

    Array(true, false).foreach { case selector =>
      val model = FuzzyCSuite.createModel(10, 3, selector)
      // Save model, load it back, and compare.
      try {
        model.save(sc, path)
        val sameModel = KMeansModel.load(sc, path)
        FuzzyCSuite.checkEqual(model, sameModel)
      } finally {
        Utils.deleteRecursively(tempDir)
      }
    }
  }

}


object FuzzyCSuite extends SparkFunSuite {
  def createModel(dim: Int, k: Int, isSparse: Boolean): KMeansModel = {
    val singlePoint = isSparse match {
      case true =>
        Vectors.sparse(dim, Array.empty[Int], Array.empty[Double])
      case _ =>
        Vectors.dense(Array.fill[Double](dim)(0.0))
    }
    new KMeansModel(Array.fill[Vector](k)(singlePoint), 2)
  }

  def checkEqual(a: KMeansModel, b: KMeansModel): Unit = {
    assert(a.k === b.k)
    assert(a.m === b.m)
    a.clusterCenters.zip(b.clusterCenters).foreach {
      case (ca: SparseVector, cb: SparseVector) =>
        assert(ca === cb)
      case (ca: DenseVector, cb: DenseVector) =>
        assert(ca === cb)
      case _ =>
        throw new AssertionError("checkEqual failed since the two clusters were not identical.\n")
    }
  }
}