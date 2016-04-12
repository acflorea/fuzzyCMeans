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

package org.apache.spark.ml.clustering

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.clustering.{FuzzyCMeans => MLlibFuzzyCMeans}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

private case class TestRow(features: Vector)

object FuzzyCMeansSuite {
  def generateFuzzyCMeansData(sql: SQLContext, rows: Int, dim: Int, k: Int): DataFrame = {
    val sc = sql.sparkContext
    val rdd = sc.parallelize(1 to rows).map(i => Vectors.dense(Array.fill(dim)((i % k).toDouble)))
      .map(v => new TestRow(v))
    sql.createDataFrame(rdd)
  }

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allParamSettings: Map[String, Any] = Map(
    "predictionCol" -> "myPrediction",
    "k" -> 3,
    "m" -> 1.5,
    "maxIter" -> 2,
    "tol" -> 0.01
  )
}

class FuzzyCMeansSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  final val k = 5
  @transient var dataset: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset = FuzzyCMeansSuite.generateFuzzyCMeansData(sqlContext, 50, 3, k)
  }

  test("default parameters") {
    val fuzzyCmeans = new FuzzyCMeans()

    assert(fuzzyCmeans.getK === 2)
    assert(fuzzyCmeans.getM === 2.0)
    assert(fuzzyCmeans.getFeaturesCol === "features")
    assert(fuzzyCmeans.getPredictionCol === "prediction")
    assert(fuzzyCmeans.getMaxIter === 20)
    assert(fuzzyCmeans.getInitMode === MLlibFuzzyCMeans.K_MEANS_PARALLEL)
    assert(fuzzyCmeans.getInitSteps === 5)
    assert(fuzzyCmeans.getTol === 1e-4)
  }

  test("set parameters") {
    val fuzzyCmeans = new FuzzyCMeans()
      .setK(9)
      .setM(1.5)
      .setFeaturesCol("test_feature")
      .setPredictionCol("test_prediction")
      .setMaxIter(33)
      .setInitMode(MLlibFuzzyCMeans.RANDOM)
      .setInitSteps(3)
      .setSeed(123)
      .setTol(1e-3)

    assert(fuzzyCmeans.getK === 9)
    assert(fuzzyCmeans.getM === 1.5)
    assert(fuzzyCmeans.getFeaturesCol === "test_feature")
    assert(fuzzyCmeans.getPredictionCol === "test_prediction")
    assert(fuzzyCmeans.getMaxIter === 33)
    assert(fuzzyCmeans.getInitMode === MLlibFuzzyCMeans.RANDOM)
    assert(fuzzyCmeans.getInitSteps === 3)
    assert(fuzzyCmeans.getSeed === 123)
    assert(fuzzyCmeans.getTol === 1e-3)
  }

  test("parameters validation") {
    intercept[IllegalArgumentException] {
      new FuzzyCMeans().setK(1)
    }
    intercept[IllegalArgumentException] {
      new FuzzyCMeans().setInitMode("no_such_a_mode")
    }
    intercept[IllegalArgumentException] {
      new FuzzyCMeans().setInitSteps(0)
    }
    intercept[IllegalArgumentException] {
      new FuzzyCMeans().setM(0.9)
    }
  }

  test("fit & transform") {
    val predictionColName = "fuzzymeans_prediction"
    val cMeans = new FuzzyCMeans().setK(k).setPredictionCol(predictionColName).setSeed(1)
    val cMeansModel = cMeans.fit(dataset)
    assert(cMeansModel.clusterCenters.length === k)

    val transformed = cMeansModel.transform(dataset)
    val expectedColumns = Array("features", predictionColName)
    expectedColumns.foreach { column =>
      assert(transformed.columns.contains(column))
    }
    val clusters = transformed.select(predictionColName).map(_.getSeq[Row](0)).distinct().collect().toSet
    assert(clusters.size === k)

    val hardPredictions = clusters.map(cluster => cluster.maxBy(_.getDouble(1))).map(_.getInt(0))

    assert(hardPredictions === Set(0, 1, 2, 3, 4))
  }

  test("read/write") {
    def checkModelData(model: FuzzyCMeansModel, model2: FuzzyCMeansModel): Unit = {
      assert(model.clusterCenters === model2.clusterCenters)
      assert(model.m === model2.m)
    }
    val fuzzyCmeans = new FuzzyCMeans()
    testEstimatorAndModelReadWrite(fuzzyCmeans, dataset, FuzzyCMeansSuite.allParamSettings, checkModelData)
  }
}
