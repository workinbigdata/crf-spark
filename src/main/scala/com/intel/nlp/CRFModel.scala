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
package com.intel.nlp


import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.hadoop.fs.{Path => PathFS}
import org.apache.spark.sql.catalyst.{ScalaReflection}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{ArrayType, StructType, StructField}
import org.json4s.{DefaultFormats}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


class CRFModel (val model :(Array[String], Array[(String, Int)], Array[Double]))
  extends Saveable with Serializable {
  def save(sc: SparkContext, path: String): Unit = {
    CRFModel.SaveLoadV1_0.save(sc, path, model)
  }

  override protected def formatVersion = "1.0"

  /**
    * Verify CRF model
    *
    * @param tests  Source files to be verified
    * @param costFactor cost factor
    * @return Source files with the predictive labels
    */
  def predict(tests: RDD[Array[String]],
              costFactor: Double): RDD[Array[String]] = {
    val bcModel = tests.context.broadcast(this)
    tests.map { test =>
      val model = bcModel.value
      testCRF(test, model, costFactor)
    }
  }

  def predict(tests: RDD[Array[String]]): RDD[Array[String]] = {
    predict(tests, 1.0)
  }

  /**
    * Internal method to test the CRF model
    *
    * @param test the same source in the CRFLearn
    * @param model the output from CRFLearn
    * @return the source with predictive labels
    */
  def testCRF(test: Array[String],
              model: CRFModel,
              costFactor: Double) = {
    val deFeatureIdx = new DecoderFeatureIndex()
    deFeatureIdx.openFromArray(model)
    val tagger = new Tagger(deFeatureIdx.labels.size, 1)
    tagger.setCostFactor(costFactor)
    tagger.read(test, deFeatureIdx)
    deFeatureIdx.buildFeatures(tagger)
    tagger.cloneFeature(deFeatureIdx)
    tagger.parse(deFeatureIdx.alpha)
    tagger.createOutput(deFeatureIdx.labels)
  }
}


object CRFModel extends Loader[CRFModel] {
  override def load(sc: SparkContext, path: String): CRFModel = {
    CRFModel.SaveLoadV1_0.load(sc, path)
  }

  def dicPath(path: String): String = new PathFS(path, "dicdata").toUri.toString
  def metadataPath(path: String): String = new PathFS(path, "metadata").toUri.toString
  def alphaPath(path: String): String = new PathFS(path, "alphadata").toUri.toString
  def headPath(path: String): String = new PathFS(path, "headdata").toUri.toString

  private object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    val thisClassName = "com.intel.nlp.CRFModel"

    def save(sc: SparkContext, path: String, model: (Array[String], Array[(String, Int)], Array[Double])) : Unit = {
      val metadata: String = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~
          ("featureSize" -> model._3.size)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))
      sc.parallelize(model._1.toSeq, 1).saveAsTextFile(headPath(path))
      sc.parallelize(model._2.map(x => Seq(x._1, x._2.toString).mkString("\t")).toSeq, 1).saveAsTextFile(dicPath(path))
      sc.parallelize(model._3.toSeq, 1).saveAsTextFile(alphaPath(path))


    }

    def load(sc: SparkContext, path: String): CRFModel = {
      val head: Array[String] = sc.textFile(headPath(path)).collect()
      val alpha: Array[Double] = sc.textFile(alphaPath(path)).map(_.toDouble).collect()
      val dic: Array[(String, Int)] = sc.textFile(dicPath(path)).map(_.split("\t"))
        .filter(_.size == 2).map(x => (x(0), x(1).toInt)).collect()
      new CRFModel((head, dic, alpha))
    }
  }


}
