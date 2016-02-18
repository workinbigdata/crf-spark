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

package com.intel.ssg.bdt.nlp

import org.apache.spark.rdd.RDD

case class CRFModel (
    head: Array[String],
    dic: Array[(String, Int)],
    alpha: Array[Double]) extends Serializable {

  protected def formatVersion = "1.0"

  override def toString: String = {
    val dicString = dic.map{case(k, v) => k + "|-|" + v.toString}
    s"${head.mkString("\t")}|--|${dicString.mkString("\t")}|--|${alpha.mkString("\t")}"
  }

  /**
    * Verify CRF model
    *
    * @param tests  Source files to be verified
    * @param costFactor cost factor
    * @return Source files with the predictive labels
    */
  def predict(
    tests: RDD[Sequence],
    costFactor: Double): RDD[Sequence] = {
      val bcModel = tests.context.broadcast(this)
      tests.map { test =>
        bcModel.value.testCRF(test, costFactor)
      }
  }

  def predict(
    tests: Array[Sequence],
    costFactor: Double): Array[Sequence] = {
    tests.map(this.testCRF(_, costFactor))
  }

  def predict(tests: RDD[Sequence]): RDD[Sequence] = {
    predict(tests, 1.0)
  }

  def predict(tests: Array[Sequence]): Array[Sequence] = {
    predict(tests, 1.0)
  }

  /**
    * Internal method to test the CRF model
    *
    * @param test the sequence to be tested
    * @return the sequence along with predictive labels
    */
  def testCRF(test: Sequence,
              costFactor: Double): Sequence = {
    val deFeatureIdx = new FeatureIndex()
    deFeatureIdx.readModel(this)
    val tagger = new Tagger(deFeatureIdx.labels.size, TestMode)
    tagger.setCostFactor(costFactor)
    tagger.read(test, deFeatureIdx)
    deFeatureIdx.buildFeatures(tagger)
    tagger.parse(deFeatureIdx.alpha)
    Sequence(test.toArray.map(x =>
      Token.put(deFeatureIdx.labels(tagger.result(test.toArray.indexOf(x))), x.tags)
    ))
  }
}

object CRFModel {
  def deSerializer(source: String): CRFModel = {
    val components = source.split("""\|--\|""")
    require(components.length == 3, "Incompatible formats in Model file")
    val head = components(0).split("\t")
    val dic = components(1).split("\t").map(x => {
      val xx = x.split("""\|-\|""")
      require(xx.length == 2, "Incompatible formats in Model file")
      (xx(0), xx(1).toInt)
    })
    val alpha = components(2).split("\t").map(_.toDouble)
    CRFModel(head, dic, alpha)
  }

  def serializer(model: CRFModel): String = {
    model.toString
  }
}
