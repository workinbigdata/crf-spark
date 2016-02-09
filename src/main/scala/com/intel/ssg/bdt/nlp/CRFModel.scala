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
    val dicString = dic.map(x => x._1 + "|-|" + x._2.toString)
    s"${head.mkString("\t")}\n${dicString.mkString("\t")}\n${alpha.mkString("\t")}"
  }

  /**
    * Verify CRF model
    *
    * @param tests  Source files to be verified
    * @param costFactor cost factor
    * @return Source files with the predictive labels
    */
  def predict[T] (
    tests: T,
    costFactor: Double): T = {
    tests match {
      case tests: RDD[_] =>
        if(tests.first().isInstanceOf[Sequence]) {
          val bcModel = tests.context.broadcast(this)
          tests.asInstanceOf[RDD[Sequence]].map { test =>
            bcModel.value.testCRF(test, costFactor)
          }.asInstanceOf[T]
        } else{
          throw new RuntimeException("Incompatible formats in Testing file")
        }
      case tests: Sequence =>
        this.testCRF(tests.asInstanceOf[Sequence], costFactor).asInstanceOf[T]
      case tests: Array[Sequence] =>
        tests.asInstanceOf[Array[Sequence]].map { test =>
          this.testCRF(test, costFactor)
        }.asInstanceOf[T]
      case _ =>
        throw new RuntimeException("Incompatible formats in Testing file")
    }
  }

  def predict[T] (tests: T): T = {
    predict(tests, 1.0)
  }

  /**
    * Internal method to test the CRF model
    *
    * @param test the line to be tested
    * @return the line along with predictive labels
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
  def parse(s: Array[String]): CRFModel = {
    require(s.length == 3, "Incompatible formats in Model file")
    val head = s(0).split("\t")
    val dic = s(1).split("\t").map(x => {
      val xx = x.split("""\|-\|""")
      require(xx.length == 2, "Incompatible formats in Model file")
      (xx(0), xx(1).toInt)
    })
    val alpha = s(2).split("\t").map(_.toDouble)
    CRFModel(head, dic, alpha)
  }
}
