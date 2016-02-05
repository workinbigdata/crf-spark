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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import riso.numerical.LBFGS
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import breeze.linalg.{DenseVector => BDV}

class CRF private (
    private var C: Double,
    private var maxIterations: Int,
    private var eta: Double,
    private var freq: Int) extends Serializable {
    //TODO add var orthant: Boolean for L1-CRF support

  def this() = this(C = 1.0, maxIterations = 1000, eta = 1E-4, freq = 1)


  def setFreq(freq: Int) = {
    this.freq = 1   //TODO add new feature for freq > 1
    this
  }


  def setC(C: Double) = {
    this.C = C
    this
  }

  def setMaxIterations(maxIterations: Int) = {
    this.maxIterations = maxIterations
    this
  }

  def setEta(eta: Double) = {
    this.eta = eta
    this
  }


  /**
   * Internal method to train the CRF model
    *
    * @param template the template to train the model
   * @param train the source for the training
   * @return the model of the source
   */
  def runCRF(template: Array[String],
            train: RDD[Array[String]]): CRFModel = {
    val featureIdx: EncoderFeatureIndex = new EncoderFeatureIndex()
    val taggerList: ArrayBuffer[Tagger] = new ArrayBuffer[Tagger]()
    featureIdx.openTemplate(template)

    val trains: Array[Array[String]] = train.collect()
    trains.foreach(t => featureIdx.openTagSet(t))
    featureIdx.labels = featureIdx.labels.distinct
    trains.foreach{ train =>
      val tagger: Tagger = new Tagger(featureIdx.labels.size, 2)
      tagger.read(train, featureIdx)
      featureIdx.buildFeatures(tagger)
      taggerList.append(tagger)
    }
    featureIdx.shrink(freq)
    featureIdx.initAlpha(featureIdx.maxid)

    val tagger = train.zipWithIndex().map(x => taggerList(x._2.toInt))
    val model = runAlgorithm(tagger, featureIdx, train.sparkContext)

    model
  }

  /**
   *
   * @param tagger the tagger in the template
   * @param featureIdx the index of the feature
   */

  def runAlgorithm(tagger: RDD[Tagger], featureIdx: EncoderFeatureIndex,
             sc: SparkContext): CRFModel = {
    var old_obj: Double = 1E37
    var converge: Int = 0
    var itr: Int = 0
    val all = tagger.map(_.x.size).reduce(_ + _)
    val sentences = tagger.count()
    val tagger__ = tagger.map(_.cloneFeature(featureIdx)).cache()
    val iFlag: Array[Int] = Array(0)
    val diagH = Array.fill(featureIdx.maxid)(0.0)
    val iPrint = Array(-1, 0)
    val xTol = 1.0E-16


    while (itr < maxIterations) {

      val bcAlpha: Broadcast[Array[Double]] = sc.broadcast(featureIdx.alpha)

      val tagger_ : Params = tagger__.mapPartitions{ x =>
        val expected: Array[Double] = Array.fill(featureIdx.maxid)(0.0)
        var obj_I: Double = 0.0
        var err_num: Int = 0
        var zeroOne: Int = 0
        while (x.hasNext){
          val cur = x.next
          obj_I += cur.gradient(expected, bcAlpha)
          val err = cur.eval()
          err_num += err
          if(err != 0) zeroOne += 1
        }
        List(new Params(err_num, zeroOne, expected, obj_I)).iterator
      }.treeReduce((p1, p2) => p1.merge(p2), 4)


      // L2 regularization
      for(k <- featureIdx.alpha.indices) {
        tagger_.obj += featureIdx.alpha(k) * featureIdx.alpha(k) / (2.0 * C) //TODO add L1 support
        tagger_.expected(k) += featureIdx.alpha(k) / C
      }

      val diff = if (itr == 0) 1.0 else math.abs((old_obj - tagger_.obj) / old_obj)
      old_obj = tagger_.obj

      printf("iter=%d, terr=%2.5f, serr=%2.5f, act=%d, obj=%2.5f, diff=%2.5f\n",
        itr, 1.0 * tagger_.err_num / all,
        1.0 * tagger_.zeroOne / sentences, featureIdx.maxid,
        tagger_.obj, diff)


      LBFGS.lbfgs(featureIdx.maxid, 5,
        featureIdx.alpha, tagger_.obj,
        tagger_.expected, false,
        diagH, iPrint, 1.0E-3, xTol, iFlag)

      converge = if (diff < eta) converge + 1 else 0
      itr = if (converge == 3) maxIterations + 1 else itr + 1
    }

    new CRFModel(featureIdx.saveModel)
  }
}

/**
  * Top-level methods for calling CRF.
  */
object CRF {

  /**
   * Train CRF Model
   *
   * @param templates Source templates for training the model
   * @param train Source files for training the model
   * @return Model
   */
  def train(templates: Array[String],
            train: RDD[Array[String]],
            C: Int,
            maxIter: Int,
            freq: Int,
            eta: Double): CRFModel = {
    new CRF().setC(C)
      .setEta(eta)
      .setFreq(freq)
      .setMaxIterations(maxIter)
      .runCRF(templates, train)
  }

  def train(templates: Array[String],
            train: RDD[Array[String]]): CRFModel = {
    new CRF().runCRF(templates, train)
  }


}
