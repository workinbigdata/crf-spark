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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.riso.numerical.LBFGS


class CRF private (
    private var regParam: Double,
    private var freq: Int,
    private var maxIterations: Int,
    private var eta: Double) extends Serializable with Logging {
    //TODO add var orthant: Boolean for L1-CRF support

  def this() = this(regParam = 0.5, freq = 1, maxIterations = 100000, eta = 1E-4)

  def setRegParam(regParam: Double) = {
    this.regParam = regParam
    this
  }

  def setFreq(freq: Int) = {
    this.freq = freq
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
   * @param trains the source for the training
   * @return the model of the source
   */
  def runCRF(
    template: Array[String],
    trains: RDD[Sequence]): CRFModel = {
    val featureIdx = new FeatureIndex()
    featureIdx.openTemplate(template)
    featureIdx.openTagSetDist(trains)

    val bcFeatureIdxI: Broadcast[FeatureIndex] = trains.context.broadcast(featureIdx)
    val taggers = trains.map(train => {
      val tagger: Tagger = new Tagger(bcFeatureIdxI.value.labels.size, LearnMode)
      tagger.read(train, bcFeatureIdxI.value)
      tagger
    })

    featureIdx.buildDictionaryDist(taggers, bcFeatureIdxI, freq)

    val bcFeatureIdxII = trains.context.broadcast(featureIdx)
    val taggerList: RDD[Tagger] = taggers.map(bcFeatureIdxII.value.buildFeatures).cache()

    val model = runAlgorithm(taggerList, featureIdx)
    taggerList.unpersist()

    model
  }

  /**
   *
   * @param taggers the tagger in the template
   * @param featureIdx the index of the feature
   */
  def runAlgorithm(
    taggers: RDD[Tagger],
    featureIdx: FeatureIndex): CRFModel = {
    var oldObj: Double = 1E37
    var converge: Int = 0
    var itr: Int = 0
    val all = taggers.map(_.x.size).reduce(_ + _)
    val sentences = taggers.count()
    val iFlag: Array[Int] = Array(0)
    val diagH = Array.fill(featureIdx.maxID)(0.0)
    val iPrint = Array(-1, 0)
    val xTol = 1.0E-16

    while (itr < maxIterations) {

      val bcAlpha: Broadcast[Array[Double]] = taggers.context.broadcast(featureIdx.alpha)
      val treeDepth = math.ceil(math.log(taggers.partitions.length) / (math.log(2) * 2)).toInt
      val results : Params = taggers.mapPartitions{ x =>
        val expected: Array[Double] = Array.fill(featureIdx.maxID)(0.0)
        var obj: Double = 0.0
        var errNum: Int = 0
        var zeroOne: Int = 0
        while (x.hasNext){
          val cur = x.next()
          obj += cur.gradient(expected, bcAlpha.value)
          val err = cur.eval()
          errNum += err
          if(err != 0) zeroOne += 1
        }
        Iterator(Params(errNum, zeroOne, expected, obj))
      }.treeReduce((p1, p2) => p1.merge(p2), treeDepth)

      // L2 regularization, TODO add L1 support
      // regParam = 1/(2.0 * sigma^2)
      for(k <- featureIdx.alpha.indices) {
        results.obj += featureIdx.alpha(k) * featureIdx.alpha(k) * regParam
        results.expected(k) += featureIdx.alpha(k) * regParam * 2.0
      }

      val diff = if (itr == 0) 1.0 else math.abs((oldObj - results.obj) / oldObj)
      oldObj = results.obj

      logInfo("iter=%d, terr=%2.5f, serr=%2.5f, act=%d, obj=%2.5f, diff=%2.5f".format(
        itr, 1.0 * results.err_num / all,
        1.0 * results.zeroOne / sentences, featureIdx.maxID,
        results.obj, diff))

      LBFGS.lbfgs(featureIdx.maxID, 5,
        featureIdx.alpha, results.obj,
        results.expected, false,
        diagH, iPrint, 1.0E-3, xTol, iFlag)

      converge = if (diff < eta) converge + 1 else 0
      itr = if (converge == 3) maxIterations + 1 else itr + 1
    }

    featureIdx.saveModel
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
  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double,
    freq: Int,
    maxIterion: Int,
    eta: Double): CRFModel = {
    new CRF().setRegParam(regParam)
      .setFreq(freq)
      .setMaxIterations(maxIterion)
      .setEta(eta)
      .runCRF(templates, train)
  }

  def train(
             templates: Array[String],
             train: RDD[Sequence],
             regParam: Double,
             freq: Int): CRFModel = {
    new CRF().setRegParam(regParam)
      .setFreq(freq)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence],
    regParam: Double): CRFModel = {
    new CRF().setRegParam(regParam)
      .runCRF(templates, train)
  }

  def train(
    templates: Array[String],
    train: RDD[Sequence]): CRFModel = {
    new CRF().runCRF(templates, train)
  }
}
