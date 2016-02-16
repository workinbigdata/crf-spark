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

/**
 * CRF with support for multiple parallel runs
 *
 * regParam = 1/(2.0 * sigma**2)
 */
class CRF private (
    private var freq: Int,
    private var regParam: Double,
    private var maxIterations: Int,
    private var tolerance: Double) extends Serializable with Logging {

  def this() = this(freq = 1, regParam = 0.5, maxIterations = 1000, tolerance = 1E-3)

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
    this.tolerance = eta
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

    logInfo("sentences: %d, features: %d, labels: %d"
      .format(taggers.count(), featureIdx.maxID, featureIdx.labels.length))

    // L2 regularization (TODO: add L1 support)
    val crfLbfgs = new CRFWithLBFGS(new CRFGradient, new L2Updater)
      .setRegParam(regParam)
      .setConvergenceTol(tolerance)

    featureIdx.initAlpha()
    featureIdx.alpha = crfLbfgs.optimizer(taggers, featureIdx.alpha)

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
