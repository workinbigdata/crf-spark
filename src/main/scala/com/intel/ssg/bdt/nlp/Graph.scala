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

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{Vector => BV}

private[nlp] class Node extends Serializable {
  var x = 0
  var y = 0
  var alpha = 0.0
  var beta = 0.0
  var cost = 0.0
  var bestCost = 0.0
  var prev: Node = _
  var fVector = 0
  val lPath = new ArrayBuffer[Path]()
  val rPath = new ArrayBuffer[Path]()
  val MINUS_LOG_EPSILON = 50.0

  def logSumExp(x: Double, y: Double, flg: Boolean): Double = {
    if (flg) y
    else {
      val vMin: Double = math.min(x, y)
      val vMax: Double = math.max(x, y)
      if (vMax > (vMin + MINUS_LOG_EPSILON)) {
        vMax
      } else {
        vMax + math.log(math.exp(vMin - vMax) + 1.0)
      }
    }
  }

  def calcAlpha(): Unit = {
    alpha = 0.0
    for(i <- lPath.indices)
      alpha = logSumExp(alpha, lPath(i).cost + lPath(i).lNode.alpha, i == 0)
    alpha += cost
  }

  def calcBeta(): Unit = {
    beta = 0.0
    for(i <- rPath.indices)
      beta = logSumExp(beta, rPath(i).cost + rPath(i).rNode.beta, i == 0)
    beta += cost
  }

  def calExpectation(expected: BV[Double], Z: Double, size: Int,
                     featureCache: ArrayBuffer[Int]): Unit = {
    val c: Double = math.exp(alpha + beta -cost - Z)

    var idx: Int = fVector
    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + y) += c
      idx += 1
    }

    for(i <- lPath.indices)
      lPath(i).calExpectation(expected, Z, size, featureCache)

  }
}

private[nlp] class Path extends Serializable {
  var rNode = new Node
  var lNode = new Node
  var cost = 0.0
  var fVector = 0

  def calExpectation(expected: BV[Double], Z: Double,
                     size: Int, featureCache: ArrayBuffer[Int]): Unit = {
    val c: Double = math.exp(lNode.alpha + cost + rNode.beta - Z)
    var idx: Int = fVector

    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + lNode.y * size + rNode.y) += c
      idx += 1
    }
  }

  def add(lnd: Node, rnd: Node): Unit = {
    lNode = lnd
    rNode = rnd
    lNode.rPath.append(this)
    rNode.lPath.append(this)
  }
}
