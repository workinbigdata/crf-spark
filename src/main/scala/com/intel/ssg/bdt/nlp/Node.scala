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

private[nlp] class Node extends Serializable {
  var x: Int = 0
  var y: Int = 0
  var alpha: Double = 0.0
  var beta: Double = 0.0
  var cost: Double = 0.0
  var bestCost: Double = 0.0
  var prev: Node = _
  var fVector: Int = 0
  val lPath: ArrayBuffer[Path] = new ArrayBuffer[Path]()
  val rPath: ArrayBuffer[Path] = new ArrayBuffer[Path]()
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

  def calExpectation(expected: Array[Double], Z: Double, size: Int,
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
