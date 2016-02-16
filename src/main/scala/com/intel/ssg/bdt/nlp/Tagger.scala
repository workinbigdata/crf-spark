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

import breeze.linalg.{DenseVector => BDV, Vector => BV}

private[nlp] trait Mode

private[nlp] case object LearnMode extends Mode

private[nlp] case object TestMode extends Mode

private[nlp] class Tagger (
    ySize: Int,
    mode: Mode) extends Serializable {
  var nBest = 0
  var cost = 0.0
  var Z = 0.0
  val MINUS_LOG_EPSILON = 50
  var obj = 0.0
  var costFactor = 1.0
  val x = new ArrayBuffer[Array[String]]()
  val node = new ArrayBuffer[ArrayBuffer[Node]]()
  val answer = new ArrayBuffer[Int]()
  val result = new ArrayBuffer[Int]()
  val featureCache = new ArrayBuffer[Int]()
  val featureCacheIndex = new ArrayBuffer[Int]()


  def setCostFactor(costFactor: Double) = {
    this.costFactor = costFactor
    this
  }

  def read(lines: Sequence, feature_idx: FeatureIndex): Unit = {
    var columns: Array[String] = null
    lines.toArray.foreach{ t =>
      mode match {
      case LearnMode =>
        for (y <- feature_idx.labels if y.equals(t.label))
          answer.append(feature_idx.labels.indexOf(y))
        x.append(t.tags)
      case TestMode =>
        x.append(t.tags)
        answer.append(0)
      }
      result.append(0)
    }
  }

  /**
   * Set node relationship and its feature index.
   * Node represents a token.
   */
  def rebuildFeatures: Unit = {
    var fid = 0

    for (i <- x.indices) {
      val nodeList: ArrayBuffer[Node] = new ArrayBuffer[Node]()
      node.append(nodeList)
      var k = 0
      while (k < ySize) {
        val nd = new Node
        nd.x = i
        nd.y = k
        nd.fVector = featureCacheIndex(fid)
        nodeList.append(nd)
        k += 1
      }
      fid += 1
      node.update(i, nodeList)
    }

    for (i <- 1 until x.size) {
      for (j <- 0 until ySize) {
        var k = 0
        while (k < ySize) {
          val path: Path = new Path
          path.add(node(i - 1)(j), node(i)(k))
          path.fVector = featureCacheIndex(fid)
          k += 1
        }
      }
      fid += 1
    }
  }

  /**
   * Calculate the expectation of each node
   */
  def forwardBackward(): Unit = {
    require(x.nonEmpty, "This sentence is null!")


    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        node(i)(j).calcAlpha()
      }
    }
    var idx: Int = x.length - 1
    while (idx >= 0) {
      for (j <- 0 until ySize) {
        node(idx)(j).calcBeta()
      }
      idx -= 1
    }

    Z = 0.0
    for (i <- 0 until ySize) {
      Z = logSumExp(Z, node(0)(i).beta, i == 0)
    }

  }

  /**
   * Get the max expectation in the nodes and predicts the most likely label
   * http://www.cs.utah.edu/~piyush/teaching/structured_prediction.pdf
   * http://www.weizmann.ac.il/mathusers/vision/courses/2007_2/files/introcrf.pdf
   * Page 15
   */
  def viterbi(): Unit = {
    var bestCost: Double = -1e37
    var best: Node = null

    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        bestCost = -1E37
        best = null
        var k = 0
        while (k < node(i)(j).lPath.length) {
          val cost = node(i)(j).lPath(k).lNode.bestCost + node(i)(j).lPath(k).cost + node(i)(j).cost
          if (cost > bestCost) {
            bestCost = cost
            best = node(i)(j).lPath(k).lNode
          }
          k += 1
        }
        node(i)(j).prev = best
        if (best != null) {
          node(i)(j).bestCost = bestCost
        } else {
          node(i)(j).bestCost = node(i)(j).cost
        }
      }
    }

    bestCost = -1E37
    best = null

    for (j <- 0 until ySize) {
      if (node(x.length - 1)(j).bestCost > bestCost) {
        best = node(x.length - 1)(j)
        bestCost = node(x.length - 1)(j).bestCost
      }
    }
    var nd = best
    while (nd != null) {
      result.update(nd.x, nd.y)
      nd = nd.prev
    }
    cost = -node(x.length - 1)(result(x.length - 1)).bestCost   // (TODO: cost will be used for nbest)
  }

  def gradient(expected: BV[Double], alpha: BDV[Double]): Double = {

    buildLattice(alpha)
    forwardBackward()

    for(i <- x.indices)
      for(j <- 0 until ySize)
        node(i)(j).calExpectation(expected, Z, ySize, featureCache)

    var s: Double = 0.0
    for (i <- x.indices) {
      var rIdx = node(i)(answer(i)).fVector
      while (featureCache(rIdx) != -1) {
        expected(featureCache(rIdx) + answer(i)) -= 1.0
        rIdx += 1
      }
      s += node(i)(answer(i)).cost
      var j = 0
      while (j < node(i)(answer(i)).lPath.length) {
        val lNode = node(i)(answer(i)).lPath(j).lNode
        val rNode = node(i)(answer(i)).lPath(j).rNode
        val lPath = node(i)(answer(i)).lPath(j)
        if (lNode.y == answer(lNode.x)) {
          rIdx = lPath.fVector
          while (featureCache(rIdx) != -1) {
            expected(featureCache(rIdx) + lNode.y * ySize + rNode.y) -= 1.0
            rIdx += 1
          }
          s += lPath.cost
        }
        j += 1
      }
    }

    viterbi()
    node.clear()
    Z - s
  }

  /**
   * simplify the log likelihood.
   */
  def logSumExp(x: Double, y: Double, flg: Boolean): Double = {
    if (flg) return y
    val vMin: Double = math.min(x, y)
    val vMax: Double = math.max(x, y)
    if (vMax > vMin + MINUS_LOG_EPSILON) vMax else vMax + math.log(math.exp(vMin - vMax) + 1.0)

}

  def parse(alpha: BDV[Double]): Unit = {
    buildLattice(alpha)
    if (nBest != 0) {
      forwardBackward()
    }   // (TODO: add nBest support)
    viterbi()
  }

  def buildLattice(alpha: BDV[Double]): Unit = {

    require(x.nonEmpty, "This sentence is null!")
    rebuildFeatures
    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        val n = calcCost(node(i)(j), alpha)
        var k: Int = 0
        while (k < n.lPath.length) {
          n.lPath(k) = calcCost(n.lPath(k), alpha)
          k += 1
        }
        node(i)(j) = n
      }
    }
  }

  def calcCost(n: Node, alpha: BDV[Double]): Node = {
    var cd: Double = 0.0
    var idx: Int = n.fVector
    n.cost = 0.0

    while (featureCache(idx) != -1) {
      cd += alpha(featureCache(idx) + n.y)
      n.cost = cd * costFactor
      idx += 1
    }

    n
  }

  def calcCost(p: Path, alpha: BDV[Double]): Path = {
    var cd: Double = 0.0
    var idx: Int = p.fVector
    p.cost = 0.0

    while (featureCache(idx) != -1) {
      cd += alpha(featureCache(idx) +
        p.lNode.y * ySize + p.rNode.y)
      p.cost = cd * costFactor
      idx += 1
    }

    p
  }
}
