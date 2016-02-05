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

import breeze.numerics.{exp, log}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.broadcast.Broadcast

private[nlp] class Tagger (var ySize: Int,
                           mode: Int) extends Serializable {
  // mode: LEARN 2, TEST 1
  var nBest: Int = 0
  var cost: Double = 0.0
  var Z: Double = 0.0
  var feature_id: Int = 0
  var thread_id: Int = 0
  var x: ArrayBuffer[Array[String]] = new ArrayBuffer[Array[String]]()
  var node: ArrayBuffer[ArrayBuffer[Node]] = new ArrayBuffer[ArrayBuffer[Node]]
  var answer: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  var result: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  val MINUS_LOG_EPSILON = 50
  var obj: Double = 0.0
  var featureCache = Array[Int]()
  var featureCacheH= Array[Int]()
  var costFactor: Double = 1.0

  def setCostFactor(costFactor: Double) = {
    this.costFactor = costFactor
    this
  }

  def read(lines: Array[String], feature_idx: Feature) {
    var i: Int = 0
    var columns: Array[String] = null
    var j: Int = 0
    lines.foreach{ t =>
      columns = t.split('|')
      if (mode == 2) {
        for (y <- feature_idx.labels if y.equals(columns(feature_idx.tokensSize)))
          answer.append(feature_idx.labels.indexOf(y))
        x.append(columns.dropRight(1))
      } else {
        x.append(columns)
        answer.append(0)
      }
      result.append(0)
    }
  }

  def setFeatureId(id: Int): Unit = {
    feature_id = id
  }


  /**
   * Build the matrix to calculate
   * cost of each node according to template
   */
  def buildLattice(bcAlpha: Broadcast[Array[Double]]): Unit = {
    require(x.nonEmpty, "This sentence is null!")
    rebuildFeatures

    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        node(i)(j) = calcCost(node(i)(j), bcAlpha)
        var k = 0
        while (k < node(i)(j).lPath.length) {
          node(i)(j).lPath(k) = calcCost(node(i)(j).lPath(k), bcAlpha)
          k += 1
        }
      }
    }

  }



  def calcCost(n: Node, bcAlpha: Broadcast[Array[Double]]): Node = {
    require(bcAlpha.value.nonEmpty, "There is no Alpha[Double] Broadcasted")
    n.cost = 0.0
    var cd: Double = 0.0
    var idx: Int = n.fVector
    while (featureCache(idx) != -1) {
        cd += bcAlpha.value(featureCache(idx) + n.y)
        n.cost = cd
        idx += 1
    }
    n
  }

  def calcCost(p: Path, bcAlpha: Broadcast[Array[Double]]): Path = {
    var c: Float = 0
    var cd: Double = 0.0
    var idx: Int = p.fVector
    p.cost = 0.0
    if (bcAlpha.value.nonEmpty) {
      while (featureCache(idx) != -1) {
        cd += bcAlpha.value(featureCache(idx) +
          p.lNode.y * ySize + p.rNode.y)
        p.cost = cd
        idx += 1
      }
    }
    p
  }


  /**
   * Set node relationship and its feature index.
   * Node represents a word.
   */
  def rebuildFeatures {
    var fid = feature_id

    for (i <- x.indices) {
      val nodeList: ArrayBuffer[Node] = new ArrayBuffer[Node]()
      node.append(nodeList)
      var k = 0
      while (k < ySize) {
        val nd = new Node
        nd.x = i
        nd.y = k
        nd.fVector = featureCacheH(fid)
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
          path.fVector = featureCacheH(fid)
          k += 1
        }
      }
      fid += 1
    }
  }

  /**
   *Calculate the expectation of each node
   * https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
   * http://www.cs.columbia.edu/~mcollins/fb.pdf
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
    var bestc: Double = -1e37
    var best: Node = null

    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        bestc = -1E37
        best = null
        var k = 0
        while (k < node(i)(j).lPath.length) {
          val cost = node(i)(j).lPath(k).lNode.bestCost + node(i)(j).lPath(k).cost + node(i)(j).cost
          if (cost > bestc) {
            bestc = cost
            best = node(i)(j).lPath(k).lNode
          }
          k += 1
        }
        node(i)(j).prev = best
        if (best != null) {
          node(i)(j).bestCost = bestc
        } else {
          node(i)(j).bestCost = node(i)(j).cost
        }
      }
    }

    bestc = -1E37
    best = null

    for (j <- 0 until ySize) {
      if (node(x.length - 1)(j).bestCost > bestc) {
        best = node(x.length - 1)(j)
        bestc = node(x.length - 1)(j).bestCost
      }
    }
    var nd = best
    while (nd != null) {
      result.update(nd.x, nd.y)
      nd = nd.prev
    }
    cost = -node(x.length - 1)(result(x.length - 1)).bestCost   // TODO cost Never used
  }

  def gradient(expected: Array[Double], bcAlpha: Broadcast[Array[Double]]): Double = {


    buildLattice(bcAlpha)
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

  def eval(): Int = {
    var err: Int = 0
    for(i <- x.indices)
      if (answer(i) != result(i))
        err += 1
    err
  }

  /**
   * simplify the log likelihood.
   */
  def logSumExp(x: Double, y: Double, flg: Boolean): Double = {
    if (flg) return y
    val vMin: Double = math.min(x, y)
    val vMax: Double = math.max(x, y)
    if (vMax > vMin + MINUS_LOG_EPSILON) vMax else vMax + log(exp(vMin - vMax) + 1.0)

}

  def parse(alpha: Array[Double]): Unit = {
    buildLattice(alpha)
    if (nBest != 0) {
      forwardBackward()
    }   //TODO add nBest support
    viterbi()
  }


  def createOutput(y: ArrayBuffer[String]) = {
    val content: ArrayBuffer[String] = new ArrayBuffer[String]
    x.foreach{ s =>
      s.foreach{ t =>
        content.append(t)}
      content.append(y(result(x.indexOf(s))))
    }
    content.toArray
  }

  def buildLattice(alpha: Array[Double]): Unit = {

    require(x.nonEmpty, "This sentence is null!")
    rebuildFeatures
    for (i <- x.indices) {
      for (j <- 0 until ySize) {
        node(i)(j) = calcCost(node(i)(j), alpha)
        var k: Int = 0
        while (k < node(i)(j).lPath.length) {
          node(i)(j).lPath(k) = calcCost(node(i)(j).lPath(k), alpha)
          k += 1
        }
      }
    }
  }

  def calcCost(n: Node, alpha: Array[Double]): Node = {
    var cd: Double = 0.0
    var idx: Int = n.fVector
    n.cost = 0.0
    if (alpha.nonEmpty) {
      while (featureCache(idx) != -1) {
        cd += alpha(featureCache(idx) + n.y)
        n.cost = cd * costFactor
        idx += 1
      }
    }
    n
  }

  def calcCost(p: Path, alpha: Array[Double]): Path = {
    var cd: Double = 0.0
    var idx: Int = p.fVector
    p.cost = 0.0
    if (alpha.nonEmpty) {
      while (featureCache(idx) != -1) {
        cd += alpha(featureCache(idx) +
          p.lNode.y * ySize + p.rNode.y)
        p.cost = cd * costFactor
        idx += 1
      }
    }
    p
  }

  def cloneFeature(feature: Feature) = {
    featureCache = Array.fill(feature.featureCache.size)(0)
    featureCacheH = Array.fill(feature.featureCacheIndex.size)(0)
    feature.featureCache.copyToArray(featureCache)
    feature.featureCacheIndex.copyToArray(featureCacheH)
    this
  }
}
