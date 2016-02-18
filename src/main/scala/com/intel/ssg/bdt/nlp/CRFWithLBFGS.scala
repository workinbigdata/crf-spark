package com.intel.ssg.bdt.nlp

import scala.collection.mutable

import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import breeze.linalg.{DenseVector => BDV, sum => Bsum}

import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.linalg.{Vector => SparkVector}


class CRFWithLBFGS(private var gradient: CRFGradient, private var updater: Updater)
  extends LBFGS(gradient: Gradient, updater: Updater) {

  private val numCorrections = 5
  private var maxNumIterations = 100
  private var convergenceTol = 1E-4
  private var regParam = 0.5

  /**
   * Set the regularization parameter. Default 0.5.
   */
  override def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance of iterations for L-BFGS. Default 1E-4.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * This value must be nonnegative. Lower convergence values are less tolerant
   * and therefore generally cause more iterations to be run.
   */
  override def setConvergenceTol(tolerance: Double): this.type = {
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the maximal number of iterations for L-BFGS. Default 100.
   */
  override def setNumIterations(iters: Int): this.type = {
    this.maxNumIterations = iters
    this
  }

  def optimizer(data: RDD[Tagger], initialWeights: BDV[Double]): BDV[Double] = {
    CRFWithLBFGS.runLBFGS(data,
      gradient,
      updater,
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeights)
  }
}

object CRFWithLBFGS extends Logging {
  def runLBFGS(
      data: RDD[Tagger],
      gradient: CRFGradient,
      updater: Updater,
      numCorrections: Int,
      convergenceTol: Double,
      maxNumIterations: Int,
      regParam: Double,
      initialWeights: BDV[Double]): BDV[Double] = {

    val costFun = new CostFun(data, gradient, updater, regParam)

    val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)

    val states = lbfgs.iterations(new CachedDiffFunction[BDV[Double]](costFun), initialWeights)

    val lossHistory = mutable.ArrayBuilder.make[Double]
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }

//    logInfo("LBFGS.runLBFGS finished. last 10 losses: %s".format(
//      lossHistory.result().takeRight(10).mkString(" -> ")))
    state.x
  }
}

class CRFGradient extends Gradient {
  def compute(
     data: SparkVector,
     label: Double,
     weights: SparkVector,
     cumGradient: SparkVector): Double = {
    throw new Exception("The original compute() method is not supported")
  }

  def computeCRF(sentences: Iterator[Tagger], weights: BDV[Double]): (BDV[Double], Double) = {

    val expected = BDV.zeros[Double](weights.length)
    var obj: Double = 0.0
    while (sentences.hasNext)
      obj += sentences.next().gradient(expected, weights)

    (expected, obj)
  }
}

class L2Updater extends Updater {
  def compute(
      weightsOld: SparkVector,
      gradient: SparkVector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (SparkVector, Double) = {
    throw new Exception("The original compute() method is not supported")
  }

  def computeCRF(
      weightsOld: BDV[Double],
      gradient: BDV[Double],
      regParam: Double): (BDV[Double], Double) ={
    val loss = Bsum(weightsOld :* weightsOld :* regParam)
    gradient :+= weightsOld :* (regParam * 2.0)
    (gradient, loss)
  }
}

private class CostFun(
    taggers: RDD[Tagger],
    gradient: CRFGradient,
    updater: Updater,
    regParam: Double) extends DiffFunction[BDV[Double]] with Serializable {

  override def calculate(weigths: BDV[Double]): (Double, BDV[Double]) = {

    val bcWeights = taggers.context.broadcast(weigths)
    lazy val treeDepth = math.ceil(math.log(taggers.partitions.length) / (math.log(2) * 2)).toInt

    val (expected, obj) = taggers.mapPartitions(sentences =>
      Iterator(gradient.computeCRF(sentences, bcWeights.value))
    ).treeReduce((p1, p2) => (p1, p2) match {
      case ((expected1, obj1),(expected2, obj2)) =>
        (expected1 + expected2, obj1 + obj2)}, treeDepth)

    updater match {
      case updater: L2Updater =>
        val (grad, loss) = updater.computeCRF(weigths, expected, regParam)
        (obj + loss, grad)
      case _ =>
        throw new Exception("only support L2-CRF now")
    }
  }
}

