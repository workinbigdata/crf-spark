package com.intel.nlp

/**
 * An example demonstrating a CRF.
 * Run with
 * {{{
 * bin/run-example ml.ConditionalRandomFieldExample <modelFile> <featureFile>
 * }}}
 */

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ConditionalRandomFieldExample {
  def main(args: Array[String]) {
    if (args.length != 4) {
      // scalastyle:off println
      System.err.println("Usage: ml.CRFExample <templateFile> <trainFile> <testFile>")
      // scalastyle:on println
      System.exit(1)
    }
    val template = args(0)
    val train = args(1)
    val test = args(2)
    val slices = args(3).toInt
    val modelPath = "result/CRFModel"

    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}")
    val sc = new SparkContext(conf)

    val templates: Array[String] = sc.textFile(template).filter(_.nonEmpty).map(_.split("\t")).collect().flatten
    val trainRDD: RDD[Array[String]] = sc.textFile(train, slices).filter(_.nonEmpty).map(_.split("\t"))
    val model: CRFModel = CRF.train(templates, trainRDD)


    val testRDD: RDD[Array[String]] = sc.textFile(test, slices).filter(_.nonEmpty).map(_.split("\t"))
//    val model: CRFModel = CRFModel.load(sc, modelPath)
    val results = model.predict(testRDD).collect()

    var i = 0
    var j = 0
    var right = 0
    var total = 0
    val xsize = new DecoderFeatureIndex().openFromArray(model).tokensSize

    while (i < results.length){
      while(j < results(i).length) {
        if (results(i)(j + xsize) == results(i)(j + xsize + 1)) right += 1
        total += 1
        j += xsize + 2
      }
      j = 0
      i += 1
    }
    println(s"Results: ${right} / ${total}")

//    model.save(sc, modelPath)
    sc.stop()
  }

}
