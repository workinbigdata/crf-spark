package com.intel.nlp

import scala.collection.mutable.ArrayBuffer


private[nlp] class DecoderFeatureIndex extends Feature {
  def getId(src: String): Int = {
    dic.getOrElse(src, (-1, 0))._1
  }

  def openFromArray(models: CRFModel) = {
    val contents: Array[String] = models.model._1
    models.model._2.foreach(x => dic.update(x._1, (x._2, 1)))
    alpha = models.model._3

    var i: Int = 0
    var readMaxId: Boolean = false
    var readCostFactor: Boolean = false
    var readXSize: Boolean = false
    var readLabels: Boolean = false
    var readUGrams: Boolean = false
    var readBGrams: Boolean = false
    val alpha_tmp = new ArrayBuffer[Double]()
    while (i < contents.length) {
      contents(i) match {
        case "maxid:" =>
          readMaxId = true
        case "cost-factor:" =>
          readMaxId = false
          readCostFactor = true
        case "xsize:" =>
          readCostFactor = false
          readXSize = true
        case "Labels:" =>
          readXSize = false
          readLabels = true
        case "UGrams:" =>
          readLabels = false
          readUGrams = true
        case "BGrams:" =>
          readUGrams = false
          readBGrams = true
        case _ =>
          i -= 1
      }
      i += 1
      if (readMaxId) {
        maxid = contents(i).toInt
      } else if (readXSize) {
        tokensSize = contents(i).toInt
      } else if (readLabels) {
        labels.append(contents(i))
      } else if (readUGrams) {
        unigramTempls.append(contents(i))
      } else if (readBGrams) {
        bigramTempls.append(contents(i))
      }
      i += 1
    }
    this
  }
}