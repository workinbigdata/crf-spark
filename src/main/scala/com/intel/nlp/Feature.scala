package com.intel.nlp


import scala.collection.mutable.{ArrayBuffer, Map}


trait Feature extends Serializable {

  val featureCache: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  val featureCacheIndex: ArrayBuffer[Int] = new ArrayBuffer[Int]()
  var maxid: Int = 0
  var alpha: Array[Double] = Array[Double]()
  var tokensSize: Int = 0
  var unigramTempls: ArrayBuffer[String] = new ArrayBuffer[String]()
  var bigramTempls: ArrayBuffer[String] = new ArrayBuffer[String]()
  var labels: ArrayBuffer[String] = ArrayBuffer[String]()
  var templs: String = new String
  val dic = Map[String, (Int, Int)]()
  val kMaxContextSize: Int = 8
  val BOS = Array("_B-1", "_B-2", "_B-3", "_B-4",
    "_B-5", "_B-6", "_B-7", "_B-8")
  val EOS = Array("_B+1", "_B+2", "_B+3", "_B+4",
    "_B+5", "_B+6", "_B+7", "_B+8")


  def getId(src: String): Int

  def openTagSet(sentence: Array[String]) {
    var max: Int = 0
    sentence.foreach{ token =>
      val tag = token.split('|')
      if (tag.length > max) max = tag.length
      labels.append(tag(tag.length - 1))
    }
    if(tokensSize > max - 1) throw new RuntimeException("Incompatible formats in Training file")
    tokensSize = max - 1
  }

  /**
   * Build feature index
   */
  def buildFeatures(tagger: Tagger) {
    tagger.setFeatureId(featureCacheIndex.size)
    List(unigramTempls, bigramTempls).foreach{ templs =>
      tagger.x.foreach{ token =>
        featureCacheIndex.append(featureCache.length)
        templs.foreach{ templ =>
          val os = applyRule(templ, tagger.x.indexOf(token), tagger)
          val id = getId(os)
          if (id != -1) featureCache.append(id)
        }
        featureCache.append(-1)
      }
    }
  }

  def applyRule(src: String, idx: Int, tagger: Tagger): String = {
    val templ = src.split(":")
    if (templ.size == 2) {
      val cols = templ(1).split("/").map(_.substring(2))
      templ(0) + ":" + cols.map(getIndex(_, idx, tagger)).reduce(_ + "/" + _)
    } else if (templ.size == 1) {
      templ(0)
    } else
        throw new RuntimeException("Incompatible formats in Template")

  }

  def getIndex(src: String, pos: Int, tagger: Tagger): String = {
    val coor = src.drop(1).dropRight(1).split(",")
    require(coor.size == 2, "Incompatible formats in Template")
    val row = coor(0).toInt
    val col = coor(1).toInt
    if (row < -kMaxContextSize || row > kMaxContextSize ||
      col < 0 || col >= tokensSize) {
      throw new RuntimeException("Incompatible formats in Template")
    }
    val idx = pos + row
    if (idx < 0) {
      BOS(- idx - 1)
    } else if (idx >= tagger.x.size) {
      EOS(idx - tagger.x.size)
    } else {
      tagger.x(idx)(col)
    }
  }
}
