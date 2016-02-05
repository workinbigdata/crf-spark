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

import scala.collection.mutable.ArrayBuffer


private[nlp] class EncoderFeatureIndex extends Feature {

  /**
   * Read one template file
    *
    * @param lines the unit template file
   */
  def openTemplate(lines: Array[String]): Unit = {
    var i: Int = 0
    lines.foreach{ t =>
      t.charAt(0) match{
        case 'U' => unigramTempls += t
        case 'B' => bigramTempls += t
        case '#' =>
        case _ => throw new RuntimeException("Incompatible formats in Templates")
      }}

    unigramTempls.foreach(templs += _)
    bigramTempls.foreach(templs += _)
  }

  def shrink(freq: Int)  {  //TODO check if works
    var newMaxId: Int = 0
    val key: String = ""
    val count: Int = 0
    val currId: Int = 0

    if (freq > 1) {
      while (dic.iterator.hasNext) {
        dic.getOrElse(key, (currId, count))
        if (count > freq) {
          dic.getOrElseUpdate(key, (newMaxId, count))
        }

        if (key.toString.charAt(0) == 'U') {
          newMaxId += labels.size
        } else {
          newMaxId += labels.size * labels.size
        }
      }
      maxid = newMaxId
    }
  }

  def getId(src: String): Int = {

    if (dic.get(src).isEmpty) {
      dic.update(src, (maxid, 1))
      val n = maxid
      if (src.charAt(0) == 'U') {
        // Unigram
        maxid += labels.size
      } else {
        // Bigram
        maxid += labels.size * labels.size
      }
      n
    } else {
      val idx = dic.get(src).get._2 + 1
      val fid = dic.get(src).get._1
      dic.update(src, (fid, idx))
      fid
    }
  }

  def initAlpha(size: Int): Unit = {
    alpha = Array.fill(maxid)(0.0)
  }

  def saveModel: (Array[String], Array[(String, Int)], Array[Double]) = {
    val head = new ArrayBuffer[String]()

    head.append("maxid:")
    head.append(maxid.toString)
    head.append("cost-factor:")
    head.append(1.toString)
    head.append("xsize:")
    head.append(tokensSize.toString)
    head.append("Labels:")
    labels.foreach(head.append(_))
    head.append("UGrams:")
    unigramTempls.foreach(head.append(_))
    head.append("BGrams:")
    bigramTempls.foreach(head.append(_))

    (head.toArray, dic.map { case (k, v) => (k, v._1) }.toArray, alpha)
  }
}

