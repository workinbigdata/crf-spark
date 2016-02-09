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

private[nlp] class Params (
    var err_num: Int,
    var zeroOne: Int,
    var expected: Array[Double],
    var obj: Double) extends Serializable {
  def merge(A: Params) = {
    this.err_num += A.err_num
    this.zeroOne += A.zeroOne
    this.expected = (breeze.linalg.Vector(A.expected) + breeze.linalg.Vector(this.expected)).toArray
    this.obj += A.obj
    this
  }
}
