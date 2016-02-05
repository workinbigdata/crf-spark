package com.intel.nlp

private[nlp] class Params(var err_num: Int, var zeroOne: Int, var expected: Array[Double], var obj: Double) extends Serializable {
    def merge(A: Params) = {
        this.err_num += A.err_num
        this.zeroOne += A.zeroOne
        this.expected = (breeze.linalg.Vector(A.expected) + breeze.linalg.Vector(this.expected)).toArray
        this.obj += A.obj
        this
    }
}