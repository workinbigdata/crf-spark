# CRF-Spark
A Spark-based implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data.

## Requirements
This documentation is for Spark 1.4+. Other version will probably work yet not tested.

## Features

`CRF-Spark` provides following features:
* Training in parallel based on Spark RDD
* Support a simple format of training and test file. Any other format also can be read by a simple tokenizer.
* A common feature templates design, which is also used in other machine learning tools, such as [CRF++](https://taku910.github.io/crfpp/) and [miralium](https://code.google.com/archive/p/miralium/)
* Fast training based on LBFGS, a quasi-newton algorithm for large scale numerical optimization problem
* Linear-chain (first-order Markov) CRF
* Test can run both in parallel and in serial

## Example

### Scala API

```scala
  val template = Array("U00:%x[-1,0]", "U01:%x[0,0]", "U02:%x[1,0]", "B")
  val train = Array("B-NP|--|Friday|-|NNP\tB-NP|--|'s|-|POS", "I-NP|--|Market|-|NNP\tI-NP|--|Activity|-|NN")
  val test = Array("null|--|Market|-|NNP\tnull|--|Activity|-|NN")

  val trainRdd = sc.parallelize(train).map(Sequence.deSerializer)

  val model = CRF.train(template, trainRdd)
  val result = model.predict(test.map(Sequence.deSerializer))
```

### Building From Source

```scala
sbt package
```

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * qian.huang@intel.com
 * jiayin.hu@intel.com
 * rui.sun@intel.com
 * hao.cheng@intel.com
