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
  val template = Array("U00:%x[-1,0]", "U01:%x[0,0]", "U00:%x[1,0]", "B")
  val train = "B-NP|--|Friday|-|NNP\tB-NP|--|'s|-|POS\tI-NP|--|Market|-|NNP\tI-NP|--|Activity|-|NN"
  val test = "B-NP|--|It|-|PRP\tB-VP|--|was|-|VBD\tB-NP|--|a|-|DT\tI-NP|--|Friday|-|NNP\tB-PP|--|in|-|IN\tB-NP|--|June|-|NNP\tO|--|.|-|."

  val trainRdd = sc.parallelize(Sequence.deSerializer(train))

  val model = CRF.train(template, trainRdd)
  val result = model.predict(Sequence.deSerializer(test))

```