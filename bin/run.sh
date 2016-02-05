/home/qhuang/Documents/spark/apache/spark-1.4.0/bin/spark-submit \
    --class com.intel.nlp.ConditionalRandomFieldExample \
    --master local[4] target/scala-2.10/crf-spark-assembly-0.0.2.jar \
    data/template \
    data/train.data.spark \
    data/test.data.spark \
    4
