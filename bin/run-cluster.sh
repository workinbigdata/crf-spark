/home/qhuang/works-2.0/sync/spark/spark-1.4.0/bin/spark-submit \
    --class com.intel.nlp.ConditionalRandomFieldExample \
    --master spark://sr464:7077 \
    target/scala-2.10/crf-spark-assembly-0.0.2.jar \
    data/template \
    data/test.data.spark \
    data/train.data.spark \
    96