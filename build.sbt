name := "crf-spark"

version := "0.1.0"

scalaVersion := "2.10.4"

spName := "hqzizania/crf-spark"

sparkVersion := "1.4.0"

sparkComponents += "mllib"

resolvers += Resolver.sonatypeRepo("public")

/********************
  * Release settings *
  ********************/

spShortDescription := "crf-spark"

spDescription := """A Spark-based implementation of Conditional Random Fields (CRFs) for labeling sequential data""".stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := true

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

