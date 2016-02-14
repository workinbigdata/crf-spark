name := "crf-spark"

version := "0.0.2"

scalaVersion := "2.11.7"


libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.0"  % "provided"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.6.0"  % "provided"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0" 

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.2"

resolvers += Resolver.sonatypeRepo("public")
