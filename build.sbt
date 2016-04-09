name := "fuzzyCMeans"

version := "1.5.2"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.2"
  , "org.apache.spark" %% "spark-mllib" % "1.5.2"
  , "org.scalactic" %% "scalactic" % "2.2.6"
  , "org.scalatest" %% "scalatest" % "2.2.6" % "test")
