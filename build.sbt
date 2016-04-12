name := "fuzzyCMeans"

version := "1.6.1"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1"
  , "org.apache.spark" %% "spark-mllib" % "1.6.1"
  , "org.scalactic" %% "scalactic" % "2.2.6"
  , "org.scalatest" %% "scalatest" % "2.2.6" % "test")

// enable publishing the jar produced by `test:package`
publishArtifact in(Test, packageBin) := true