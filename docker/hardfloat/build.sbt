// The HardFloat generator (docs/toolchain-image.md): Berkeley HardFloat's Chisel sources from the pinned
// submodule plus our `Generate` main, against the Chisel release the image pins (CHISEL_VERSION, from
// config.toml through docker/build.sh). Built only in the image's `hardfloat` stage.
val chiselVersion = sys.env.getOrElse("CHISEL_VERSION", "")
val scalaV = sys.env.getOrElse("SCALA_VERSION", "")

lazy val root = (project in file("."))
  .settings(
    name := "hardfloat-gen",
    scalaVersion := scalaV,
    Compile / unmanagedSourceDirectories += baseDirectory.value / "hardfloat-src",
    libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion,
    addCompilerPlugin(("org.chipsalliance" % "chisel-plugin" % chiselVersion).cross(CrossVersion.full)),
    scalacOptions ++= Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations"),
  )
