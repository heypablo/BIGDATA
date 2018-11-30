//Well this is the final Proyect in BIGDATA
//Lets get over it
//NOW the import, we need all the librerys
//Is very important this part
  import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.sql.SparkSession
  import org.apache.log4j._
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.classification.LinearSVC
  import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  import org.apache.log4j._
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.classification.DecisionTreeClassificationModel
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
  import org.apache.spark.ml.classification.LogisticRegression
  Logger.getLogger("org").setLevel(Level.ERROR)
  //Now we create the seccion of spark to LOAD are DT
  //this part is triky
  val spark = SparkSession.builder().getOrCreate()
  //For example this CSV has alittle and big issue
  //if you load the csv without the .option("delimiter")all the dt will be a mess, you have to put that to proce ...
  val DataF = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
  //Here, this cloumn webe are label, i can do in another way right, but for the moment this esenario is perfect
  //when the comand find a yes it will change for 1 and the same case in No it changes for 2
  val c1 = DataF.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
  val c2 = c1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
  //And now the column is still working like a string we need to changes that
  val c3 = c2.withColumn("y",'y.cast("Int"))
  //Using the VectorAssembler is more easy to create the features, only you have to do is select all the cloumns you need
  val assemFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
  val Limdf = assemFeatures.transform(c3)
  Limdf.show(1)
  //Oh in hre we change the name of the column Y
  val change = Limdf.withColumnRenamed("y", "label")
  val ft = change.select("label","features")
  //and this our DF
  ft.show()

//And now you have to use that DT in al the methods of the Proyect

//Multilayer
//DONE
// Split the data into train and test
val splits = ft.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](5, 2, 2, 4)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
--------------
//Regresion Logistica
//state Done
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit the model
val lrModel = lr.fit(ft)
// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
// We can also use the multinomial family for binary classification
val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val mlrModel = mlr.fit(ft)
// Print the coefficients and intercepts for logistic regression with multinomial family
println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${mlrModel.interceptVector}")
Threes
DONE
// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(ft)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) // features with > 4 distinct values are treated as continuous.  .fit(data)
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = ft.randomSplit(Array(0.7, 0.3))
// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)
// Make predictions.
val predictions = model.transform(testData)
// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
SVM
DONE
//We have to do this first beacuse the algoritm only accept binary types in the label column so this is the code to do that
val cs1 = ft.withColumn("1abel",when(col("label").equalTo("1"),0).otherwise(col("label")))
val cs2 = cs1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val cs3 = cs2.withColumn("label",'label.cast("Int"))
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// Fit the model
val lsvcModel = lsvc.fit(cs3)
// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
