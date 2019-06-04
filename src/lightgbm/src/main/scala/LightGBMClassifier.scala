// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}

import scala.reflect.runtime.universe.{TypeTag, typeTag}

object LightGBMClassifier extends DefaultParamsReadable[LightGBMClassifier]

/** Trains a LightGBM Classification model, a fast, distributed, high performance gradient boosting
  * framework based on decision tree algorithms.
  * For more information please see here: https://github.com/Microsoft/LightGBM.
  * For parameter information see here: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
  * @param uid The unique ID.
  */
@InternalWrapper
class LightGBMClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, LightGBMClassifier, LightGBMClassificationModel]
  with LightGBMBase[LightGBMClassificationModel] {
  def this() = this(Identifiable.randomUID("LightGBMClassifier"))

  // Set default objective to be binary classification
  setDefault(objective -> LightGBMConstants.binaryObjective)

  val isUnbalance = new BooleanParam(this, "isUnbalance",
    "Set to true if training data is unbalanced in binary classification scenario")
  setDefault(isUnbalance -> false)

  def getIsUnbalance: Boolean = $(isUnbalance)
  def setIsUnbalance(value: Boolean): this.type = set(isUnbalance, value)

  def getTrainParams(numWorkers: Int, categoricalIndexes: Array[Int], dataset: Dataset[_]): TrainParams = {
    /* The native code for getting numClasses is always 1 unless it is multiclass-classification problem
     * so we infer the actual numClasses from the dataset here
     */
    val actualNumClasses = getNumClasses(dataset)
    val metric =
      if (getObjective == LightGBMConstants.binaryObjective) "binary_logloss,auc"
      else "multiclass"
    val modelStr = if (getModelString == null || getModelString.isEmpty) None else get(modelString)
    ClassifierTrainParams(getParallelism, getNumIterations, getLearningRate, getNumLeaves,
      getMaxBin, getBaggingFraction, getBaggingFreq, getBaggingSeed, getEarlyStoppingRound,
      getFeatureFraction, getMaxDepth, getMinSumHessianInLeaf, numWorkers, getObjective, modelStr,
      getIsUnbalance, getVerbosity, categoricalIndexes, actualNumClasses, metric, getBoostFromAverage,
      getBoostingType, getLambdaL1, getLambdaL2, getIsProvideTrainingMetric)
  }

  def getModel(trainParams: TrainParams, lightGBMBooster: LightGBMBooster): LightGBMClassificationModel = {
    val classifierTrainParams = trainParams.asInstanceOf[ClassifierTrainParams]
    new LightGBMClassificationModel(uid, lightGBMBooster, getLabelCol, getFeaturesCol,
      getPredictionCol, getProbabilityCol, getRawPredictionCol,
      if (isDefined(thresholds)) Some(getThresholds) else None, classifierTrainParams.numClass)
  }

  def stringFromTrainedModel(model: LightGBMClassificationModel): String = {
    model.getModel.model
  }

  override def copy(extra: ParamMap): LightGBMClassifier = defaultCopy(extra)
}

/** Model produced by [[LightGBMClassifier]]. */
@InternalWrapper
class LightGBMClassificationModel(
  override val uid: String, model: LightGBMBooster, labelColName: String,
  featuresColName: String, predictionColName: String, probColName: String,
  rawPredictionColName: String, thresholdValues: Option[Array[Double]],
  actualNumClasses: Int)
    extends ProbabilisticClassificationModel[Vector, LightGBMClassificationModel]
    with ConstructorWritable[LightGBMClassificationModel] {

  // Update the underlying Spark ML params
  // (for proper serialization to work we put them on constructor instead of using copy as in Spark ML)
  set(labelCol, labelColName)
  set(featuresCol, featuresColName)
  set(predictionCol, predictionColName)
  set(probabilityCol, probColName)
  set(rawPredictionCol, rawPredictionColName)
  if (thresholdValues.isDefined) set(thresholds, thresholdValues.get)

  /**
    * Implementation based on ProbabilisticClassifier with slight modifications to
    * avoid calling raw2probabilityInPlace to defer the probability calculation
    * to lightgbm native code.
    *
    * @param dataset input dataset
    * @return transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    // Output selected columns only.
    var outputData = dataset
    var numColsOutput = 0
    if ($(rawPredictionCol).nonEmpty) {
      val predictRawUDF = udf { (features: Any) =>
        predictRaw(features.asInstanceOf[Vector])
      }
      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
      numColsOutput += 1
    }
    if ($(probabilityCol).nonEmpty) {
      val probabilityUDF = udf { (features: Any) =>
        predictProbability(features.asInstanceOf[Vector])
      }
      val probUDF = probabilityUDF(col($(featuresCol)))
      outputData = outputData.withColumn($(probabilityCol), probUDF)
      numColsOutput += 1
    }
    if ($(predictionCol).nonEmpty) {
      val predUDF = if ($(rawPredictionCol).nonEmpty && !isDefined(thresholds)) {
        // Note: Only call raw2prediction if thresholds not defined
        udf(raw2prediction _).apply(col($(rawPredictionCol)))
      } else if ($(probabilityCol).nonEmpty) {
        udf(probability2prediction _).apply(col($(probabilityCol)))
      } else {
        val predictUDF = udf { (features: Any) =>
          predict(features.asInstanceOf[Vector])
        }
        predictUDF(col($(featuresCol)))
      }
      outputData = outputData.withColumn($(predictionCol), predUDF)
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: LightGBMClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    throw new NotImplementedError("Unexpected error in LightGBMClassificationModel:" +
      " raw2probabilityInPlace should not be called!")
  }

  override def numClasses: Int = this.actualNumClasses

  override protected def predictRaw(features: Vector): Vector = {
    Vectors.dense(model.score(features, true, true))
  }

  override protected def predictProbability(features: Vector): Vector = {
    Vectors.dense(model.score(features, false, true))
  }

  override def copy(extra: ParamMap): LightGBMClassificationModel =
    new LightGBMClassificationModel(uid, model, labelColName, featuresColName, predictionColName, probColName,
      rawPredictionColName, thresholdValues, actualNumClasses)

  override val ttag: TypeTag[LightGBMClassificationModel] =
    typeTag[LightGBMClassificationModel]

  override def objectsToSave: List[Any] =
    List(uid, model, getLabelCol, getFeaturesCol, getPredictionCol,
         getProbabilityCol, getRawPredictionCol, thresholdValues, actualNumClasses)

  def saveNativeModel(filename: String, overwrite: Boolean): Unit = {
    val session = SparkSession.builder().getOrCreate()
    model.saveNativeModel(session, filename, overwrite)
  }

  def getFeatureImportances(importanceType: String): Array[Double] = {
    model.getFeatureImportances(importanceType)
  }

  def getModel: LightGBMBooster = this.model
}

object LightGBMClassificationModel extends ConstructorReadable[LightGBMClassificationModel] {
  def loadNativeModelFromFile(filename: String, labelColName: String = "label",
                              featuresColName: String = "features", predictionColName: String = "prediction",
                              probColName: String = "probability",
                              rawPredictionColName: String = "rawPrediction"): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassifier")
    val session = SparkSession.builder().getOrCreate()
    val textRdd = session.read.text(filename)
    val text = textRdd.collect().map { row => row.getString(0) }.mkString("\n")
    val lightGBMBooster = new LightGBMBooster(text)
    val actualNumClasses = lightGBMBooster.getNumClasses()
    new LightGBMClassificationModel(uid, lightGBMBooster, labelColName, featuresColName,
      predictionColName, probColName, rawPredictionColName, None, actualNumClasses)
  }

  def loadNativeModelFromString(model: String, labelColName: String = "label",
                                featuresColName: String = "features", predictionColName: String = "prediction",
                                probColName: String = "probability",
                                rawPredictionColName: String = "rawPrediction"): LightGBMClassificationModel = {
    val uid = Identifiable.randomUID("LightGBMClassifier")
    val lightGBMBooster = new LightGBMBooster(model)
    val actualNumClasses = lightGBMBooster.getNumClasses()
    new LightGBMClassificationModel(uid, lightGBMBooster, labelColName, featuresColName,
      predictionColName, probColName, rawPredictionColName, None, actualNumClasses)
  }
}
