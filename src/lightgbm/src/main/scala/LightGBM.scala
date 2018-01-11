// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.ml.lightgbm._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml._
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType

import scala.reflect.runtime.universe.{TypeTag, typeTag}

/** Trains a LightGBM model.
  */
class LightGBM(override val uid: String) extends Estimator[LightGBMModel]
  with HasLabelCol with MMLParams {

  def this() = this(Identifiable.randomUID("LightGBM"))

  val featuresColumn = this.uid + "_features"

  /** Fits the LightGBM model.
    *
    * @param dataset The input dataset to train.
    * @return The trained model.
    */
  override def fit(dataset: Dataset[_]): LightGBMModel = {
    val nodesKeys = dataset.sparkSession.sparkContext.getExecutorMemoryStatus.keys
    val nodes = nodesKeys.mkString(",")
    val numNodes = nodesKeys.count((node: String) => true)
    val defaultLocalListenPort = 12400
    val defaultListenTimeout = 120
    // Featurize the input dataset
    val featurizer = new Featurize()
      .setFeatureColumns(Map(featuresColumn -> featureColumns))
      .setOneHotEncodeCategoricals(oneHotEncodeCategoricals)
      .setNumberOfFeatures(featuresToHashTo)
    // Run a parallel job via map partitions
    // Initialize the network communication
    lightgbmlib.LGBM_NetworkInit(nodes, defaultListenTimeout, defaultLocalListenPort, numNodes)
    // Convert the numeric rows from spark format to LightGBM format
    // Train the model
    // Merge the models
    new LightGBMModel(uid)
  }

  override def copy(extra: ParamMap): Estimator[LightGBMModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

object LightGBM extends DefaultParamsReadable[LightGBM]

/** Model produced by [[LightGBM]]. */
class LightGBMModel(val uid: String)
    extends Model[LightGBMModel] with ConstructorWritable[LightGBMModel] {

  val ttag: TypeTag[LightGBMModel] = typeTag[LightGBMModel]
  val objectsToSave: List[AnyRef] = List(uid)

  override def copy(extra: ParamMap): LightGBMModel =
    new LightGBMModel(uid)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.toDF()
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

object LightGBMModel extends ConstructorReadable[LightGBMModel]
