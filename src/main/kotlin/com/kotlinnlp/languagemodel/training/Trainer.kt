/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

/**
 * Class responsible for the performance of training process in epochs.
 *
 * @param model the model
 * @param modelFilename where to save the serialized model
 * @param corpusFilePath where to find the corpus for training
 * @param maxSentences maximum number of sentences in the corpus to be used for training
 * @param epochs number of training epochs
 * @param updateMethod the update method (e.g. ADAM, AdaGrad, LearningRate
 * @param verbose whether to display info during the training
 */
class Trainer(
  private val model: CharLM,
  private val modelFilename: String,
  private val corpusFilePath: String,
  private val maxSentences: Int? = null,
  private val epochs: Int,
  private val updateMethod: UpdateMethod<*>,
  private val verbose: Boolean = true
) {

  /**
   * A timer to track the elapsed time.
   */
  private var timer = Timer()

  /**
   * The neural processor to train the [CharLM].
   */
  private val processor = Processor(this.model, useDropout = true)

  /**
   * Used to update the [CharLM] parameters based on the backward errors.
   */
  private val optimizer = Optimizer(this.model, updateMethod = this.updateMethod)

  /**
   * Check requirements.
   */
  init {
    require(this.epochs > 0) { "The number of epochs must be > 0" }
  }

  /**
   * Start the training.
   */
  fun train() {

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch()

      this.logTrainingEnd()
    }
  }

  /**
   * Function responsible for the single-epoch training step.
   *
   * Each epoch trains the model based on the given training sentences.
   * Each time a best value is reached (i.e. higher score), the model trained so far
   * is saved.
   */
  private fun trainEpoch() {

    File(this.corpusFilePath).forEachIndexedSentence(this.maxSentences) { i, sentence ->

      this.newBatch() // TODO: what is a batch here?
      this.newExample()

      if (this.model.reverseModel)
        this.trainSentence(sentence.reversed())
      else
        this.trainSentence(sentence)

      this.optimizer.update()

      if (i > 0 && i % 100 == 0) { // TODO: check best accuracy

        this.model.dump(FileOutputStream(File(this.modelFilename)))

        println("\n[$i] Model saved to \"${this.modelFilename}\"")
      }
    }
  }

  /**
   * Train a single sentence.
   *
   * @param sentence the sentence
   */
  private fun trainSentence(sentence: String) {

    val prediction = this.processor.forward(sentence)

    val expectedOutput: List<DenseNDArray> = this.getExpectedCharsSequence(sentence).map {
      DenseNDArrayFactory.oneHotEncoder(length = this.model.classifier.outputSize, oneAt = it)
    }

    val errors = SoftmaxCrossEntropyCalculator().calculateErrors(
      outputSequence = prediction,
      outputGoldSequence = expectedOutput)

    if (this.verbose) {

      val loss = SoftmaxCrossEntropyCalculator().calculateMeanLoss(
        outputSequence = prediction,
        outputGoldSequence = expectedOutput)

      val perplexity = this.calculatePerplexity(prediction)

      println("Loss: $loss Perplexity: $perplexity") // TODO: print to improve
    }

    this.processor.backward(errors)
    this.optimizer.accumulate(processor.getParamsErrors(copy = false))
  }

  /**
   *
   */
  private fun calculatePerplexity(prediction: List<DenseNDArray>): Double {

    val x = prediction.sumByDouble { Math.log(it.max()) } / this.model.classifierOutputSize
    return exp(-x)
  }

  /**
   * @param s an input string
   *
   * @return the expected output sequence
   */
  private fun getExpectedCharsSequence(s: String): List<Int> = (0 until s.length).map { i ->
    if (i < s.lastIndex) this.model.getCharId(s[i + 1]) else model.eosId
  }

  /**
   * Beat the occurrence of a new example.
   */
  private fun newExample() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   * Beat the occurrence of a new batch.
   */
  private fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  private fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Log when training starts.
   *
   * @param epochIndex the current epoch index
   */
  private fun logTrainingStart(epochIndex: Int) {

    if (this.verbose) {

      this.timer.reset()

      println("\nEpoch ${epochIndex + 1} of ${this.epochs}")
      println("\nStart training...")
    }
  }

  /**
   * Log when training ends.
   */
  private fun logTrainingEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }
}