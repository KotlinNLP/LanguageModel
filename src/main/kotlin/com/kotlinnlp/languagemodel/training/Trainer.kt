/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralprocessor.ChainProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor.EmbeddingsProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.safeLog
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
 * @param sentences the training sentences
 * @param epochs number of training epochs
 * @param updateMethod the update method (e.g. ADAM, AdaGrad, LearningRate
 * @param verbose whether to display info during the training
 */
class Trainer(
  private val model: CharLM,
  private val modelFilename: String,
  private val sentences: Sequence<String>,
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
  private val processor = buildProcessor()

  /**
   * Used to update the [CharLM] parameters based on the backward errors.
   */
  private val optimizer = ParamsOptimizer(updateMethod = this.updateMethod)

  /**
   * The perplexity calculated during the training.
   */
  private val avgPerplexity = MovingAverage()

  /**
   * The lowest perplexity calculated during the training.
   */
  private var bestPerplexity: Double? = null

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

    this.sentences.forEachIndexed { i, sentence ->

      this.newBatch() // TODO: what is a batch here?
      this.newExample()

      val perplexity = this.trainSentence(
        if (this.model.reverseModel)
          sentence.reversed()
        else
          sentence
      )

      this.avgPerplexity.add(perplexity)

      this.optimizer.update() // optimize for each sentence

      if (i > 0 && i % 100 == 0) {

        print("\nAfter %d examples: perplexity mean = %.2f, variance = %.2f"
          .format(i, this.avgPerplexity.mean, this.avgPerplexity.variance))

        if (this.bestPerplexity != null)
          println(" (former best = %.2f)".format(this.bestPerplexity))
        else
          println()

        this.evaluateAndSaveModel()

      } else if (i % 10 == 0) {
        print(".")
      }
    }
  }

  /**
   * Evaluate and save the best model.
   */
  private fun evaluateAndSaveModel() {

    if (this.bestPerplexity == null || this.bestPerplexity!! > this.avgPerplexity.mean) {

      this.bestPerplexity = this.avgPerplexity.mean

      println("[NEW BEST PERPLEXITY!] Saving model...")

      this.model.setAvgPerplexity(this.bestPerplexity!!)
      this.model.dump(FileOutputStream(File(this.modelFilename)))
    }
  }

  /**
   * Train a single sentence.
   *
   * @param sentence the sentence
   *
   * @return the avg perplexity of the given [sentence]
   */
  private fun trainSentence(sentence: String): Double {

    if (sentence.contains(CharLM.ETX) || sentence.contains(CharLM.UNK)) {
      throw RuntimeException("The String can't contain NULL or ETX chars")
    }

    val prediction = this.processor.forward(sentence.toList())

    // The target is always the next character.
    val targets = (0 until sentence.length)
      .map { i -> if (i == sentence.lastIndex) this.model.etxCharId else this.model.getCharId(sentence[i + 1]) }
      .map { charId -> DenseNDArrayFactory.oneHotEncoder(length = this.model.classifier.outputSize, oneAt = charId) }

    val errors = SoftmaxCrossEntropyCalculator().calculateErrors(
      outputSequence = prediction,
      outputGoldSequence = targets)

    this.processor.backward(errors)
    this.optimizer.accumulate(this.processor.getParamsErrors(copy = false))

    val loss = prediction.zip(targets).map { (y, g) -> -safeLog(y[g.argMaxIndex()]) }.average()

    return exp(loss) // perplexity
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

  /**
   * @return the processor to train the CharLM model
   */
  private fun buildProcessor() = ChainProcessor(
    inputProcessor = EmbeddingsProcessor(
      embeddingsMap = this.model.charsEmbeddings,
      useDropout = true),
    hiddenProcessors = listOf(
      RecurrentNeuralProcessor(
        model = this.model.recurrentNetwork,
        useDropout = true,
        propagateToInput = true)),
    outputProcessor = BatchFeedforwardProcessor(
      model = this.model.classifier,
      useDropout = true,
      propagateToInput = true))
}
