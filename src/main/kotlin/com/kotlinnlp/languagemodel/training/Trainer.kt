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
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor.EmbeddingsProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
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
 * @param batchSize the size of each characters batch
 * @param updateMethod the update method (e.g. ADAM, AdaGrad, LearningRate
 * @param verbose whether to display info during the training
 */
class Trainer(
  private val model: CharLM,
  private val modelFilename: String,
  private val sentences: Sequence<String>,
  private val epochs: Int,
  private val batchSize: Int,
  private val updateMethod: UpdateMethod<*>,
  private val verbose: Boolean = true
) {

  /**
   * A timer to track the elapsed time.
   */
  private var timer = Timer()

  /**
   * The processor of the input network.
   */
  private val inputProcessor: EmbeddingsProcessor<Char> = EmbeddingsProcessor(
    embeddingsMap = this.model.charsEmbeddings,
    useDropout = true)

  /**
   * The processors of the hidden networks.
   */
  private val hiddenProcessors: List<RecurrentNeuralProcessor<DenseNDArray>> = listOf(
    RecurrentNeuralProcessor(
      model = this.model.recurrentNetwork,
      useDropout = true,
      propagateToInput = true))

  /**
   * The processor of the output network.
   */
  private val outputProcessor: BatchFeedforwardProcessor<DenseNDArray> = BatchFeedforwardProcessor(
    model = this.model.classifier,
    useDropout = true,
    propagateToInput = true)

  /**
   * Used to update the [CharLM] parameters based on the backward errors.
   */
  private val optimizer = ParamsOptimizer(updateMethod = this.updateMethod)

  /**
   *
   */
  private lateinit var initHiddens: List<List<DenseNDArray>>

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

    this.sentences
      .filter { it.isNotEmpty() }
      .forEachIndexed { i, sentence ->

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

      this.model.avgPerplexity = this.bestPerplexity!!
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

    var loss = 0.0
    var start = 0

    while (start < sentence.length) {

      val end: Int = Math.min(start + this.batchSize, sentence.length)
      val nextChar: Char? = if (end < sentence.length) sentence[end] else null

      this.newBatch()
      this.newExample()

      loss += this.trainBatch(batch = sentence.substring(start, end), nextChar = nextChar, isFirst = start == 0)

      start = end
    }

    return exp(loss / sentence.length) // perplexity
  }

  /**
   * Train a batch of characters.
   *
   * @param batch a characters batch
   * @param nextChar the char that follows the batch or null if the batch is the last of the sentence
   * @param isFirst whether it is the first batch
   *
   * @return the negative logarithmic loss accumulated during the training of the given [batch]
   */
  private fun trainBatch(batch: String, nextChar: Char?, isFirst: Boolean): Double {

    val predictions: List<DenseNDArray> = this.outputProcessor.forward(
      this.hiddenProcessors.forward(
        input = this.inputProcessor.forward(batch.toList()),
        initHiddens = if (isFirst) null else this.initHiddens))

    this.initHiddens = this.hiddenProcessors.map { it.getCurState(copy = true)!! }

    // The target is always the next character.
    val lastCharId: Int = nextChar?.let { this.model.getCharId(nextChar)} ?: this.model.etxCharId
    val targets: List<DenseNDArray> = (0 until batch.length).map { i ->
      DenseNDArrayFactory.oneHotEncoder(
        length = this.model.classifier.outputSize,
        oneAt = if (i == batch.lastIndex) lastCharId else this.model.getCharId(batch[i + 1]))
    }

    val errors: List<DenseNDArray> = SoftmaxCrossEntropyCalculator().calculateErrors(
      outputSequence = predictions,
      outputGoldSequence = targets)

    this.inputProcessor.backward(
      this.hiddenProcessors.backwardAndGetInputErrors(
        this.outputProcessor.let {
          it.backward(errors)
          it.getInputErrors(copy = false)
        }
      )
    )

    val paramsErrors: ParamsErrorsList = this.inputProcessor.getParamsErrors(copy = false) +
      this.hiddenProcessors.flatMap { it.getParamsErrors(copy = false) } +
      this.outputProcessor.getParamsErrors(copy = false)

    this.optimizer.accumulate(paramsErrors)

    return predictions.zip(targets).map { (prediction, target) -> -safeLog(prediction[target.argMaxIndex()]) }.sum()
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
   * Perform the forward of the hidden processors.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  private fun List<RecurrentNeuralProcessor<DenseNDArray>>.forward(
    input: List<DenseNDArray>,
    initHiddens: List<List<DenseNDArray>?>?
  ): List<DenseNDArray> {

    var curInput = input

    this.forEachIndexed { i, processor ->
      processor.forward(input = curInput, initHiddenArrays = initHiddens?.get(i))
      curInput = processor.getOutputSequence(copy = false)
    }

    return curInput.map { it.copy() }
  }

  /**
   * Perform the backward of the hidden processors returning the input errors.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors of the first hidden processor
   */
  private fun List<RecurrentNeuralProcessor<DenseNDArray>>.backwardAndGetInputErrors(
    outputErrors: List<DenseNDArray>
  ): List<DenseNDArray> {

    var errors = outputErrors

    this.asReversed().forEach { proc ->

      errors = proc.let {
        it.backward(errors)
        it.getInputErrors(copy = false)
      }
    }

    return errors
  }
}
