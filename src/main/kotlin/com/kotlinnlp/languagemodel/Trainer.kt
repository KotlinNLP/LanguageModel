/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel

import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor.EmbeddingsProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Trainer as TrainingHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.safeLog
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.stats.MovingAverage
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

/**
 * The trainer of a [CharLM].
 *
 * @param model the model
 * @param modelFilename where to save the serialized model
 * @param sentences the training sentences
 * @param charsBatchesSize the max size of each batch
 * @param charsDropout the chars embeddings dropout
 * @param gradientClipping the gradient clipper
 * @param updateMethod the update method (e.g. ADAM, AdaGrad, LearningRate)
 */
class Trainer(
  private val model: CharLM,
  modelFilename: String,
  sentences: Iterable<String>,
  private val charsBatchesSize: Int,
  private val charsDropout: Double,
  gradientClipping: GradientClipping?,
  updateMethod: UpdateMethod<*>
) : TrainingHelper<String>(
  modelFilename = modelFilename,
  optimizers = listOf(ParamsOptimizer(updateMethod = updateMethod, gradientClipping = gradientClipping)),
  examples = sentences,
  epochs = 1,
  batchSize = 1,
  evaluator = null,
  shuffler = null,
  verbose = false
) {

  /**
   * A batch of characters representing a training example.
   *
   * @param text the text of the batch
   * @param nextChar the char that follows the batch or `null` if this batch is the last of the sentence
   * @param isSentenceStart `true` if this is the first batch of a sentence, otherwise `false`
   */
  data class CharsBatch(val text: String, val nextChar: Char?, val isSentenceStart: Boolean)

  /**
   * The processor of the input network.
   */
  private val inputProcessor: EmbeddingsProcessor<Char> = EmbeddingsProcessor(
    embeddingsMap = this.model.charsEmbeddings,
    dropout = this.charsDropout)

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
   * Support to save the initial hidden arrays during the batch learning.
   */
  private lateinit var initHiddens: List<List<DenseNDArray>>

  /**
   * The loss accumulated during the training.
   */
  private val avgLoss = MovingAverage(200)

  /**
   * The lowest loss calculated during the training.
   */
  private var bestLossMean: Double? = null

  /**
   * The number of sentences seen at a given time.
   */
  private var sentencesCount = 0

  /**
   * A timer to track the elapsed time.
   */
  private val timer = Timer()

  /**
   * Function disabled, an explicit call to [accumulateBatchErrors] is made instead.
   */
  override fun accumulateErrors() = Unit

  /**
   * Function disabled, an explicit call to [evaluateAndSaveModel] is made instead.
   */
  override fun dumpModel() = Unit

  /**
   * Learn from a single sentence (forward + backward).
   *
   * @param example a training sentence
   */
  override fun learnFromExample(example: String) {

    this.sentencesCount++

    example.toBatches().forEach {
      this.learnFromBatch(it)
      this.accumulateBatchErrors()
    }

    if (this.sentencesCount % 10 == 0) print(".")

    if (this.sentencesCount % 100 == 0) {
      this.printProgress()
      this.evaluateAndSaveModel() // TODO: model saved before the last update
    }
  }

  /**
   * Learn from a single chars batch (forward + backward).
   *
   * @param batch a chars batch
   */
  private fun learnFromBatch(batch: CharsBatch) {

    val predictions: List<DenseNDArray> = this.outputProcessor.forward(
      this.hiddenProcessors.forward(
        input = this.inputProcessor.forward(batch.text.toList()),
        initHiddens = if (batch.isSentenceStart) null else this.initHiddens))

    this.initHiddens = this.hiddenProcessors.map { it.getCurState(copy = true)!! }

    // The target is always the next character.
    val lastCharId: Int = batch.nextChar?.let { this.model.getCharId(batch.nextChar)} ?: this.model.etxCharId
    val targets: List<DenseNDArray> = batch.text.indices.map { i ->
      DenseNDArrayFactory.oneHotEncoder(
        length = this.model.classifier.outputSize,
        oneAt = if (i == batch.text.lastIndex) lastCharId else this.model.getCharId(batch.text[i + 1]))
    }

    val errors: List<DenseNDArray> =
      SoftmaxCrossEntropyCalculator.calculateErrors(outputSequence = predictions, outputGoldSequence = targets)

    this.inputProcessor.backward(
      this.hiddenProcessors.backwardAndGetInputErrors(
        this.outputProcessor.let {
          it.backward(errors)
          it.getInputErrors(copy = false)
        }
      )
    )

    predictions.zip(targets).forEach { (prediction, target) ->
      this.avgLoss.add(-safeLog(prediction[target.argMaxIndex()]))
    }
  }

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromBatch].
   */
  private fun accumulateBatchErrors() {

    val paramsErrors: ParamsErrorsList = this.inputProcessor.getParamsErrors(copy = false) +
      this.hiddenProcessors.flatMap { it.getParamsErrors(copy = false) } +
      this.outputProcessor.getParamsErrors(copy = false)

    this.optimizers.first().accumulate(paramsErrors)
  }

  /**
   * Evaluate the model and save it if it is the best.
   */
  private fun evaluateAndSaveModel() {

    val lossMean: Double = this.avgLoss.calcMean()

    if (this.bestLossMean == null || lossMean < this.bestLossMean!!) {

      this.bestLossMean = lossMean

      println("[NEW BEST PERPLEXITY!] Saving the model to '$modelFilename'...")

      this.model.avgPerplexity = exp(this.bestLossMean!!)
      this.model.dump(FileOutputStream(File(this.modelFilename)))
    }
  }

  /**
   * Print the current progress.
   */
  private fun printProgress() {

    print("\n[${this.timer.formatElapsedTime()}] After ${this.sentencesCount} sentences: " +
      "loss mean = %.2f, std dev = %.2f".format(this.avgLoss.calcMean(), this.avgLoss.calcStdDev()))

    if (this.bestLossMean != null)
      println(" (former best = %.2f)".format(this.bestLossMean))
    else
      println()
  }

  /**
   * Convert this string to a sequence of [Trainer.CharsBatch], splitting it in batches with a max size.
   *
   * @return a sequence of training bathes
   */
  private fun String.toBatches(): Sequence<CharsBatch> = sequence {

    val str: String = this@toBatches
    val maxSize: Int = this@Trainer.charsBatchesSize

    if (str.isNotBlank()) {

      if (str.contains(CharLM.ETX) || str.contains(CharLM.UNK))
        throw RuntimeException("The training sentence cannot contain the NULL or ETX chars")

      var start = 0

      while (start < str.length) {

        val end: Int = (start + maxSize).coerceAtMost(str.length)
        val nextChar: Char? = if (end < str.length) str[end] else null

        yield(CharsBatch(text = str.substring(start, end), nextChar = nextChar, isSentenceStart = start == 0))

        start = end
      }
    }
  }


  /**
   * Execute the forward of the hidden processors.
   *
   * @param input the input
   * @param initHiddens the initial hidden arrays
   *
   * @return the output arrays
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
