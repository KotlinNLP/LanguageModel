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
 * @param charsDropout the dropout probability of the chars embeddings (default 0.0)
 * @param hiddenDropout the dropout of the hidden encoder (default 0.0)
 * @param classifierDropout the dropout of the output classifier (default 0.0)
 * @param gradientClipping the gradient clipper
 * @param updateMethod the update method (e.g. ADAM, AdaGrad, LearningRate)
 */
class Trainer(
  private val model: CharLM,
  modelFilename: String,
  sentences: Iterable<String>,
  private val charsBatchesSize: Int,
  charsDropout: Double = 0.0,
  hiddenDropout: Double = 0.0,
  classifierDropout: Double = 0.0,
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
  private inner class CharsBatch(val text: String, val nextChar: Char?, val isSentenceStart: Boolean) {

    /**
     * @return the classification targets for this batch (the next chars classifications)
     */
    fun getClassificationTargets(): List<DenseNDArray> {

      val model: CharLM = this@Trainer.model
      val lastCharId: Int = this.nextChar?.let { model.getCharId(this.nextChar)} ?: model.etxCharId

      return this.text.indices
        .asSequence()
        .map { i -> if (i == this.text.lastIndex) lastCharId else model.getCharId(this.text[i + 1]) }
        .map { charId -> DenseNDArrayFactory.oneHotEncoder(length = model.outputClassifier.outputSize, oneAt = charId) }
        .toList()
    }
  }

  /**
   * The input embeddings processor.
   */
  private val embProcessor: EmbeddingsProcessor<Char> =
    EmbeddingsProcessor(embeddingsMap = this.model.charsEmbeddings, dropout = charsDropout)

  /**
   * The hidden processor to auto-encode the input.
   */
  private val hiddenProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(model = this.model.hiddenNetwork, dropout = hiddenDropout, propagateToInput = true)

  /**
   * The output classifier.
   */
  private val outputClassifier: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.model.outputClassifier, dropout = classifierDropout, propagateToInput = true)

  /**
   * Support to save the initial hidden arrays during the batch learning.
   */
  private lateinit var initHiddens: List<DenseNDArray>

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
    this.backward(predictions = this.forward(batch), targets = batch.getClassificationTargets())
  }

  /**
   * Forward a batch returning the next chars predictions.
   *
   * @param batch a training batch
   *
   * @return the next chars predictions
   */
  private fun forward(batch: CharsBatch): List<DenseNDArray> {

    val charsEmbeddings: List<DenseNDArray> = this.embProcessor.forward(batch.text.toList())

    val charsEncodings: List<DenseNDArray> = this.hiddenProcessor.let {
      it.forward(input = charsEmbeddings, initHiddenArrays = if (batch.isSentenceStart) null else this.initHiddens)
      this.initHiddens = it.getCurState(copy = true)!!
      it.getOutputSequence(copy = false)
    }

    return this.outputClassifier.forward(charsEncodings)
  }

  /**
   * Execute the backward of the processors and accumulate the loss.
   *
   * @param predictions the next chars predictions of the last batch
   * @param targets the given predictions targets
   */
  private fun backward(predictions: List<DenseNDArray>, targets: List<DenseNDArray>) {

    val outputErrors: List<DenseNDArray> =
      SoftmaxCrossEntropyCalculator.calculateErrors(outputSequence = predictions, outputGoldSequence = targets)

    val hiddenErrors: List<DenseNDArray> = this.outputClassifier.let {
      it.backward(outputErrors)
      it.getInputErrors(copy = false)
    }

    val embeddingsErrors: List<DenseNDArray> = this.hiddenProcessor.let {
      it.backward(hiddenErrors)
      it.getInputErrors(copy = false)
    }

    this.embProcessor.backward(embeddingsErrors)

    predictions.zip(targets).forEach { (prediction, target) ->
      this.avgLoss.add(-safeLog(prediction[target.argMaxIndex()]))
    }
  }

  /**
   * Accumulate the errors of the model resulting after the [learnFromBatch].
   */
  private fun accumulateBatchErrors() {

    val paramsErrors: ParamsErrorsList = this.embProcessor.getParamsErrors(copy = false) +
      this.hiddenProcessor.getParamsErrors(copy = false) +
      this.outputClassifier.getParamsErrors(copy = false)

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

      if (str.contains(CharLM.UNK) || str.contains(CharLM.ETX))
        throw RuntimeException("The training sentence cannot contain the UNK or ETX chars")

      var start = 0

      while (start < str.length) {

        val end: Int = (start + maxSize).coerceAtMost(str.length)
        val nextChar: Char? = if (end < str.length) str[end] else null

        yield(CharsBatch(text = str.substring(start, end), nextChar = nextChar, isSentenceStart = start == 0))

        start = end
      }
    }
  }
}
