/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The neural processor used to train the [CharLM].
 *
 * @param model the model
 * @param useDropout whether to apply the dropout during the forward
 */
internal class Processor(
  private val model: CharLM,
  override val useDropout: Boolean
) : NeuralProcessor<
  String, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  NeuralProcessor.NoInputErrors // InputErrorsType
  > {

  /**
   * Whether to propagate the errors to the input during the [backward] (not supported)
   */
  override val propagateToInput: Boolean = false

  /**
   * The id for the pool (not supported).
   */
  override val id: Int = 0

  /**
   * The recurrent processor.
   */
  private val recurrentProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    model = this.model.recurrentNetwork,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The feed-forward processor.
   */
  private val classifierProcessor = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model.classifier,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * List of embeddings related to the last forward.
   */
  private var lastCharsEmbeddings = listOf<ParamsArray>()

  /**
   * List of embeddings errors related to the last backward.
   */
  private var lastCharsEmbeddingsErrors = ParamsErrorsAccumulator()

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: String): List<DenseNDArray> {

    this.lastCharsEmbeddings = input.map { c -> this.model.charsEmbeddings[c] }

    return this.classifierProcessor.forward(
      this.recurrentProcessor.forward(this.lastCharsEmbeddings.map { it.values })) // TODO: copy?
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.classifierProcessor.backward(outputErrors)
    this.recurrentProcessor.backward(this.classifierProcessor.getInputErrors(copy = false))
    this.backwardEmbeddings(this.recurrentProcessor.getInputErrors(copy = false))
  }

  /**
   * Propagate the given [errors] to the [lastCharsEmbeddings].
   *
   * @param errors the embeddings errors
   */
  private fun backwardEmbeddings(errors: List<DenseNDArray>) {

    this.lastCharsEmbeddingsErrors.clear()

    this.lastCharsEmbeddings.zip(errors).forEach { (charEmbedding, charErrors) ->

      this.lastCharsEmbeddingsErrors.accumulate(charEmbedding, charErrors)
    }

    this.lastCharsEmbeddingsErrors.averageErrors()
  }

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean) =
    this.recurrentProcessor.getParamsErrors(copy = copy) +
      this.classifierProcessor.getParamsErrors(copy = copy) +
      this.lastCharsEmbeddingsErrors.getParamsErrors(copy = copy)
}
