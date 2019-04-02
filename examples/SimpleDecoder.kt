/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 * A simple decoder that uses a CharLM language model to generates a sequence of characters starting from a first
 * input sequence.
 *
 * @param model the model
 */
class SimpleDecoder(private val model: CharLM) {

  /**
   * The recurrent processor used to process a character at time.
   */
  private val recurrentProcessor = RecurrentNeuralProcessor<SparseBinaryNDArray>(
    model = model.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The classifier that predict the next character of the sequence.
   */
  private val classifierProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = model.classifier,
    useDropout = false,
    propagateToInput = false)

  /**
   * @param input the first characters of the sequence
   * @param maxSentenceLength the max number of character of the output sequence
   *
   * @return the predicted sequence
   */
  fun decode(input: String, maxSentenceLength: Int): String {

    val sentence = StringBuffer(input)

    this.initSequence(input)

    var nextChar: Char = this.classifierProcessor.forward(this.recurrentProcessor.getOutput(copy = false)).let {
      this.model.getChar(it.argMaxIndex(exceptIndex = this.model.etxCharId))
    }

    while (sentence.length <= maxSentenceLength && !this.model.isEndOfSentence(nextChar)) {

      sentence.append(nextChar)

      val distribution = this.forward(nextChar, firstState = false)

      nextChar = this.model.getChar(distribution.toDoubleArray().weightedRandomChoice())
    }

    return sentence.toString()
  }

  /**
   * Perform a random generation of the indices of the array with the probability defined in the array itself.
   *
   * @return an index of the array
   */
  private fun DoubleArray.weightedRandomChoice(): Int {

    var prob = this.reduce { a, b -> a + b } * Math.random()

    return this.indexOfFirst { a ->
      prob -= a
      prob <= 0
    }
  }

  /**
   * Process the [input] with the recurrent processor.
   *
   * @param input the first characters of the sequence
   */
  private fun initSequence(input: String) =
    this.recurrentProcessor.forward(input.map {          SparseBinaryNDArrayFactory.arrayOf(
      shape = Shape(this.model.recurrentNetwork.inputSize),
      activeIndices = listOf(this.model.charsDict.getId(it)!!)) })

  /**
   * @param c the input character
   * @param firstState whether this is the first prediction to build a new sequence
   *
   * @return the distribution of possible next character
   */
  private fun forward(c: Char, firstState: Boolean): DenseNDArray =
    this.classifierProcessor.forward(this.recurrentProcessor.forward(

      input = SparseBinaryNDArrayFactory.arrayOf(
        shape = Shape(this.model.recurrentNetwork.inputSize),
        activeIndices = listOf(this.model.charsDict.getId(c)!!)),

      firstState = firstState))
}
