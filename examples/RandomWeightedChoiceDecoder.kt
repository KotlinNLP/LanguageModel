/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Generate texts from a starting sequence of chars, choosing the following chars with a random weighted choice based
 * on the prediction of a characters based language model.
 *
 * @param model a char language model
 */
class RandomWeightedChoiceDecoder(private val model: CharLM) {

  /**
   * The recurrent processor used to process a character at time.
   */
  private val recurrentProcessor = RecurrentNeuralProcessor<DenseNDArray>(
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
    var nextChar: Char = this.initSequence(input)

    while (sentence.length < maxSentenceLength && !this.model.isEndOfSentence(nextChar)) {

      sentence.append(nextChar)

      nextChar = this.predictNextChar(nextChar)
    }

    return sentence.toString()
  }

  /**
   * Process the [input] with the recurrent processor.
   *
   * @param input the first characters of the sequence
   *
   * @return the next char predicted
   */
  private fun initSequence(input: String): Char {

    val charsEmbeddings: List<DenseNDArray> = input.map { this.model.charsEmbeddings[it].values }
    val prediction: DenseNDArray = this.classifierProcessor.forward(
      this.recurrentProcessor.forward(charsEmbeddings).last())

    return this.model.getChar(prediction.argMaxIndex())
  }

  /**
   * @param lastChar the last char predicted
   *
   * @return the next char predicted
   */
  private fun predictNextChar(lastChar: Char): Char {

    val lastCharEmbedding: DenseNDArray = this.model.charsEmbeddings[lastChar].values
    val prediction: DenseNDArray = this.classifierProcessor.forward(
      this.recurrentProcessor.forward(lastCharEmbedding, firstState = false))

    var prob: Double = Math.random()
    val charIndex: Int = prediction.toDoubleArray().indexOfFirst { x ->
      prob -= x
      prob < 0
    }

    return this.model.getChar(charIndex)
  }
}
