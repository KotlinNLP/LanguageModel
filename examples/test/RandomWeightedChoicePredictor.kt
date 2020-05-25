/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package test

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
internal class RandomWeightedChoicePredictor(private val model: CharLM) {

  /**
   * The recurrent hidden processor that encodes a character at time.
   */
  private val hiddenProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(model = this.model.hiddenNetwork, useDropout = false, propagateToInput = false)

  /**
   * The classifier that predicts the next character of the sequence.
   */
  private val classifierProcessor: FeedforwardNeuralProcessor<DenseNDArray> =
    FeedforwardNeuralProcessor(model = this.model.outputClassifier, useDropout = false, propagateToInput = false)

  /**
   * Predict the continuation of a characters sequence, based on the given language model.
   *
   * @param input the input chars sequence
   * @param maxSentenceLength the max number of characters of the output sequence
   *
   * @return the predicted sequence
   */
  fun predict(input: String, maxSentenceLength: Int): String {

    val sentence = StringBuffer(input)
    var nextChar: Char = this.initPrediction(input)

    while (sentence.length < maxSentenceLength && !this.model.isEndOfSentence(nextChar)) {

      sentence.append(nextChar)

      nextChar = this.predictNextChar(nextChar)
    }

    return sentence.toString()
  }

  /**
   * Initialize the prediction processing the whole input sequence and returning the next char predicted.
   *
   * @param input the input chars sequence
   *
   * @return the next char predicted
   */
  private fun initPrediction(input: String): Char {

    val charsEmbeddings: List<DenseNDArray> = input.map { this.model.charsEmbeddings[it].values }
    val charsEncodings: List<DenseNDArray> = this.hiddenProcessor.forward(charsEmbeddings)
    val prediction: DenseNDArray = this.classifierProcessor.forward(charsEncodings.last())

    return this.model.getChar(prediction.argMaxIndex())
  }

  /**
   * Predict the next char of the current sequence.
   *
   * @param lastChar the last char predicted
   *
   * @return the next char predicted
   */
  private fun predictNextChar(lastChar: Char): Char {

    val lastCharEmbedding: DenseNDArray = this.model.charsEmbeddings[lastChar].values
    val prediction: DenseNDArray = this.classifierProcessor.forward(
      this.hiddenProcessor.forward(lastCharEmbedding, firstState = false))

    var prob: Double = Math.random()
    val charIndex: Int = prediction.toDoubleArray().indexOfFirst { x ->
      prob -= x
      prob < 0
    }

    return this.model.getChar(charIndex)
  }
}
