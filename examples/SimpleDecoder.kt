/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
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
 * A simple decoder that uses a CharLM language model to generates a sequence of characters starting given the first one.
 *
 * @param model the model
 */
class SimpleDecoder(private val model: CharLM) {

  /**
   * The recurrent processor used to process a character at time.
   */
  private val recurrentProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    neuralNetwork = model.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The classifier that predict the next character of the sequence.
   */
  private val classifierProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = model.classifier,
    useDropout = false,
    propagateToInput = false)

  /**
   * @param firstChar the first character of the sequence
   * @param maxSentenceLength the max number of character of the output sequence
   *
   * @return the predicted sequence
   */
  fun decode(firstChar: Char, maxSentenceLength: Int): String {

    val sentence = StringBuffer()
    var curChar = firstChar

    loop@ for (i in 0 until maxSentenceLength) {

      sentence.append(curChar)

      val bestId = this.forward(curChar, firstState = i == 0).argMaxIndex()

      if (bestId == model.eosId)
        break@loop // end-of-sentence
      else
        curChar = model.getChar(bestId)
    }

    return sentence.toString()
  }

  /**
   * @param c the input character
   * @param firstState whether this is the first prediction to build a new sequence
   *
   * @return the distribution of possible next character
   */
  private fun forward(c: Char, firstState: Boolean): DenseNDArray =
    this.classifierProcessor.forward(
      this.recurrentProcessor.forward(
        this.model.charsEmbeddings.get(c).array.values, firstState))
}