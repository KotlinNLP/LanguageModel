/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.neuralprocessor.ChainProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.embeddingsprocessor.EmbeddingsProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.safeLog
import java.io.File
import java.io.FileInputStream
import kotlin.math.exp

/**
 * Get the perplexity of a sentence using the language model.
 * The perplexity is calculated as `exp(loss)`.
 *
 * One argument is required: the model filename.
 */
fun main(args: Array<String>) {

  val model = CharLM.load(FileInputStream(File(args[0])))
  val processor = buildProcessor(model)

  while (true) {

    val inputText: String = (readValue() ?: break).let { if (it.length == 1) " $it " else it }
    val chars: List<Char> = inputText.asSequence().take(inputText.length - 1).toList()
    val predictions: List<DenseNDArray> = processor.forward(chars)

    // The target is always the next character.
    val targets: List<DenseNDArray> = (1 until inputText.length).map { i ->
      DenseNDArrayFactory.oneHotEncoder(
        length = model.outputClassifier.outputSize,
        oneAt = model.getCharId(inputText[i]))
    }

    val loss: Double = predictions.zip(targets).map { (y, g) -> -safeLog(y[g.argMaxIndex()]) }.average()
    val perplexity: Double = exp(loss)

    println("Perplexity: $perplexity")
  }

  println("\nExiting...")
}

/**
 * Read a value from the standard input.
 *
 * @return the string read or `null` if the input was blank
 */
private fun readValue(): String? {

  print("\nType the beginning of the sequence. Even a single character (empty to exit): ")

  return readLine()!!.ifBlank { null }
}

/**
 * @param model a char language model
 *
 * @return a next char classifier based on the given language model
 */
private fun buildProcessor(model: CharLM) = ChainProcessor(
  inputProcessor = EmbeddingsProcessor(model.charsEmbeddings),
  hiddenProcessors = listOf(RecurrentNeuralProcessor(
    model = model.hiddenNetwork,
    useDropout = false,
    propagateToInput = false)),
  outputProcessor = BatchFeedforwardProcessor(
    model = model.outputClassifier,
    useDropout = false,
    propagateToInput = false))
