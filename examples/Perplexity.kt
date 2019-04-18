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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.safeLog
import java.io.File
import java.io.FileInputStream
import kotlin.math.exp

/**
 * Get the perplexity of a sentence using the language model.
 * The perplexity is calculated with 'exp(loss)'.
 *
 * The first argument is the model file name.
 */
fun main(args: Array<String>) {

  val model = CharLM.load(FileInputStream(File(args[0])))
  val processor = buildProcessor(model)

  while (true) {

    val inputText = readValue().let { if (it.length == 1) " $it " else it}

    if (inputText.isEmpty()) {

      break

    } else {

      val prediction = processor.forward(inputText.take(inputText.length - 1).toList())

      // The target is always the next character.
      val targets = (0 until inputText.length - 1)
        .map { i -> model.getCharId(inputText[i + 1]) }
        .map { charId -> DenseNDArrayFactory.oneHotEncoder(length = model.classifier.outputSize, oneAt = charId) }

      val loss = prediction.zip(targets).map { (y, g) -> -safeLog(y[g.argMaxIndex()]) }.average()
      val perplexity = exp(loss)

      println("Perplexity: $perplexity")
    }
  }

  println("\nExiting...")
}

/**
 * Read a value from the standard input.
 *
 * @return the string read
 */
private fun readValue(): String {

  print("\nType the beginning of the sequence. Even a single character (empty to exit): ")

  return readLine()!!
}

/**
 * @return the processor to use the CharLM model
 */
private fun buildProcessor(model: CharLM) = ChainProcessor(
  inputProcessor = EmbeddingsProcessor(
    embeddingsMap = model.charsEmbeddings,
    useDropout = false),
  hiddenProcessors = listOf(
    RecurrentNeuralProcessor(
      model = model.recurrentNetwork,
      useDropout = false,
      propagateToInput = false)),
  outputProcessor = BatchFeedforwardProcessor(
    model = model.classifier,
    useDropout = false,
    propagateToInput = false))

