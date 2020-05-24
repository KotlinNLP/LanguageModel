/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.languagemodel.training.Trainer
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.DictionarySet
import com.xenomachina.argparser.mainBody
import java.io.File

/**
 * Train and validate a [CharLM] model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val corpusFile = File(parsedArgs.trainingSetPath)
  val charsDict = DictionarySet<Char>()

  collectChars(strings = getLinesSequence(file = corpusFile, maxLength = 100000), dictionary = charsDict)

  CharLM.addSpecialChars(charsDict)

  println("Dictionary size: ${charsDict.size}")
  println("Number of training sentences: ${getLinesSequence(corpusFile).count()}")

  if (parsedArgs.reverse) println("Train the reverse model.")

  val model = CharLM(
    charsDict = charsDict,
    inputSize = 25,
    recurrentHiddenSize = 200,
    recurrentHiddenDropout = 0.0,
    recurrentConnectionType = LayerType.Connection.LSTM,
    recurrentHiddenActivation = Tanh,
    recurrentLayers = 1)

  val trainer = Trainer(
    model = model,
    modelFilename = parsedArgs.modelPath,
    sentences = getLinesSequence(file = corpusFile, reverseChars = parsedArgs.reverse),
    batchSize = 50,
    charsDropout = 0.25,
    gradientClipping = GradientClipping.byValue(0.25),
    epochs = 1,
    updateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    verbose = true)

  trainer.train()
}

/**
 * Get the lines sequence of a file, with the possibility to read only the first lines and to revert their chars order.
 *
 * @param file the input file
 * @param maxLength the max length of the sequence or `null` to read all the lines
 * @param reverseChars whether to revert the chars order of each lines (default false)
 */
private fun getLinesSequence(file: File, maxLength: Int? = null, reverseChars: Boolean = false) = sequence {

  file.useLines { lines ->
    lines.forEachIndexed { i, line ->
      if (maxLength == null || i < maxLength)
        yield(if (reverseChars) line.reversed() else line)
      else
        return@useLines
    }
  }
}

/**
 * Collect the chars from a sequence of strings to a given dictionary.
 *
 * @param strings a sequence of strings
 * @param dictionary a dictionary set of chars
 */
private fun collectChars(strings: Sequence<String>, dictionary: DictionarySet<Char>) {

  strings.forEach { str ->

    if (str.contains(CharLM.ETX) || str.contains(CharLM.UNK))
      throw RuntimeException("The line can't contain the NULL or the ETX chars")

    str.forEach { dictionary.add(it) }
  }
}
