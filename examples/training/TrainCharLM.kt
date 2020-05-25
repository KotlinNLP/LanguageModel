/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.languagemodel.Trainer
import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.utils.DictionarySet
import com.xenomachina.argparser.mainBody
import java.io.File

/**
 * Train a [CharLM] model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val corpusFile = File(parsedArgs.trainingSetPath)
  val charsDict = DictionarySet<Char>().apply {
    getLinesSequence(file = corpusFile, maxLength = 100000).forEach { it.forEach { char -> add(char) } }
  }

  println("Dictionary size: ${charsDict.size}")
  println("Number of training sentences: ${getLinesSequence(corpusFile).count()}")

  if (parsedArgs.reverse) println("Train the reverse model.")

  println("\n-- START TRAINING")

  Trainer(
    model = CharLM(charsDict),
    modelFilename = parsedArgs.modelPath,
    sentences = getLinesSequence(file = corpusFile, reverseChars = parsedArgs.reverse).asIterable(),
    charsBatchesSize = 50,
    charsDropout = 0.25,
    gradientClipping = GradientClipping.byValue(0.25),
    updateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)
  ).train()
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
