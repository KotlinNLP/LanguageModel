/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package test

import com.kotlinnlp.languagemodel.CharLM
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Test a [CharLM] model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val model: CharLM = parsedArgs.modelPath.let {
    println("Loading CharLM model from '$it'...")
    CharLM.load(FileInputStream(File(it)))
  }

  println("Max sentence length = ${parsedArgs.maxOutputLength}")

  val decoder = RandomWeightedChoiceDecoder(model)

  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty())
      break
    else
      println(decoder.decode(input = inputText, maxSentenceLength = parsedArgs.maxOutputLength))
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
