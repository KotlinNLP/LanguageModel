/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import java.io.File
import java.io.FileInputStream

/**
 * Test the CharLM.
 *
 * The first argument is the model file name.
 * The second argument is the max length of an output sequence (default 100).
 */
fun main(args: Array<String>) {

  val model = CharLM.load(FileInputStream(File(args[0])))
  val maxSentenceLength: Int = if (args.size > 1) args[1].toInt() else 100

  val decoder = SimpleDecoder(model)

  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty())
      break
    else
      println(decoder.decode(firstChar = inputText[0], maxSentenceLength = maxSentenceLength))
  }

  println("\nExiting...")
}

/**
 * Read a value from the standard input.
 *
 * @return the string read
 */
private fun readValue(): String {

  print("\nType a single character (empty to exit): ")

  return readLine()!!
}