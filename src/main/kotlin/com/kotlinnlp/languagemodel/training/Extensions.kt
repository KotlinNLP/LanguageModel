/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.utils.DictionarySet
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader

/**
 * @param maxSentences the max number of sentences to read (can be null)
 * @param reverse whether to train the model reverting the chars sequences (default false)
 */
fun File.toSequence(maxSentences: Int? = null, reverse: Boolean = false) = sequence<String> {

  var i = 0

  this@toSequence.walk().filter { it.isFile }.forEach { file ->

    BufferedReader(InputStreamReader(FileInputStream(file), Charsets.UTF_8)).lines().use {

      for (line in it) {
        if (maxSentences == null || i < maxSentences) {
          i++
          yield(if (reverse) line.reversed() else line)
        } else {
          break
        }
      }
    }
  }
}

/**
 * @param destination where to collect the chars
 */
fun Sequence<String>.collectChars(destination: DictionarySet<Char>) {

  this.forEach { line ->

    if (line.contains(CharLM.ETX) || line.contains(CharLM.UNK)) {
      throw RuntimeException("The line can't contain NULL or ETX chars")
    }

    line.forEach { destination.add(it) }
  }
}
