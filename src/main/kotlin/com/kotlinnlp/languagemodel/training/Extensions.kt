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
 */
fun File.forEachIndexedSentence(maxSentences: Int? = null, callback: (Int, String) -> Unit) {

  var i = 0

  this.walk().filter { it.isFile }.forEach { file ->

    BufferedReader(InputStreamReader(FileInputStream(file), Charsets.UTF_8)).lines().use {
      for (line in it) {
        if (maxSentences == null || i < maxSentences) callback(i++, line) else break
      }
    }
  }
}

/**
 * @param destination where to collect the chars
 * @param maxSentences the max number of sentences to read (can be null)
 */
fun File.collectChars(destination: DictionarySet<Char>, maxSentences: Int? = null) {

  this.forEachIndexedSentence(maxSentences) { _, line ->

    if (line.contains(CharLM.ETX) || line.contains(CharLM.UNK)) {
      throw RuntimeException("The line can't contain NULL or ETX chars")
    }

    line.forEach { destination.add(it) }
  }
}

/**
 * Add the special chars used to identify the unknown and the end of the sentence.
 */
fun DictionarySet<Char>.addSpecialChars() {

  require(!this.contains(CharLM.UNK))
  require(!this.contains(CharLM.ETX))

  this.add(CharLM.UNK)
  this.add(CharLM.ETX)
}
