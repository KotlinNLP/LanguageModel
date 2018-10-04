/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.utils.DictionarySet
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader

/**
 * @param maxSentences the max number of sentences to read
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
 *
 */
fun DictionarySet<Char>.collectChars(corpus: File, maxSentences: Int) {

  corpus.forEachIndexedSentence(maxSentences) { _, line ->
    line.forEach { this.add(it) }
  }
}