/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.languagemodel.training.Trainer
import com.kotlinnlp.languagemodel.training.addSpecialChars
import com.kotlinnlp.languagemodel.training.collectChars
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.DictionarySet
import java.io.File

/**
 * Train the CharLM.
 *
 * The first argument is the corpus file path.
 * The second argument is the filename in which to save the model.
 * The presence of a third parameter indicates whether to train in reverse mode.
 */
fun main(args: Array<String>) {

  val corpusFilePath = args[0]
  val reverse: Boolean = args.size > 2
  val modelFileName = args[1] + if (reverse) ".rev" else ""
  val charsDict = DictionarySet<Char>()

  File(corpusFilePath).collectChars(charsDict, maxSentences = 100000)
  charsDict.addSpecialChars()

  println("Collected ${charsDict.size} characters.")

  if (reverse) println("Train the reverse model.")

  val model = CharLM(
    reverseModel = reverse,
    charsDict = charsDict,
    inputSize = charsDict.size,
    inputDropout = 0.0,
    recurrentHiddenSize = 1000,
    recurrentHiddenDropout = 0.0,
    recurrentConnectionType = LayerType.Connection.LSTM,
    recurrentHiddenActivation = Tanh(),
    recurrentLayers = 1)

  val trainer = Trainer(
    model = model,
    modelFilename = modelFileName,
    corpusFilePath = corpusFilePath,
    epochs = 1,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    verbose = true)

  trainer.train()
}
