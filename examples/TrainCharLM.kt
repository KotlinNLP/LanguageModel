/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.languagemodel.training.Trainer
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
 */
fun main(args: Array<String>) {

  val corpusFilePath = args[0]
  val modelFileName = args[1]

  val charsDict = DictionarySet<Char>()

  charsDict.collectChars(corpus = File(corpusFilePath), maxSentences = 50000)

  val model = CharLM(
    reverseModel = true,
    charsDict = charsDict,
    inputSize = 20,
    inputDropout = 0.0,
    recurrentHiddenSize = 300,
    recurrentHiddenDropout = 0.0,
    recurrentConnectionType = LayerType.Connection.RAN,
    recurrentHiddenActivation = Tanh(),
    recurrentLayers = 1)

  val trainer = Trainer(
    model = model,
    modelFilename = modelFileName,
    corpusFilePath = corpusFilePath,
    epochs = 10,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    verbose = true)

  trainer.train()
}
