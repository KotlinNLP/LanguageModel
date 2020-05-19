/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.languagemodel.training.Trainer
import com.kotlinnlp.languagemodel.training.collectChars
import com.kotlinnlp.languagemodel.training.toSequence
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.DictionarySet
import java.io.File

/**
 * Train and validate a [CharLM] model.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val corpus = File(parsedArgs.trainingSetPath)
  val charsDict = DictionarySet<Char>()

  corpus.toSequence(maxSentences = 100000).collectChars(charsDict)

  CharLM.addSpecialChars(charsDict)

  println("Dictionary size: ${charsDict.size}")
  println("Number of training sentences: ${corpus.toSequence().count()}")

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
    sentences = corpus.toSequence(reverse = parsedArgs.reverse),
    batchSize = 50,
    charsDropout = 0.25,
    gradientClipping = GradientClipping.byValue(0.25),
    epochs = 1,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    verbose = true)

  trainer.train()
}
