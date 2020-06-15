/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of the character language-model.
 *
 * This implementation uses the characters as atomic units of language modelling.
 * The texts are treated as a sequence of characters passed to a recurrent classifier which at each point in the
 * sequence is trained to predict the next character.
 *
 * @param charsDict the dictionary containing the known characters (used both for embeddings and for prediction output)
 * @param charsEmbeddingsSize the size of the chars embeddings (default 25)
 * @param recurrentHiddenSize the size of the recurrent hidden layers (default 200)
 * @param recurrentConnection the recurrent connection type (e.g. LSTM -default-, GRU, RAN, etc...)
 * @param recurrentHiddenActivation the activation function of the recurrent layers (default [Tanh])
 * @param numOfRecurrentLayers the number of recurrent layers (default 1)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class CharLM(
  private val charsDict: DictionarySet<Char>,
  charsEmbeddingsSize: Int = 25,
  recurrentHiddenSize: Int = 200,
  recurrentConnection: LayerType.Connection = LayerType.Connection.LSTM,
  recurrentHiddenActivation: ActivationFunction? = Tanh,
  numOfRecurrentLayers: Int = 1,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * The special char used to identify unknown chars.
     */
    internal const val UNK: Char = 0.toChar()

    /**
     * The special char used to identify the end of text.
     */
    internal const val ETX: Char = 3.toChar()

    /**
     * Read a [CharLM] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [CharLM]
     *
     * @return the [CharLM] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): CharLM = Serializer.deserialize(inputStream)
  }

  /**
   * The chars embeddings.
   */
  val charsEmbeddings = EmbeddingsMap<Char>(charsEmbeddingsSize)

  /**
   * The hidden recurrent network that encodes the sequence of chars.
   */
  val hiddenNetwork: StackedLayersParameters = StackedLayersParameters(
    layersConfiguration = listOf(
      LayerInterface(size = charsEmbeddingsSize, type = LayerType.Input.Dense)
    ) + (0 until numOfRecurrentLayers).map {
      LayerInterface(
        size = recurrentHiddenSize,
        activationFunction = recurrentHiddenActivation,
        connectionType = recurrentConnection
      )},
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  ).apply {
    (paramsPerLayer.last() as? LSTMLayerParameters)?.initForgetGateBiasToOne()
  }

  /**
   * The feed-forward network that predicts the next char of the sequence.
   */
  val outputClassifier: StackedLayersParameters = StackedLayersParameters(
    LayerInterface(
      size = this.hiddenNetwork.outputSize,
      type = LayerType.Input.Dense),
    LayerInterface(
      size = this.charsDict.size + 2, // 2 special chars
      activationFunction = Softmax(),
      connectionType = LayerType.Connection.Feedforward),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The average perplexity of the model, calculated during the training.
   */
  var avgPerplexity: Double = 0.0
    internal set

  /**
   * The id of the [UNK] char in the dictionary.
   */
  private val unkCharId: Int by lazy { this.charsDict.getId(UNK)!! }

  /**
   * The id of the [ETX] char in the dictionary.
   */
  val etxCharId: Int by lazy { this.charsDict.getId(ETX)!! }

  /**
   * Check requirements.
   * Complete the dictionary and the chars embeddings map with the special chars.
   */
  init {

    require(numOfRecurrentLayers > 0) { "The number of recurrent layers must be > 0." }
    require(recurrentConnection.property == LayerType.Property.Recurrent) {
      "The connection type must be recurrent."
    }

    require(!this.charsDict.contains(UNK)) { "The chars dictionary cannot contain the special UNK char." }
    require(!this.charsDict.contains(ETX)) { "The chars dictionary cannot contain the special ETX char." }

    this.charsDict.add(UNK)
    this.charsDict.add(ETX)

    this.charsDict.getElements().forEach { this.charsEmbeddings.set(it) }
  }

  /**
   * @param c a char
   *
   * @return the id of the char [c] if present in the dictionary, otherwise the id of the unknown char
   */
  fun getCharId(c: Char): Int = this.charsDict.getId(c) ?: this.unkCharId

  /**
   * @param id a char id
   *
   * @return the char with the given [id]
   */
  fun getChar(id: Int): Char = this.charsDict.getElement(id)!!

  /**
   * @param c a char
   *
   * @return whether the given char [c] is the "end of text" character
   */
  fun isEndOfSentence(c: Char): Boolean = c == ETX

  /**
   * Serialize this [CharLM] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [CharLM]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
