/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of the character language-model.
 *
 * This implementation uses the characters as atomic units of language modeling,allowing text to be treated as a
 * sequence of characters passed to an recurrent classifier network which at each point in the sequence is trained
 * to predict the next character.
 *
 * @property reverseModel where to train the model by seeing the sequence reversed (default false)
 * @param charsDict the dictionary containing the known characters (used both for embeddings and for prediction output)
 * @param inputSize the input size
 * @param inputDropout the input dropout (used during training only, default 0.0)
 * @param recurrentHiddenSize the size of the recurrent hidden layers
 * @param recurrentHiddenDropout the dropout of the recurrent hidden layers (used during training only, default 0.0)
 * @param recurrentConnectionType the recurrent connection type (e.g. LSTM, GRU, RAN)
 * @param recurrentHiddenActivation the activation function of the recurrent layers (can be null)
 * @param recurrentLayers the number of recurrent layers (min 1)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class CharLM(
  val reverseModel: Boolean = false,
  private val charsDict: DictionarySet<Char>,
  inputSize: Int = 20,
  inputDropout: Double = 0.0,
  recurrentHiddenSize: Int = 100,
  recurrentHiddenDropout: Double = 0.0,
  recurrentConnectionType: LayerType.Connection,
  recurrentHiddenActivation: ActivationFunction?,
  recurrentLayers: Int,
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
     * The char used for unknown chars.
     */
    const val UNK: Char = 0.toChar()

    /**
     * The char used to identify the end of text.
     */
    const val ETX = 3.toChar()

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
   * The CharLM contains the parameter of its sub-networks.
   *
   * @property recurrentParams network parameters of the recurrent neural network
   * @property classifierParams network parameters of the feed-forward neural network
   */
  class RecurrentClassifierParameters(
    val recurrentParams: NetworkParameters,
    val classifierParams: NetworkParameters
  ) : IterableParams<RecurrentClassifierParameters>() {

    /**
     * The list of all parameters.
     */
    override val paramsList: List<UpdatableArray<*>> = this.buildParamsList()

    /**
     * @return a new copy of all parameters of this
     */
    override fun copy() = RecurrentClassifierParameters(
      recurrentParams = this.recurrentParams.copy(),
      classifierParams = this.classifierParams.copy())

    /**
     * @return the list with parameters of all layers
     */
    private fun buildParamsList(): List<UpdatableArray<*>> =
      this.recurrentParams.paramsList + this.classifierParams.paramsList
  }

  /**
   * The chars embeddings.
   */
  val charsEmbeddings = EmbeddingsMap<Char>(size = inputSize)

  /**
   * The Recurrent Network to process the sequence left-to-right, or right-to-left if [reverseModel].
   */
  val recurrentNetwork: NeuralNetwork

  /**
   * The Feed-forward Network to predict the next char of the sequence.
   */
  val classifier: NeuralNetwork

  /**
   * The output size of the classifier.
   */
  val classifierOutputSize: Int = this.charsDict.size

  /**
   * Contains the aggregated parameters.
   */
  val params: RecurrentClassifierParameters

  /**
   * The id of the [UNK] char in the dictionary.
   */
  private val unkCharId: Int by lazy { this.charsDict.getId(UNK)!! }

  /**
   * The id of the [ETX] char in the dictionary.
   */
  val etxCharId: Int by lazy { this.charsDict.getId(ETX)!! }

  init {

    require(recurrentLayers >= 0) { "The number of recurrent layers must be >= 0." }
    require(recurrentConnectionType.property == LayerType.Property.Recurrent) { "The connection type must be recurrent." }

    val layersConfiguration = mutableListOf<LayerInterface>()

    layersConfiguration.add(LayerInterface(size = inputSize, type = LayerType.Input.Dense, dropout = inputDropout))

    layersConfiguration.addAll((0 until recurrentLayers).map {
      LayerInterface(
        size = recurrentHiddenSize,
        activationFunction = recurrentHiddenActivation,
        connectionType = recurrentConnectionType,
        dropout = recurrentHiddenDropout
      )})

    this.recurrentNetwork = NeuralNetwork(
      layerConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    (this.recurrentNetwork.model.paramsPerLayer.last() as? LSTMLayerParameters)?.initForgetGateBiasToOne()

    this.classifier = NeuralNetwork(
      LayerInterface(
        size = this.recurrentNetwork.outputSize,
        type = LayerType.Input.Dense),
      LayerInterface(
        size = this.classifierOutputSize,
        activationFunction = Softmax(),
        connectionType = LayerType.Connection.Feedforward),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    this.params = RecurrentClassifierParameters(
      recurrentParams = this.recurrentNetwork.model,
      classifierParams = this.classifier.model)

    this.charsDict.getElements().forEach { char ->
      if (char != ETX) this.charsEmbeddings.set(char)
    }
  }

  /**
   * @param c a char
   *
   * @return the id of the char [c] in the dictionary (or the unknown-id if it does not exist)
   */
  fun getCharId(c: Char): Int = this.charsDict.getId(c) ?: this.unkCharId

  /**
   * @param id an id
   *
   * @return the char corresponding to the [id]
   */
  fun getChar(id: Int): Char = this.charsDict.getElement(id)!!

  /**
   *
   */
  fun isEndOfSentence(c: Char): Boolean = c == ETX

  /**
   * Serialize this [CharLM] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [CharLM]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
