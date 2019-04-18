/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of the character language-model.
 *
 * This implementation uses the characters as atomic units of language modeling, allowing text to be treated as a
 * sequence of characters passed to a recurrent classifier which at each point in the sequence is trained
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
  private val inputSize: Int = 20,
  private val inputDropout: Double = 0.0,
  private val recurrentHiddenSize: Int = 100,
  private val recurrentHiddenDropout: Double = 0.0,
  private val recurrentConnectionType: LayerType.Connection,
  private val recurrentHiddenActivation: ActivationFunction?,
  private val recurrentLayers: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : IterableParams<CharLM>(), Serializable {

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
     * Add to the [dict] the special chars used to identify the unknown and the end of the sentence.
     *
     * @param dict the chars dictionary set
     */
    fun addSpecialChars(dict: DictionarySet<Char>) {

      require(!dict.contains(CharLM.UNK))
      require(!dict.contains(CharLM.ETX))

      dict.add(CharLM.UNK)
      dict.add(CharLM.ETX)
    }

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
  val charsEmbeddings = EmbeddingsMap<Char>(size = inputSize)

  /**
   * The Recurrent Network to process the sequence left-to-right, or right-to-left if [reverseModel].
   */
  val recurrentNetwork: StackedLayersParameters

  /**
   * The Feed-forward Network to predict the next char of the sequence.
   */
  val classifier: StackedLayersParameters

  /**
   * The output size of the classifier.
   */
  val classifierOutputSize: Int = this.charsDict.size

  /**
   * The average perplexity of the model, calculated during the training.
   */
  var avgPerplexity: Double = 0.0
    private set

  /**
   * The list of all parameters.
   */
  override val paramsList: List<ParamsArray>

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

    this.recurrentNetwork = StackedLayersParameters(
      layersConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    (this.recurrentNetwork.paramsPerLayer.last() as? LSTMLayerParameters)?.initForgetGateBiasToOne()

    this.classifier = StackedLayersParameters(
      LayerInterface(
        size = this.recurrentNetwork.outputSize,
        type = LayerType.Input.Dense),
      LayerInterface(
        size = this.classifierOutputSize,
        activationFunction = Softmax(),
        connectionType = LayerType.Connection.Feedforward),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    this.paramsList = this.recurrentNetwork.paramsList + this.classifier.paramsList

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

  /**
   * Set the average perplexity of the model.
   */
  fun setAvgPerplexity(value: Double) { this.avgPerplexity = value }

  /**
   * @return a new copy of all parameters of this
   */
  override fun copy(): CharLM = CharLM(
    reverseModel = reverseModel,
    charsDict = charsDict,
    inputDropout = inputDropout,
    inputSize = inputSize,
    recurrentHiddenSize = recurrentHiddenSize,
    recurrentHiddenDropout = recurrentHiddenDropout,
    recurrentConnectionType = recurrentConnectionType,
    recurrentHiddenActivation = recurrentHiddenActivation,
    recurrentLayers = recurrentLayers,
    weightsInitializer = null,
    biasesInitializer = null
  ).apply { assignValues(this) }
}
