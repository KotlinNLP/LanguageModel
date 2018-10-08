/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * The class used to update the CharLM model based on the backward errors.
 *
 * @param model the model
 * @param updateMethod the update method
 */
internal class Optimizer(
  model: CharLM,
  updateMethod: UpdateMethod<*>
) : Optimizer<ParamsErrors>(updateMethod = updateMethod) {

  /**
   * The optimizer used to optimize the network parameters of the CharLM.
   */
  private val paramsOptimizer = ParamsOptimizer(model.params, updateMethod)

  /**
   * The optimizer used to optimize the characters embeddings.
   */
  private val embeddingsOptimizer = EmbeddingsOptimizer(model.charsEmbeddings, updateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {

    this.paramsOptimizer.update()
    this.embeddingsOptimizer.update()
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the parameters errors to accumulate
   * @param copy a Boolean indicating if the [paramsErrors] can be used as reference or must be copied. Set copy = false
   *             to optimize the accumulation when the amount of the errors to accumulate is 1. (default = true)
   */
  override fun accumulate(paramsErrors: ParamsErrors, copy: Boolean) {

    this.paramsOptimizer.accumulate(paramsErrors.recurrentClassifier, copy = copy)
    paramsErrors.embeddings.forEach { this.embeddingsOptimizer.accumulate(it, errors = it.array.values) }
  }
}
