/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.simplednn.core.embeddings.Embedding

/**
 * The class used to save the params errors during the training of a [CharLM].
 *
 * @property recurrentClassifier
 * @property embeddings
 */
internal data class ParamsErrors(
  val recurrentClassifier: CharLM.RecurrentClassifierParameters,
  val embeddings: List<Embedding>
)