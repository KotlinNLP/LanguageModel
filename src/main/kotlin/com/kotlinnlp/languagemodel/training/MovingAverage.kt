/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagemodel.training

/**
 * Compute the moving average giving more importance to the recent added values. *
 * TODO: move to KotlinNLP/Utils
 *
 * @param windowSize the size of the observation window
 */
class MovingAverage(private val windowSize: Int) {

  /**
   * The mean.
   */
  var mean: Double = 0.0
    private set

  /**
   * The variance.
   */
  var variance: Double = 0.0
    private set

  /**
   * The standard deviation.
   */
  var stdDev: Double = 0.0
    private set

  /**
   * The values collected.
   */
  private val values: MutableList<Double> = mutableListOf()

  /**
   * Add the given [value] to the moving average
   *
   * @param value the value to add to the moving average
   */
  fun add(value: Double) {

    this.values.add(value)

    if (this.values.size > this.windowSize) this.values.removeAt(0)

    this.mean = this.values.average()
    this.variance = this.values.asSequence().map { (it - this.mean) * (it - this.mean) }.average()
    this.stdDev = Math.sqrt(this.variance)
  }
}
