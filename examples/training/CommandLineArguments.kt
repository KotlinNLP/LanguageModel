/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import com.xenomachina.argparser.ArgParser

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path in which to serialize the model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the file path in which to serialize the model"
  )

  /**
   * The file path of the training dataset.
   */
  val trainingSetPath: String by parser.storing(
    "-t",
    "--training-set-path",
    help="the file path of the training dataset"
  )

  /**
   * whether to train in reverse mode.
   */
  val reverse: Boolean by parser.flagging(
    "-r",
    "--reverse",
    help="whether to train in reverse mode"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
