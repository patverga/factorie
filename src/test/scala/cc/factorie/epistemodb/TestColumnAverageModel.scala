package cc.factorie.epistemodb

import org.scalatest.junit.JUnitSuite
import cc.factorie.util
import org.junit.Test
import scala.util.Random
import org.junit.Assert._
import scala.Some

class TestColumnAverageModel extends JUnitSuite  with util.FastLogging {

  @Test def testSplitRandomizedTest() {
    val numRows = 1000
    val numCols = 10000
    val nnz = 100000

    val numTopics = 100
    val noise1 = 0.1

    // Test whether objective function goes up
    for (seed <- 0 until 2) {
      val random = new Random(seed)
      val m = CoocMatrix.randomOneZeroMatrix(numRows, numCols, nnz, random, numTopics, noise1).prune(1, 1)._1
      println("nnz: " + m.nnz())

      val stepsize = 0.1
      val regularizer = 0.01
      val dim = 10
      val iters = 10

      val rowToCols = m.rowToColAndVal.map{ case (row, cols) => row -> cols.keys.toIndexedSeq}.toMap
      val model = ColumnAverageModel.randomModel(rowToCols, numCols, dim, random)
      val trainer = new ColumnAverageTrainer(regularizer, stepsize, dim, m, model, random)
      val objectiveValues = trainer.train(iters)
      assertTrue(objectiveValues(0) < objectiveValues(9))
      assertTrue(objectiveValues(0) < objectiveValues(4))
      assertTrue(objectiveValues(4) < objectiveValues(9))
    }

    val numDevNNZ = 0
    val numTestNNZ = 150

    for (seed <- 0 until 2) {
      val random = new Random(seed)
      val m = CoocMatrix.randomOneZeroMatrix(numRows, numCols, nnz, random, numTopics, noise1).prune(1, 1)._1
      val rowToCols = m.rowToColAndVal.map{ case (row, cols) => row -> cols.keys.toIndexedSeq}.toMap

      println("nnz: " + m.nnz())
      val (mTrain, mDev, mTest) = m.randomTestSplit(numDevNNZ, numTestNNZ, None, Some(Set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)), random)

      val stepsize = 0.1
      val regularizer = 0.01
      val dim = 10

      // Train model for different number of iterations
      val model0 = ColumnAverageModel.randomModel(rowToCols, numCols, dim, random)
      val model5 = ColumnAverageModel.randomModel(rowToCols,  numCols, dim, random)
      val trainer5 = new ColumnAverageTrainer(regularizer, stepsize, dim, mTrain, model5, random)
      trainer5.train(5)
      println("--")
      val model10 = ColumnAverageModel.randomModel(rowToCols, numCols, dim, random)
      val trainer10 = new ColumnAverageTrainer(regularizer, stepsize, dim, mTrain, model10, random)
      trainer10.train(10)

      val result0 = model0.similaritiesAndLabels(mTrain, mTest)
      val result5 = model5.similaritiesAndLabels(mTrain, mTest)
      val result10 = model10.similaritiesAndLabels(mTrain, mTest)

      println("0 iters map: " + Evaluator.meanAveragePrecision(result0))
      println("5 iters map: " + Evaluator.meanAveragePrecision(result5))
      println("10 iters map: " + Evaluator.meanAveragePrecision(result10))

      assertTrue(Evaluator.meanAveragePrecision(result5) > Evaluator.meanAveragePrecision(result0))
      assertTrue(Evaluator.meanAveragePrecision(result10) > Evaluator.meanAveragePrecision(result5))
    }
  }

}