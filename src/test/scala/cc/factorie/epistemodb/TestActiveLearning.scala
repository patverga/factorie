package cc.factorie.epistemodb

import org.scalatest.junit.JUnitSuite
import cc.factorie.util
import org.junit.Test
import org.junit.Assert._

import scala.util.Random

/**
 * Created by beroth on 6/4/15.
 */
class TestActiveLearning extends JUnitSuite  with util.FastLogging  {
  val eps = 1e-4

  @Test def rankCorrelationTest() {
    val ranks1 = Seq((0,0),(1,3),(2,2),(3,1),(4,4))
    val ranks2 = Seq((0,4),(1,1),(2,2),(3,3),(4,0))
    assertEquals(0.6, Evaluator.spearmansRankCorrelation(ranks1), eps)
    assertEquals(-0.6, Evaluator.spearmansRankCorrelation(ranks2), eps)
    val values1 = Seq((0.0,0.0),(0.1,0.3),(0.2,0.2),(0.3,0.1),(4.0,4.0))
    assertEquals(ranks1.sorted, Evaluator.ranksFromValues(values1))
  }



  @Test def expectedRankingGainTest() {
    val numRows = 1000
    val numCols = 1000
    val nnz = 5000

    val numTopics = 100
    val dim = 10
    val noise1 = 0.1

    val numDevNNZ = 0
    val numTestNNZ = 0//150
    val mTest = new CoocMatrix(numRows, numCols)
    val testCols = Set(0,1,2,3,4,5,6,7,8,9)
    for (col <- testCols) {
      for (row <- Range(0, numRows)) {
        if (row % numTopics == col % numTopics) {
          mTest.set(row, col, 1.0)
        }
      }
    }

    for (seed <- 0 until 10) {
      val random = new Random(seed)
      val m = CoocMatrix.randomOneZeroMatrix(numRows, numCols, nnz, random, numTopics, noise1) //.prune(1,1)._1

      println("nnz: " + m.nnz())


      val (mTrain,mDev,mTestUnused) = m.randomTestSplit(numDevNNZ, numTestNNZ, None, Some(testCols), random)



      val stepsize = 0.1
      val regularizer = 0.01
      val seedForModels = random.nextInt()


      val mTrainGoldAnno = mTrain.copy()
      val modelGoldAnnotation = UniversalSchemaModel.randomModel(numRows, numCols, dim, new Random(seedForModels))
      val trainerGoldAnnotation = new RegularizedBprUniversalSchemaTrainer(regularizer, stepsize, dim, mTrainGoldAnno, modelGoldAnnotation, new Random(seedForModels))
      trainerGoldAnnotation.train(10)
      val inititalResultGoldAnno = modelGoldAnnotation.similaritiesAndLabels(mTrain, mTest)
      println("\nInitial MAP gold annotation: " + Evaluator.meanAveragePrecision(inititalResultGoldAnno) + "\n")

      val mTrainAnno = mTrain.copy()
      val modelAnno = UniversalSchemaModel.randomModel(numRows, numCols, dim, new Random(seedForModels))
      val trainerForAnnotation = new RegularizedBprUniversalSchemaTrainer(regularizer, stepsize, dim, mTrainAnno, modelAnno, new Random(seedForModels))
      trainerForAnnotation.train(10)
      val inititalResultAnno = modelAnno.similaritiesAndLabels(mTrain, mTest)
      println("\nInitial MAP selected rules: " + Evaluator.meanAveragePrecision(inititalResultAnno) + "\n")


      val mTrainNoAnno = mTrain.copy()
      val modelNoAnno = UniversalSchemaModel.randomModel(numRows, numCols, dim, new Random(seedForModels))
      val trainerNoAnno = new RegularizedBprUniversalSchemaTrainer(regularizer, stepsize, dim, mTrainNoAnno, modelNoAnno, new Random(seedForModels))
      trainerNoAnno.train(10)
      val initialResultNoAnno = modelAnno.similaritiesAndLabels(mTrainNoAnno, mTest)
      println("\nInitial MAP no annotation: " + Evaluator.meanAveragePrecision(initialResultNoAnno) + "\n")


      val mTrainRandomAnno = mTrain.copy()
      val modelRandomAnno = UniversalSchemaModel.randomModel(numRows, numCols, dim, new Random(seedForModels))
      val trainerRandomAnnotation = new RegularizedBprUniversalSchemaTrainer(regularizer, stepsize, dim, mTrainRandomAnno, modelRandomAnno, new Random(seedForModels))
      trainerRandomAnnotation.train(10)
      val initialResultRandomAnno = modelRandomAnno.similaritiesAndLabels(mTrainRandomAnno, mTest)
      println("\nInitial MAP random rules: " + Evaluator.meanAveragePrecision(initialResultRandomAnno) + "\n")


      assertEquals(Evaluator.meanAveragePrecision(initialResultRandomAnno), Evaluator.meanAveragePrecision(inititalResultAnno), 0.01)

      var numSelectedAnnotations = 0
      var numRandomAnnotations = 0

      var filledCellsIfHoldsHeuristic = 0
      var filledCellsIfNotHoldsHeuristic = 0

      var filledCellsIfHoldsRandom = 0
      var filledCellsIfNotHoldsRandom = 0

      for (targetIdx <- testCols) {

        for(row <- Range(0, mTrainGoldAnno.numRows())) {
          if (row % numTopics == targetIdx % numTopics) {
            mTrainGoldAnno.set(row, targetIdx, 1.0)
          }
        }

        val bestAntecedents = modelAnno.columnToExpectedRankingGain(mTrainAnno, targetIdx, 2).toSeq.sortBy(-_._2).slice(0,20)
        //val bestAntecedents = modelAnno.columnToFreq(mTrainAnno, targetIdx, 2).toSeq.sortBy(-_._2).slice(0,20)
        for (antecedentIdx <- bestAntecedents.map(_._1)) {
          val relationHolds = ((antecedentIdx % numTopics) == (targetIdx % numTopics))
          if (relationHolds) {
            for(row <- mTrainAnno.colToRows.get(antecedentIdx).get) {
              // Get all nnz rows for antecedent
              val antecedentVal = mTrainAnno.get(row, antecedentIdx)

              if (antecedentVal == 1) {
                if (row % numTopics == antecedentIdx % numTopics) {
                  filledCellsIfHoldsHeuristic += 1
                } else {
                  filledCellsIfNotHoldsHeuristic += 1
                }
                if (antecedentVal > mTrainRandomAnno.get(row, targetIdx)) {
                  mTrainAnno.set(row, targetIdx, antecedentVal)
                  numSelectedAnnotations += 1
                }
              }
            }
          }
        }

        val randomAntecedents = random.shuffle(modelRandomAnno.columnToExpectedRankingGain(mTrainRandomAnno, targetIdx, 2).toSeq).slice(0,20)
        for (antecedentIdx <- randomAntecedents.map(_._1)) {
          val relationHolds = ((antecedentIdx % numTopics) == (targetIdx % numTopics))
          if (relationHolds) {
            for(row <- mTrainRandomAnno.colToRows.get(antecedentIdx).get) {
              // Get all nnz rows for antecedent
              val antecedentVal = mTrainRandomAnno.get(row, antecedentIdx)

              if (antecedentVal == 1) {
                if (row % numTopics == antecedentIdx % numTopics) {
                  filledCellsIfHoldsRandom += 1
                } else {
                  filledCellsIfNotHoldsRandom += 1
                }
                if (antecedentVal > mTrainRandomAnno.get(row, targetIdx)) {
                  mTrainRandomAnno.set(row, targetIdx, antecedentVal)
                  numRandomAnnotations += 1
                }
              }
            }
          }
        }
      }


      println("SUGGESTED HERUISTIC:")
      println("Antecedent cells following pattern: " + filledCellsIfHoldsHeuristic)
      println("Antecedent cells not following pattern: " + filledCellsIfNotHoldsHeuristic)
      println("===")
      println("RANDOM SELECTION:")
      println("Antecedent cells following pattern: " + filledCellsIfHoldsRandom)
      println("Antecedent cells not following pattern: " + filledCellsIfNotHoldsRandom)
      println("===")

      println("selected annotations: " + numSelectedAnnotations)
      println("random annotations: " + numRandomAnnotations)

      println("\ntraining gold annotations:")
      trainerGoldAnnotation.train(10)
      println("\ntraining heuristic annotations:")
      trainerForAnnotation.train(10)
      println("\ntraining random annotations:")
      trainerRandomAnnotation.train(10)
      println("\ntraining no annotations:")
      trainerNoAnno.train(10)

      // Note: we are using mTrain here, in order to allow for annotated cells to have direct positive (or negative) impact.
      val resultGoldAnno = modelGoldAnnotation.similaritiesAndLabels(mTrain, mTest)
      val resultAnno = modelAnno.similaritiesAndLabels(mTrain, mTest)
      val resultRandomAnno = modelRandomAnno.similaritiesAndLabels(mTrain, mTest)
      val resultNoAnno = modelNoAnno.similaritiesAndLabels(mTrain, mTest)

      println("MAP gold annotation: " + Evaluator.meanAveragePrecision(resultGoldAnno))
      println("MAP selected rules: " + Evaluator.meanAveragePrecision(resultAnno))
      println("MAP random rules: " + Evaluator.meanAveragePrecision(resultRandomAnno))
      println("MAP no annotation: " + Evaluator.meanAveragePrecision(resultNoAnno))

      assertTrue(Evaluator.meanAveragePrecision(resultGoldAnno) >= Evaluator.meanAveragePrecision(resultAnno))
      assertTrue(Evaluator.meanAveragePrecision(resultAnno) > Evaluator.meanAveragePrecision(resultRandomAnno))
      assertTrue(Evaluator.meanAveragePrecision(resultAnno) >= Evaluator.meanAveragePrecision(resultNoAnno))

      println("===")
    }




  }

}
