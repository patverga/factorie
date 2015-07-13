package cc.factorie.epistemodb.activelearning

import cc.factorie.epistemodb.{Evaluator, RegularizedBprUniversalSchemaTrainer, UniversalSchemaModel, CoocMatrix}
import scala.util.Random
import scala.Some

/**
 * Created by beroth on 6/25/15.
 */
object SyntheticActiveLearningExperiments {

  def stddev(xs: List[Double], avg: Double): Double = xs match {
    case Nil => 0.0
    case ys => math.sqrt((0.0 /: ys) {
      (a,e) => a + math.pow(e - avg, 2.0)
    } / (xs.size))
  }


  def meanAndStandardError(l : List[Double]) : (Double, Double) = {
    val mean = l.reduceLeft(_ + _) / l.size
    val sdev : Double = stddev(l, mean)
    val stderr = sdev / math.sqrt(l.size)
    (mean, stderr)
  }

  def main(args: Array[String]) : Unit = {

    val numRows = 1000
    val numCols = 1000
    val nnz = 5000

    val numTopics = 100
    val dim = 10
    val noise1 = 0.1

    val stepsize = 0.1
    val regularizer = 0.01

    val numDevNNZ = 0
    val numTestNNZ = 0//150

    // Test matrix is constructed following underlying pattern.
    val mTest = new CoocMatrix(numRows, numCols)
    val testCols = Set(0,1,2,3,4,5,6,7,8,9)
    for (col <- testCols) {
      for (row <- Range(0, numRows)) {
        if (row % numTopics == col % numTopics) {
          mTest.set(row, col, 1.0)
        }
      }
    }


    var resultsGoldAnno = List[Double]()
    var resultsAnno = List[Double]()
    var resultsRandomAnno = List[Double]()
    var resultsNoAnno = List[Double]()

    var rulesProposed = 20

    var rulesAppliedAnno = List[Double]()
    var rulesAppliedRandomAnno = List[Double]()

    var cellsFilledCorrectAnno = List[Double]()
    var cellsFilledCorrectRandomAnno = List[Double]()
    var cellsFilledIncorrectAnno = List[Double]()
    var cellsFilledIncorrectRandomAnno = List[Double]()

    for (seed <- 0 until 10) {
      val random = new Random(seed)
      val m = CoocMatrix.randomOneZeroMatrix(numRows, numCols, nnz, random, numTopics, noise1) //.prune(1,1)._1

      println("nnz: " + m.nnz())

      val (mTrain,mDev,mTestUnused) = m.randomTestSplit(numDevNNZ, numTestNNZ, None, Some(testCols), random)

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

      var acceptedRulesHeuristic = 0.0
      var acceptedRulesRandom = 0.0

      var filledCellsIfHoldsHeuristic = 0.0
      var filledCellsIfNotHoldsHeuristic = 0.0

      var filledCellsIfHoldsRandom = 0.0
      var filledCellsIfNotHoldsRandom = 0.0

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
            acceptedRulesHeuristic += 1

            for(row <- mTrainAnno.colToRows.get(antecedentIdx).get) {
              // Get all nnz rows for antecedent
              val antecedentVal = mTrainAnno.get(row, antecedentIdx)

              if (antecedentVal == 1) {

                if (antecedentVal > mTrainRandomAnno.get(row, targetIdx)) {
                  mTrainAnno.set(row, targetIdx, antecedentVal)
                  if (row % numTopics == antecedentIdx % numTopics) {
                    filledCellsIfHoldsHeuristic += 1
                  } else {
                    filledCellsIfNotHoldsHeuristic += 1
                  }
                }
              }
            }
          }
        }

        val randomAntecedents = random.shuffle(modelRandomAnno.columnToExpectedRankingGain(mTrainRandomAnno, targetIdx, 2).toSeq).slice(0,20)
        for (antecedentIdx <- randomAntecedents.map(_._1)) {
          val relationHolds = ((antecedentIdx % numTopics) == (targetIdx % numTopics))
          if (relationHolds) {
            acceptedRulesRandom += 1

            for(row <- mTrainRandomAnno.colToRows.get(antecedentIdx).get) {
              // Get all nnz rows for antecedent
              val antecedentVal = mTrainRandomAnno.get(row, antecedentIdx)
              if (antecedentVal == 1) {
                if (antecedentVal > mTrainRandomAnno.get(row, targetIdx)) {
                  mTrainRandomAnno.set(row, targetIdx, antecedentVal)
                  if (row % numTopics == antecedentIdx % numTopics) {
                    filledCellsIfHoldsRandom += 1
                  } else {
                    filledCellsIfNotHoldsRandom += 1
                  }
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

      resultsGoldAnno = Evaluator.meanAveragePrecision(resultGoldAnno) :: resultsGoldAnno
      resultsAnno = Evaluator.meanAveragePrecision(resultAnno) :: resultsAnno
      resultsRandomAnno = Evaluator.meanAveragePrecision(resultRandomAnno) :: resultsRandomAnno
      resultsNoAnno = Evaluator.meanAveragePrecision(resultNoAnno) :: resultsNoAnno


      rulesAppliedAnno = acceptedRulesHeuristic :: rulesAppliedAnno
      rulesAppliedRandomAnno = acceptedRulesRandom :: rulesAppliedRandomAnno
      cellsFilledCorrectAnno = filledCellsIfHoldsHeuristic :: cellsFilledCorrectAnno
      cellsFilledCorrectRandomAnno = filledCellsIfHoldsRandom :: cellsFilledCorrectRandomAnno
      cellsFilledIncorrectAnno = filledCellsIfNotHoldsHeuristic :: cellsFilledIncorrectAnno
      cellsFilledIncorrectRandomAnno = filledCellsIfNotHoldsRandom :: cellsFilledIncorrectRandomAnno


      println("MAP gold annotation: " + Evaluator.meanAveragePrecision(resultGoldAnno))
      println("MAP selected rules: " + Evaluator.meanAveragePrecision(resultAnno))
      println("MAP random rules: " + Evaluator.meanAveragePrecision(resultRandomAnno))
      println("MAP no annotation: " + Evaluator.meanAveragePrecision(resultNoAnno))

      println("===")
    }

    println("===")
    println("Mean and standard error:")
    println("gold annotation: " + meanAndStandardError(resultsGoldAnno))
    println("selected rules: " + meanAndStandardError(resultsAnno))
    println("random rules: " + meanAndStandardError(resultsRandomAnno))
    println("no annotation: " + meanAndStandardError(resultsNoAnno))
    println("===")
    println("Rules Applied (out of 20 proposed)")
    println("heuristc: " + meanAndStandardError(rulesAppliedAnno))
    println("random: " + meanAndStandardError(rulesAppliedRandomAnno))
    println("===")
    println("Cells correctly filled:")
    println("heuristc: " + meanAndStandardError(cellsFilledCorrectAnno))
    println("random: " + meanAndStandardError(cellsFilledCorrectRandomAnno))
    println("===")
    println("Cells incorrectly filled:")
    println("heuristc: " + meanAndStandardError(cellsFilledIncorrectAnno))
    println("random: " + meanAndStandardError(cellsFilledIncorrectRandomAnno))
  }
}
