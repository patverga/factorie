package cc.factorie.epistemodb

import java.util

import cc.factorie.util.{DoubleAccumulator, Threading}
import scala.util.Random
import scala.collection._
import cc.factorie.la.{Tensor, WeightsMapAccumulator, DenseTensor1}
import cc.factorie.optimize.{MiniBatchExample, HogwildTrainer, Example, AdaGradRDA}
import cc.factorie.app.nlp.embeddings.LiteHogwildTrainer
import cc.factorie.model.{Weights, Weights1}

/**
 * Created by beroth on 2/19/15.
 */
abstract class BprTrainer {
  def stepsize: Double
  def dim: Int
  def matrix: CoocMatrix
  def model: MatrixModel
  def random: Random

  def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double

  def updateBprOnRows(rowTrue: UniversalSchemaModel.Row, rowFalse: UniversalSchemaModel.Row): Double = {
    val rowIdxTrue = rowTrue._1
    val rowIdxFalse = rowFalse._1
    val colIndicesTrueRow = rowTrue._2.keys
    val colIndicesFalseRow = rowFalse._2.keySet
    // for only updating those where ranking is incorrect, check: model.score(rowTrueIdx, ci) < model.score(rowFalseIdx, ci)
    val data: Iterable[Int] = colIndicesTrueRow.filter(!colIndicesFalseRow.contains(_))
    //    colIndices1.filter(!colIndices2.contains(_)).flatMap(ci => List((rowTrueIdx, rowFalseIdx, ci)))
    val shuffled: Iterable[Int] = random.shuffle(data)
    val objectives = shuffled.map(ci => updateBprCells(rowIdxTrue, rowIdxFalse, ci))
    //println("positive indices: " + colIndices1.length + "\n updates: " + data.length + "\n objective: " + objectives.sum)
    objectives.sum
  }

  def train(numIters: Int): IndexedSeq[Double] = {
//    val pool = Threading.newFixedThreadPool(1)
// TODO: parallelize
    val pool = Threading.newFixedThreadPool(Runtime.getRuntime.availableProcessors())
    var printNext = 1;
    val objSeq = for (t <- 0 until numIters) yield {
      var objective = 0.0
      // random order, for bpr
      //println("shuffling ...")
      // : Array[mutable.Iterable[UniversalSchemaModel.Row]]
      val batchedRows = random.shuffle(matrix.rowToColAndVal).toSeq.grouped(1000).toArray
      //println("... done.")

      Threading.parForeach(batchedRows, pool)(rowBatch => {
        // Take a sliding window of two rows, and do bpr update.
        val thisObjective = rowBatch.sliding(2).foldLeft(0.0)((obj, rowPair) =>
          obj + updateBprOnRows(rowPair(0), rowPair(1))
        )
        // For non-bpr this would be:
        //val thisObjective = exs.foldLeft(0.0)((b, a) => b + updateOnRow(a._1, a._2, stepsize))
        objective += thisObjective
      })
      if (t == printNext || t == 0 || t == numIters - 1) {
        println("finished iter " + t + " objective = " + objective / matrix.rowToColAndVal.size)
        printNext *= 2
      }
      objective
    }
    pool.shutdown()
    objSeq
  }
}

class TransEExample(val posVecE1: Weights, val posVecE2: Weights, val negVecE1: Weights, val negVecE2: Weights,
                    val colVec: Weights, posGrad : Tensor, negGrad : Tensor, obj : Double, margin : Double)
  extends Example {

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    value.accumulate(obj)
    if (obj > 0.0) {
      gradient.accumulate(posVecE1, posGrad, 1.0)
      gradient.accumulate(posVecE2, posGrad, -1.0)
      gradient.accumulate(negVecE1, negGrad, -1.0)
      gradient.accumulate(negVecE2, negGrad, 1.0)
      gradient.accumulate(colVec, posGrad - negGrad, 1.0)
    }
  }
}

class TransETrainer(val regularizer: Double, val stepsize: Double, val margin : Double, val dim: Int,
                                           val matrix: CoocMatrix, val model: TransEModel, val random: Random) extends
BprTrainer {
  val batchSize = 100
  val entityRegularizer = regularizer
  val colRegularizer = regularizer

  val optimizer = new AdaGradRDA(delta = 0.01 , rate = stepsize, l2 = regularizer)
  val trainer = new HogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue)
//  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue)

  optimizer.initializeWeights(model.parameters)


  override def train(numIters: Int): IndexedSeq[Double] = {
    val objSeq = for (t <- 0 until numIters) yield {
      val examplesAndObjectives = matrix.getNnzCells().map { case (row, col) => makeExample(row, col) }
      val examples = examplesAndObjectives.map(_._1)
      val batches = random.shuffle(examples).grouped(batchSize).map(batch => new MiniBatchExample(batch)).toSeq
      trainer.processExamples(batches)
      examplesAndObjectives.map(_._2).sum
    }
    objSeq
  }

  def makeExample(rowIndex: Int, colIndex: Int): (Example, Double) = {

    val (posE1, posE2) = this.model.rowToEnts(rowIndex)

    val takeE1 = random.nextBoolean()
    val negE = random.nextInt(model.numEnts)

    val (negE1, negE2) = if (takeE1) {
      (negE, posE2)
    } else {
      (posE1, negE)
    }

    val posGrad = model.gradient(posE1, posE2, colIndex)
    val negGrad = model.gradient(negE1, negE2, colIndex)

    val obj = margin + posGrad.twoNorm - negGrad.twoNorm

    val posVecE1 = model.entityVectors(posE1)
    val posVecE2 = model.entityVectors(posE2)
    val negVecE1 = model.entityVectors(negE1)
    val negVecE2 = model.entityVectors(negE2)
    val colVec = model.colVectors(colIndex)

    (new TransEExample(posVecE1, posVecE2, negVecE1, negVecE2, colVec, posGrad, negGrad, obj, margin), obj)

  }

 def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double = ???
//  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double = {
//    val (posE1, posE2) = this.model.rowToEnts(rowIndexTrue)
//
//    val takeE1 = random.nextBoolean()
//    val negE = random.nextInt(model.numEnts)
//
//    val (negE1, negE2) = if (takeE1) {
//      (negE, posE2)
//    } else {
//      (posE1, negE)
//    }
//
//    val posGrad = model.gradient(posE1, posE2, colIndex)
//    val negGrad = model.gradient(negE1, negE2, colIndex)
//
//    val obj = margin + posGrad.twoNorm - negGrad.twoNorm
//
//    val posVecE1 = model.entityVectors(posE1)
//    val posVecE2 = model.entityVectors(posE2)
//    val negVecE1 = model.entityVectors(negE1)
//    val negVecE2 = model.entityVectors(negE2)
//    val colVec = model.colVectors(colIndex)
//
//    trainer.processExample(new TransEExample(posVecE1, posVecE2, negVecE1, negVecE2, colVec, posGrad, negGrad, obj, margin))
//    obj
//  }
}

class RegularizedBprUniversalSchemaTrainer(val regularizer: Double, val stepsize: Double, val dim: Int,
                                            val matrix: CoocMatrix, val model: UniversalSchemaModel, val random: Random) extends
      BprTrainer {
  val rowRegularizer = regularizer
  val colRegularizer = regularizer

  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double = {
    val scoreTrueCell = model.score(rowIndexTrue, colIndex)
    val scoreFalseCell = model.score(rowIndexFalse, colIndex)
    val theta = scoreTrueCell - scoreFalseCell
    val prob = UniversalSchemaModel.calculateProb(theta)

    // TODO: include reguarizer into objective logged to output:
    // - sq(rowVectors) * model.rowRegularizer / 2
    // - sq(colVectors) * model.colRegularizer
    var thisObjective = math.log(prob)

    val step = stepsize * (1 - prob)
    val colVec = model.colVectors(colIndex).copy

    model.colVectors(colIndex).*=((1 - stepsize * colRegularizer))
    model.colVectors(colIndex).+=(model.rowVectors(rowIndexTrue) - model.rowVectors(rowIndexFalse), step)

    model.rowVectors(rowIndexTrue).*=((1 - stepsize * rowRegularizer))
    model.rowVectors(rowIndexTrue).+=(colVec, step)

    model.rowVectors(rowIndexFalse).*=((1 - stepsize * rowRegularizer))
    model.rowVectors(rowIndexFalse).+=(colVec, -step)

    thisObjective
  }
}

class NormConstrainedBprUniversalSchemaTrainer(val maxNorm: Double, val stepsize: Double, val dim: Int,
                                           val matrix: CoocMatrix, val model: UniversalSchemaModel, val random: Random) extends
    BprTrainer {

  val rowNormConstraint = maxNorm
  val colNormConstraint = maxNorm

  def constrain(vector: DenseTensor1, maxNorm: Double) {
    val norm = vector.twoNorm
    if (norm > maxNorm) {
      vector *= (maxNorm / norm)
    }
  }

  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double = {
    val scoreTrueCell = model.score(rowIndexTrue, colIndex)
    val scoreFalseCell = model.score(rowIndexFalse, colIndex)
    val theta = scoreTrueCell - scoreFalseCell
    val prob = UniversalSchemaModel.calculateProb(theta)

    var thisObjective = math.log(prob)

    val step = stepsize * (1 - prob)
    val colVec = model.colVectors(colIndex).copy

    model.colVectors(colIndex).+=(model.rowVectors(rowIndexTrue) - model.rowVectors(rowIndexFalse), step)
    constrain(model.colVectors(colIndex), colNormConstraint)

    model.rowVectors(rowIndexTrue).+=(colVec, step)
    constrain(model.rowVectors(rowIndexTrue), rowNormConstraint)

    model.rowVectors(rowIndexFalse).+=(colVec, -step)
    constrain(model.rowVectors(rowIndexFalse), rowNormConstraint)

    thisObjective
  }
}


class UniversalSchemaExample(rowVecTrue : Weights, rowVecFalse : Weights, colVec : Weights, var factor : Double) extends Example {

  factor = 1.0
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    gradient.accumulate(rowVecTrue, colVec.value, factor)
    gradient.accumulate(colVec, rowVecTrue.value, factor)

    gradient.accumulate(rowVecFalse, colVec.value, -factor)
    gradient.accumulate(colVec, rowVecFalse.value, -factor)

  }
}

class AdaGradUniversalSchemaTrainer(val maxNorm: Double, val stepsize: Double, val dim: Int,
                                    val matrix: CoocMatrix, val model: UniversalSchemaAdaGradModel, val random: Random) extends
BprTrainer {

  val regularizer = 0.01
  val entityRegularizer = regularizer
  val colRegularizer = regularizer

  val optimizer = new AdaGradRDA(delta = 0.01 , rate = stepsize, l2 = regularizer)
  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue)

  optimizer.initializeWeights(model.parameters)


  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double = {
    val scoreTrueCell = model.score(rowIndexTrue, colIndex)
    val scoreFalseCell = model.score(rowIndexFalse, colIndex)
    val theta = scoreTrueCell - scoreFalseCell
    val prob = UniversalSchemaModel.calculateProb(theta)
    var thisObjective = math.log(prob)

    val colVec = model.colVectors(colIndex)
    val rowVecTrue = model.rowVectors(rowIndexTrue)
    val rowVecFalse = model.rowVectors(rowIndexFalse)

    trainer.processExample(new UniversalSchemaExample(rowVecTrue, rowVecFalse, colVec, prob))

    thisObjective
  }
}


class ColumnAverageExample(posColVec : Weights, negColVec : Weights, sharedRowVecs : Seq[Weights]) extends Example {

  val factor = 1.0
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {

    sharedRowVecs.foreach(colVec => {
      gradient.accumulate(posColVec, colVec.value, factor)
      gradient.accumulate(colVec, posColVec.value, factor)

      gradient.accumulate(negColVec, colVec.value, -factor)
      gradient.accumulate(colVec, negColVec.value, -factor)
    })
  }
}

class ColumnAverageTrainer(val maxNorm: Double, val stepsize: Double, val dim: Int,
                                    val matrix: CoocMatrix, val model: ColumnAverageModel, val random: Random) extends
BprTrainer {

  val regularizer = 0.01
  val entityRegularizer = regularizer
  val colRegularizer = regularizer

  val optimizer = new AdaGradRDA(delta = 0.01 , rate = stepsize, l2 = regularizer)
  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue)

  optimizer.initializeWeights(model.parameters)


  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double =
  {
    val colVec = model.colVectors(colIndex)
    val sharedRowVecs = for (col <- model.rowToCols(rowIndexTrue) if col != colIndex)
      yield model.colVectors(col)


    var negColIndex = -1
    do negColIndex = random.nextInt(model.numCols) while (model.rowToCols(rowIndexTrue).contains(negColIndex))
    val negColVec = model.colVectors(negColIndex)


    val scoreTrueCell = model.score(colVec.value, sharedRowVecs.map(_.value))
    val scoreFalseCell = model.score(negColVec.value, sharedRowVecs.map(_.value))
    val theta = scoreTrueCell - scoreFalseCell
    val prob = UniversalSchemaModel.calculateProb(theta)
    val thisObjective = if (prob != 0) math.log(prob) else 0.0

    trainer.processExample(new ColumnAverageExample(colVec, negColVec, sharedRowVecs))

    thisObjective
  }
}

