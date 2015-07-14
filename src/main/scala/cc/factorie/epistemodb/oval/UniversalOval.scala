package cc.factorie.epistemodb.oval

import cc.factorie.Example
import cc.factorie._
import cc.factorie.app.nlp.embeddings.LiteHogwildTrainer
import cc.factorie.epistemodb.oval.TrainTestTacDataOval._
import cc.factorie.epistemodb.tac.TrainTestTacData
import cc.factorie.epistemodb.{StringStringKBMatrix, BprTrainer, CoocMatrix, MatrixModel}
import cc.factorie.la.WeightsMapAccumulator
import cc.factorie.model.{Weights, WeightsMap, Parameters}
import cc.factorie.optimize._
import cc.factorie.util.DoubleAccumulator

import scala.collection.Seq
import scala.util.Random

/**
 * Created by pat on 7/14/15.
 */
class UniversalOval( dim: Int, val numRows : Int, val numCols: Int, rand: Random, val ovalType: OvalType = DiagonalGaussian)
  extends MatrixModel with Parameters
{
  implicit val r = rand
  val rowVectors = Array.fill(numCols)(new DiagonalElliptic(this, dim))
  val colVectors = Array.fill(numCols)(new DiagonalElliptic(this, dim))

  val energy = ovalType match {
    case DiagonalGaussian => new DiagonalGaussianLogExpectedLikelihoodEnergy
    case DiagonalCauchy => ??? //new DiagonalCauchyLogExpectedLikelihoodEnergy
    case DiagonalNull => ??? //new DiagonalNullLogExpectedLikelihoodEnergy
    case SphericalGaussian => ???
    case SphericalCauchy => ???
  }

  def similarity01(row: Int, col: Int): Double = {
    score(rowVectors(row), colVectors(col))
  }

  def score(rowVec : DiagonalEllipticLike, colVec : DiagonalEllipticLike): Double ={
    energy.value(rowVec, colVec)
  }
}

object UniversalOval {
  def randomModel(numRows :Int, numCols: Int, dim: Int, r: Random = new Random(0)): UniversalOval = {
    new UniversalOval(dim, numRows, numCols, r)
  }
}

class UniversalOvalExample(energy: EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike], posVec: DiagonalEllipticLike, negVec: DiagonalEllipticLike,
                               colVec: DiagonalEllipticLike, margin: Double = 1.0) extends Example {

  val factor = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    // find the max score between the pos relation and all relations in eprels
    val (posenergy, EnergyGradient2(posingrad, posoutgrad)) = energy.valueAndGradient(colVec, posVec)
    val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(colVec, negVec)

    if (posenergy - negenergy < margin) {
      if (value != null)
        value.accumulate(posenergy - negenergy - margin)
      if (gradient != null) {
        gradient.accumulate(posingrad, factor)
        gradient.accumulate(posoutgrad, factor)
        gradient.accumulate(negingrad, -factor)
        gradient.accumulate(negoutgrad, -factor)
      }
    }
  }
}


class UniversalOvalTrainer(val regularizer: Double, val stepsize: Double, val dim: Int, margin: Double,
                               val matrix: CoocMatrix, val model: UniversalOval, val random: Random, val delta : Double = 0.01,
                               val variancel2 : Double = 0.5, val varianceMin : Double = 0.01, val varianceMax : Double = 100.0)
  extends BprTrainer {

  val varianceOptimizer = new AdaGrad(stepsize, delta) with WeightDecayStep with HypercubeConstraintStep with SynchronizedWeightsStep {
    //  val varianceOptimizer = new AdaGrad(stepsize, delta) with HypercubeConstraintStep {
    val min = varianceMin
    val max = varianceMax
    val lambda = variancel2
  }

  val embeddingOptimizer = new AdaGradRDA(delta = delta, rate = stepsize, l2 = regularizer)  with SynchronizedWeights

  val varianceSet = (model.colVectors ++ model.rowVectors).map(_.variance: Weights).toSet
  val meanSet = (model.colVectors ++ model.rowVectors).map(_.mean: Weights).toSet
  val optimizer = new MultiplexOptimizer(Seq(varianceOptimizer, embeddingOptimizer), w => if (meanSet(w)) embeddingOptimizer else varianceOptimizer)
  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, nThreads = 1)
  optimizer.initializeWeights(model.parameters)


  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double =
  {
    val colVec = model.colVectors(colIndex)
    val posRowVec = model.rowVectors(rowIndexTrue)
    val negRowVec = model.rowVectors(rowIndexFalse)

    val scoreTrueCell = model.score(posRowVec, colVec)
    val scoreFalseCell = model.score(negRowVec, colVec)

    val diff = scoreTrueCell - scoreFalseCell - margin
    val objective = 1 - (1 / (1 + math.exp(-diff)))
    val factor = if(objective > 0.0) 1.0 else 0.0

    trainer.processExample(new UniversalOvalExample(model.energy, posRowVec, negRowVec, colVec))

    objective
  }
}

object TrainTestTacDataUniversalOval extends TrainTestTacData{
  def main(args: Array[String]) : Unit = {
    opts.parse(args)

    val tReadStart = System.currentTimeMillis
    val kb = StringStringKBMatrix.fromTsv(opts.tacData.value).prune(2,1)
    val tRead = (System.currentTimeMillis - tReadStart)/1000.0
    println(f"Reading from file and pruning took $tRead%.2f s")

    println("Stats:")
    println("Num Rows:" + kb.numRows())
    println("Num Cols:" + kb.numCols())
    println("Num cells:" + kb.nnz())

    val random = new Random(0)
    val numDev = 0
    val numTest = 10000
    val (trainKb, devKb, testKb) = kb.randomTestSplit(numDev, numTest, None, Some(testCols), random)
    val rowToCols = trainKb.matrix.rowToColAndVal.map{ case (row, cols) => row -> cols.keys.toIndexedSeq}.toMap
    val model = UniversalOval.randomModel(kb.numRows(), kb.numCols(), opts.dim.value, random)
    val trainer = new UniversalOvalTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
      opts.margin.value, trainKb.matrix, model, random)

    evaluate(model, trainer, trainKb.matrix, testKb.matrix)
  }
}

