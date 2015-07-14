package cc.factorie.epistemodb.oval

import cc.factorie.app.nlp.embeddings._
import cc.factorie.epistemodb._
import cc.factorie.epistemodb.tac.TrainTestTacData
import cc.factorie.la.{Tensor, DenseTensor1, Tensor1, WeightsMapAccumulator}
import cc.factorie.model.{Weights1, Parameters, Weights, WeightsMap}
import cc.factorie.optimize._
import cc.factorie.util.DoubleAccumulator
import cc.factorie._

import scala.collection.Seq
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * Created by pat on 7/2/15.
 */
class ColumnAverageOval(val rowToCols : Map[Int, Seq[Int]], dim : Int, val numCols : Int, rand: Random,
                        val scoreType : String = "cbow", val ovalType : OvalType = DiagonalGaussian)
  extends MatrixModel with Parameters {

  implicit val r = rand
  val colVectors = Array.fill(numCols)(new DiagonalElliptic(this, dim))

  val energy = ovalType match {
    case DiagonalGaussian => new DiagonalGaussianLogExpectedLikelihoodEnergy
    case DiagonalCauchy => ??? //new DiagonalCauchyLogExpectedLikelihoodEnergy
    case DiagonalNull => ??? //new DiagonalNullLogExpectedLikelihoodEnergy
    case SphericalGaussian => ???
    case SphericalCauchy => ???
  }
  def similarity01(row: Int, col: Int): Double = {
    score(colVectors(col), rowToCols(row).map(colVectors(_)))
  }

  def score(targetCol : DiagonalEllipticLike, otherCols : Seq[DiagonalEllipticLike]): Double =
  {
    val values = otherCols.map(oval => energy.value(targetCol, oval))
    if (values.isEmpty)
      0.0
    else
      scoreType match {
        case "max" =>
          values.max
        case "cbow" =>
          values.sum / math.max(1.0, values.size)
        case _ => throw new NotImplementedError(s"$scoreType is not a valid value for neighborhood")
      }
  }

  def scoreAndMax(targetCol : DiagonalEllipticLike, otherCols : Seq[DiagonalEllipticLike]): (Double, Int) =
  {
    val values = otherCols.map(oval => energy.value(targetCol, oval))
    if (values.isEmpty)
      (0.0, -1)
    else
      values.zipWithIndex.maxBy(_._1)
  }
}

object ColumnAverageOval {
  def randomModel(rowToCols: Map[Int, Seq[Int]], numCols: Int, dim: Int, r: Random = new Random(0)): ColumnAverageOval = {
    new ColumnAverageOval(rowToCols, dim, numCols, r)
  }
}

class ColumnAverageOvalExample(energy: EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike], posColVec : DiagonalEllipticLike, negColVec : DiagonalEllipticLike,
                               sharedRowVecs : Seq[DiagonalEllipticLike], scoreType: String, margin : Double = 1.0) extends Example {

  val factor = 1.0

  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    scoreType match {
      case "max" =>
        // find the max score between the pos relation and all relations in eprels
        val ((posenergy, EnergyGradient2(posingrad, posoutgrad)), otherRel) = sharedRowVecs.map(rel => (energy.valueAndGradient(rel, posColVec), rel)).maxBy(_._1._1)
        val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negColVec)
        accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
      case "cbow" =>
        for (otherRel <- sharedRowVecs) {
          val (posenergy, EnergyGradient2(posingrad, posoutgrad)) = energy.valueAndGradient(otherRel, posColVec)
          val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negColVec)
          accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
        }
      case _ => throw new NotImplementedError(s"$scoreType is not a valid value for neighborhood")
    }
  }

  def accumulate(value: DoubleAccumulator, gradient: WeightsMapAccumulator, posenergy: Double, posingrad: WeightsMap, posoutgrad: WeightsMap, negenergy: Double, negingrad: WeightsMap, negoutgrad: WeightsMap): Unit = {
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

class ColumnAverageOvalTrainer(val regularizer: Double, val stepsize: Double, val dim: Int, margin: Double,
                           val matrix: CoocMatrix, val model: ColumnAverageOval, val random: Random, val delta : Double = 0.01,
                           val variancel2 : Double = 0.5, val varianceMin : Double = 0.01, val varianceMax : Double = 100.0)
  extends BprTrainer {

  val varianceOptimizer = new AdaGrad(stepsize, delta) with WeightDecayStep with HypercubeConstraintStep with SynchronizedWeightsStep {
//  val varianceOptimizer = new AdaGrad(stepsize, delta) with HypercubeConstraintStep {
    val min = varianceMin
    val max = varianceMax
    val lambda = variancel2
  }

  val embeddingOptimizer = new AdaGradRDA(delta = delta, rate = stepsize, l2 = regularizer)  with SynchronizedWeights

  val varianceSet = model.colVectors.map(_.variance: Weights).toSet
  val meanSet = model.colVectors.map(_.mean: Weights).toSet
  val optimizer = new MultiplexOptimizer(Seq(varianceOptimizer, embeddingOptimizer), w => if (meanSet(w)) embeddingOptimizer else varianceOptimizer)
  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue, nThreads = 1)
  optimizer.initializeWeights(model.parameters)


  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double =
  {
    val posColVec = model.colVectors(colIndex)
    val sharedRowVecs = for (col <- model.rowToCols(rowIndexTrue) if col != colIndex)
      yield model.colVectors(col)

    var negColIndex = random.nextInt(model.numCols)
    while (model.rowToCols(rowIndexTrue).contains(negColIndex)) negColIndex = random.nextInt(model.numCols)
    val negColVec = model.colVectors(negColIndex)

    val scoreTrueCell = model.score(posColVec, sharedRowVecs)
    val scoreFalseCell = model.score(negColVec, sharedRowVecs)
    val diff = scoreTrueCell - scoreFalseCell - margin
    val objective = 1 - (1 / (1 + math.exp(-diff)))
    val factor = if(objective > 0.0) 1.0 else 0.0

    trainer.processExample(new ColumnAverageOvalExample(model.energy, posColVec, negColVec, sharedRowVecs, scoreType = model.scoreType))

    objective
  }
}

object TrainTestTacDataOval extends TrainTestTacData{
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
    val model = ColumnAverageOval.randomModel(rowToCols, kb.numCols(), opts.dim.value, random)
    val trainer = new ColumnAverageOvalTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
      opts.margin.value, trainKb.matrix, model, random)

    evaluate(model, trainer, trainKb.matrix, testKb.matrix)

    //    if (!opts.patternsOut.value.isEmpty) {
    //      kb.writeTopPatterns(testCols, model, 0.5, opts.patternsOut.value)
    //    }

  }
}

