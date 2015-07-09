package cc.factorie.epistemodb.oval

import cc.factorie.app.nlp.embeddings._
import cc.factorie.epistemodb.{BprTrainer, ColumnAverageModel, CoocMatrix, MatrixModel}
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
    case DiagonalGaussian => new CBOWDiagonalGaussianLogExpectedLikelihoodEnergy
    case DiagonalCauchy => ??? //new DiagonalCauchyLogExpectedLikelihoodEnergy
    case DiagonalNull => ??? //new DiagonalNullLogExpectedLikelihoodEnergy
    case SphericalGaussian => ???
    case SphericalCauchy => ???
  }

  def similarity(vec1: Tensor, vec2: Tensor): Double = vec1.cosineSimilarity(vec2)
  // cosine similarity normalized to lie between 0 and one
  def similarity01(vec1: Tensor, vec2: Tensor): Double =
    (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

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
          values.sum / values.size
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

  def cosSimilarity01(vec1: Tensor, vec2: Tensor): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

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

class ColumnAverageOvalTrainer(val maxNorm: Double, val stepsize: Double, val dim: Int,
                           val matrix: CoocMatrix, val model: ColumnAverageOval, val random: Random, val delta : Double = 0.01,
                           val variancel2 : Double = 0.5, val varianceMin : Double = 0.01, val varianceMax : Double = 100.0)
  extends BprTrainer {

  val regularizer = 0.01
  val margin = 1.0

  val varianceOptimizer = new AdaGrad(stepsize, delta) with WeightDecayStep with HypercubeConstraintStep with SynchronizedWeightsStep {
    val min = varianceMin
    val max = varianceMax
    val lambda = variancel2
  }

  val embeddingOptimizer = new AdaGradRDA(delta = delta, rate = stepsize, l2 = regularizer)  with SynchronizedWeights

  val varianceSet = model.colVectors.map(_.variance: Weights).toSet
  val meanSet = model.colVectors.map(_.mean: Weights).toSet
  val optimizer = new MultiplexOptimizer(Seq(varianceOptimizer, embeddingOptimizer), w => if (meanSet(w)) embeddingOptimizer else varianceOptimizer)
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

    val scoreTrueCell = model.score(colVec, sharedRowVecs)
    val scoreFalseCell = model.score(negColVec, sharedRowVecs)
    val diff: Double = scoreTrueCell - scoreFalseCell - margin
    val objective = 1 - (1 / (1 + math.exp(-diff)))
    val factor = if(objective > 0.0) 1.0 else 0.0

    trainer.processExamples(Seq(new ColumnAverageOvalExample(model.energy, colVec, negColVec, sharedRowVecs, scoreType = model.scoreType)))

    objective
  }
}

// Need to try non-log version for mixture model (tho this is a variational lower bound)
// lambda gives the "regularization" coefficient -- 1.0 corresponds to proper normalization
class CBOWDiagonalGaussianLogExpectedLikelihoodEnergy(lambda: Double = 1.0) extends EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike] {
  override def valueAndGradient(v1: DiagonalEllipticLike, v2: DiagonalEllipticLike): (Double, EnergyGradient2) = {
    val (value, m1grad, c1grad, m2grad, c2grad) = getValueAndGradient(v1.mean.value, v1.variance.value, v2.mean.value, v2.variance.value)
    val v1grad = new WeightsMap(_.newBlankTensor)
    val v2grad = new WeightsMap(_.newBlankTensor)
    v1grad(v1.mean) = m1grad
    v1grad(v1.variance) = c1grad
    v2grad(v2.mean) = m2grad
    v2grad(v2.variance) = c2grad
    (value, EnergyGradient2(v1grad, v2grad))
  }

  def getValueAndGradient(m1: Tensor1, c1: Tensor1, m2: Tensor1, c2: Tensor1): (Double, Tensor1, Tensor1, Tensor1, Tensor1) = {
    val dim = m1.dim1
    val m1grad = new DenseTensor1(dim)
    val m2grad = new DenseTensor1(dim)
    val cgrad = new DenseTensor1(dim)
    var value = 0.0
    var i = 0
    while (i < dim) {
      val csum = c1(i) + c2(i)
      val diff = m2(i) - m1(i)
      val diffSq = diff * diff
      val ratio = diff / csum
      value += -0.5 * (diffSq / csum + lambda * math.log(csum))
      m1grad(i) = ratio
      m2grad(i) = -ratio
      cgrad(i) = 0.5 * (diffSq - lambda * csum) / (csum * csum)
      i += 1
    }
    (value, m1grad, cgrad, m2grad, cgrad)
  }
}


