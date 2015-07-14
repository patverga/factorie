package cc.factorie.epistemodb.oval

import cc.factorie.la.{Tensor1, DenseTensor1}
import cc.factorie.model.{WeightsSet, Parameters, Weights1, WeightsMap}
import cc.factorie.optimize.{GradientStep, GradientOptimizer}

import scala.util.Random

/**
 * Created by pat on 7/9/15.
 */
sealed trait OvalType
case object DiagonalGaussian extends OvalType
case object DiagonalCauchy extends OvalType
case object DiagonalNull extends OvalType
case object SphericalGaussian extends OvalType
case object SphericalCauchy extends OvalType

trait EnergyFunction2[V1, V2] {
  def valueAndGradient(v1: V1, v2: V2): (Double, EnergyGradient2)
  def value(v1: V1, v2: V2): Double = valueAndGradient(v1, v2)._1
}
trait EnergyFunction3[V1, V2, V3] {
  def valueAndGradient(v1: V1, v2: V2, v3: V3): (Double, EnergyGradient3)
  def value(v1: V1, v2: V2, v3: V3): Double = valueAndGradient(v1, v2, v3)._1
}

case class EnergyGradient2(v1grad: WeightsMap, v2grad: WeightsMap)
case class EnergyGradient3(v1grad: WeightsMap, v2grad: WeightsMap, v3grad: WeightsMap)

trait DiagonalEllipticLike {
  def variance: Weights1
  def mean: Weights1
}

class DiagonalElliptic(params: Parameters, dims: Int)(implicit r: Random) extends DiagonalEllipticLike {
  var mean = params.Weights(new DenseTensor1(dims))
  mean.value := Array.fill(dims)(r.nextDouble() / dims / 10 - 0.5 / dims / 10)
  var variance = params.Weights(new DenseTensor1(dims))
  variance.value := Array.fill(dims)(1.0)
}

class DiagonalEllipticPrecomputed(params: Parameters, __mean: DenseTensor1, __variance: DenseTensor1)(implicit r: Random) extends DiagonalEllipticLike {
  val mean = params.Weights(__mean)
  val variance = params.Weights(__variance)
}

trait SphericalEllipticLike {
  // this should be 1-dimensional with variance in 0th position
  def variance: Weights1
  def mean: Weights1
}

// Need to try non-log version for mixture model (tho this is a variational lower bound)
// lambda gives the "regularization" coefficient -- 1.0 corresponds to proper normalization
class DiagonalGaussianLogExpectedLikelihoodEnergy(lambda: Double = 1.0) extends EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike] {
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
      val delta = -0.5 * (diffSq / csum + lambda * math.log(csum))
      value += delta
      if (value.isNaN)
        println(value)
      m1grad(i) = ratio
      m2grad(i) = -ratio
      cgrad(i) = 0.5 * (diffSq - lambda * csum) / (csum * csum)
      i += 1
    }

    (value, m1grad, cgrad, m2grad, cgrad)
  }
}



class NullOptimizer extends GradientOptimizer {
  override def step(weights: WeightsSet, gradient: WeightsMap, value: Double): Unit = { }
  override def initializeWeights(weights: WeightsSet): Unit = { }
  override def reset(): Unit = { }
  override def finalizeWeights(weights: WeightsSet): Unit = { }
  override def isConverged: Boolean = true
}

trait WeightDecayStep extends GradientStep {
  def lambda: Double
  abstract override def doGradStep(weights: WeightsSet, gradient: WeightsMap, value: Double): Unit = {
    for (k <- gradient.keys) gradient(k) +=(weights(k), -lambda)
    super.doGradStep(weights, gradient, value)
  }
}

trait SynchronizedWeightsStep extends GradientStep {
  abstract override def doGradStep(weights: WeightsSet, gradient: WeightsMap, value: Double): Unit = {
    for ((k, v) <- gradient.toSeq) k.synchronized {
      val gradForWeight = new WeightsMap(_.newBlankTensor)
      gradForWeight(k) = v
      super.doGradStep(weights, gradForWeight, value)
    }
  }
  override def initializeWeights(weights: WeightsSet): Unit = this.synchronized {super.initializeWeights(weights)}
  override def reset(): Unit = this.synchronized {super.reset()}
  override def finalizeWeights(weights: WeightsSet): Unit = this.synchronized {super.finalizeWeights(weights)}
}