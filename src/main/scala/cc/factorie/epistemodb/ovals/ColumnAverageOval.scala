//package cc.factorie.epistemodb.ovals
//
//import cc.factorie.app.nlp.embeddings._
//import cc.factorie.app.nlp.embeddings.oval.DiagonalElliptic
//import cc.factorie.epistemodb.{BprTrainer, ColumnAverageModel, CoocMatrix, MatrixModel}
//import cc.factorie.la.{Tensor, DenseTensor1, Tensor1, WeightsMapAccumulator}
//import cc.factorie.model.{Weights1, Parameters, Weights, WeightsMap}
//import cc.factorie.optimize._
//import cc.factorie.util.DoubleAccumulator
//import cc.factorie._
//
//import scala.collection.Seq
//import scala.collection.mutable.ArrayBuffer
//import scala.util.Random
//
///**
// * Created by pat on 7/2/15.
// */
//class ColumnAverageOval(val rowToCols : Map[Int, Seq[Int]], dim : Int, val numCols : Int, val combineType : String = "cbow", val ovalType : OvalType = DiagonalGaussian)
//  extends MatrixModel with Parameters {
//
//  val colVectors = Array.fill(numCols)(new DiagonalElliptic(this, dim))
//
//  val energy = ovalType match {
//    case DiagonalGaussian => new CBOWDiagonalGaussianLogExpectedLikelihoodEnergy
//    case DiagonalCauchy => ??? //new DiagonalCauchyLogExpectedLikelihoodEnergy
//    case DiagonalNull => ??? //new DiagonalNullLogExpectedLikelihoodEnergy
//    case SphericalGaussian => ???
//    case SphericalCauchy => ???
//  }
//
//  def similarity(vec1: Tensor, vec2: Tensor): Double = vec1.cosineSimilarity(vec2)
//  // cosine similarity normalized to lie between 0 and one
//  def similarity01(vec1: Tensor, vec2: Tensor): Double =
//    (1.0 + vec1.cosineSimilarity(vec2)) / 2.0
//
//  def similarity01(row: Int, col: Int): Double = {
//    score(colVectors(col), rowToCols(row).map(colVectors(_)))
//  }
//
//  def score(targetCol : DiagonalElliptic, otherCols : Seq[DiagonalElliptic]): Double =
//  {
//    val values = otherCols.map(oval => energy.value(targetCol, oval))
//    if (values.isEmpty)
//      0.0
//    else
//      combineType match {
//        case "max" =>
//          values.max
//        case "cbow" =>
//          values.sum / math.min(1.0, values.size)
//        case _ => throw new NotImplementedError(s"$combineType is not a valid value for neighborhood")
//      }
//  }
//
//  def cosSimilarity01(vec1: Tensor, vec2: Tensor): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0
//
//  def getScoredColumns(v: DenseTensor1): Iterable[(Int, Double)] = ???
//
//  def getScoredRows(v: DenseTensor1): Iterable[(Int, Double)] = ???
//}
//
//object ColumnAverageOval {
//  def randomModel(rowToCols: Map[Int, Seq[Int]], numCols: Int, dim: Int, random: Random = new Random(0)): ColumnAverageOval = {
//    new ColumnAverageOval(rowToCols, dim, numCols)
//  }
//}
//
//class ColumnAverageExample(posColVec : Weights, negColVec : Weights, sharedRowVecs : Seq[Weights], combineType: String, margin : Double = 1.0) extends Example {
//
//  val factor = 1.0
//  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
//
//
//    sharedRowVecs.foreach(colVec => {
//      gradient.accumulate(posColVec, colVec.value, factor)
//      gradient.accumulate(colVec, posColVec.value, factor)
//
//      gradient.accumulate(negColVec, colVec.value, -factor)
//      gradient.accumulate(colVec, negColVec.value, -factor)
//    })
//  }
//  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
//    combineType match {
//      case "max" =>
//        // find the max score between the pos relation and all relations in eprels
//        val ((posenergy, EnergyGradient2(posingrad, posoutgrad)), otherRel) = epRels.map(rel => (energy.valueAndGradient(rel, posRel), rel)).maxBy(_._1._1)
//        for (negRel <- negRels) {
//          val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negRel)
//          accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
//        }
//      case "cbow" =>
//        for (otherRel <- epRels) {
//          val (posenergy, EnergyGradient2(posingrad, posoutgrad)) = energy.valueAndGradient(otherRel, posRel)
//          for (negRel <- negRels) {
//            val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negRel)
//            accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
//          }
//        }
//      case _ => throw new NotImplementedError(s"$neighborhood is not a valid value for neighborhood")
//    }
//  }
//
//  def accumulate(value: DoubleAccumulator, gradient: WeightsMapAccumulator, posenergy: Double, posingrad: WeightsMap, posoutgrad: WeightsMap, negenergy: Double, negingrad: WeightsMap, negoutgrad: WeightsMap): Unit = {
//    if (posenergy - negenergy < margin) {
//      if (value != null)
//        value.accumulate(posenergy - negenergy - margin)
//      if (gradient != null) {
//        gradient.accumulate(posingrad, 1.0)
//        gradient.accumulate(posoutgrad, 1.0)
//        gradient.accumulate(negingrad, -1.0)
//        gradient.accumulate(negoutgrad, -1.0)
//      }
//    }
//  }
//
//}
//
//class ColumnAverageTrainer(val maxNorm: Double, val stepsize: Double, val dim: Int,
//                           val matrix: CoocMatrix, val model: ColumnAverageModel, val random: Random, val delta : Double = 0.01,
//                           val variancel2 : Double = 0.5, val varianceMin : Double = 0.01, val varianceMax : Double = 100.0)
//  extends BprTrainer {
//
//  val regularizer = 0.01
//
//  val varianceOptimizer = new AdaGrad(stepsize, delta) with WeightDecayStep with HypercubeConstraintStep with SynchronizedWeightsStep {
//    val min = varianceMin
//    val max = varianceMax
//    val lambda = variancel2
//  }
//
//  val embeddingOptimizer = new AdaGradRDA(delta = delta, rate = stepsize, l2 = regularizer) with SynchronizedWeights
//
//  val varianceSet = model.colVectors.map(_.variance: Weights).toSet
//  val meanSet = model.colVectors.map(_.mean: Weights).toSet
//  val optimizer = new MultiplexOptimizer(Seq(varianceOptimizer, embeddingOptimizer), w => if (meanSet(w)) embeddingOptimizer else varianceOptimizer)
//  val trainer = new LiteHogwildTrainer(weightsSet = model.parameters, optimizer = optimizer, maxIterations = Int.MaxValue)
//  optimizer.initializeWeights(parameters)
//
//
//  override def updateBprCells(rowIndexTrue: Int, rowIndexFalse: Int, colIndex: Int): Double =
//  {
//    val colVec = model.colVectors(colIndex)
//    val sharedRowVecs = for (col <- model.rowToCols(rowIndexTrue) if col != colIndex)
//      yield model.colVectors(col)
//
//
//    var negColIndex = -1
//    do negColIndex = random.nextInt(model.numCols) while (model.rowToCols(rowIndexTrue).contains(negColIndex))
//    val negColVec = model.colVectors(negColIndex)
//
//
//    val scoreTrueCell = model.score(colVec.value, sharedRowVecs.map(_.value))
//    val scoreFalseCell = model.score(negColVec.value, sharedRowVecs.map(_.value))
//    val diff: Double = scoreTrueCell - scoreFalseCell - 0.0
//    val thisObjective = 1 - (1 / (1 + math.exp(-diff)))
//
//    trainer.processExample(new ColumnAverageExample(colVec, negColVec, sharedRowVecs))
//
//    thisObjective
//  }
//}
//
//class CBOWval(ovalType: OvalType = DiagonalGaussian, opts: RelationExtractionOpts){
//  var energy = null: EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike]
//  var relationEmbeddings = Array[DiagonalEllipticLike]()
//
//  override protected def makeExample(example: RelationTriplet): Example = new Example {
//    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
//      val negesamples = new ArrayBuffer[DiagonalEllipticLike]
//      val entPairRelations = trainEntPairRelations(example.entPair.get)
//
//      while (negesamples.size < opts.negative.value) {
//        var negRel = rand.nextInt(relationEmbeddings.size)
//        while (entPairRelations.contains(negRel) || negRel == example.relation) negRel = rand.nextInt(relationEmbeddings.size)
//        negesamples += relationEmbeddings(negRel)
//      }
//      val ex = new CBOWvalMarginTripletExample(energy, relationEmbeddings(example.relation), negesamples, trainEntPairRelations(example.entPair.get).map(relationEmbeddings(_)), opts.margin.value, opts.neighborhood.value)
//      ex.accumulateValueAndGradient(value, gradient)
//    }
//  }
//
//
//
//
//  override def getScore(triplet: RelationTriplet): Double = {
//    val RelationTriplet(ep, e1, e2, rel, label) = triplet
//    if (ep.isDefined) {
//      val epRelOvals = trainEntPairRelations(ep.get).map(relationEmbeddings(_))
//      val values = epRelOvals.map(oval => energy.value(relationEmbeddings(rel), oval))
//      if (values.size == 0)
//        0.0
//      else
//        opts.neighborhood.value match {
//          case "max" =>
//            values.max
//          case "cbow" =>
//            values.sum / math.min(1.0, values.size)
//          case _ => throw new NotImplementedError(s"${opts.neighborhood.value} is not a valid value for neighborhood")
//        }
//    }
//    else
//      0.0
//  }
//
//  // TODO : is this correct ?
//  def averageOvals(ovals: Seq[DiagonalEllipticLike]): DiagonalEllipticLike = {
//    val avgMean = new DenseTensor1(opts.dimension.value, 0.0)
//    val avgVariance = new DenseTensor1(opts.dimension.value, 0.0)
//    ovals.foreach(oval => {
//      avgVariance += oval.variance.value
//      avgMean += oval.mean.value
//    })
//    avgVariance /= ovals.size
//    avgMean /= ovals.size
//    new DiagonalElliptic(this, opts.dimension.value)
//
//  }
//}
//
//// Need to try non-log version for mixture model (tho this is a variational lower bound)
//// lambda gives the "regularization" coefficient -- 1.0 corresponds to proper normalization
//class CBOWDiagonalGaussianLogExpectedLikelihoodEnergy(lambda: Double = 1.0) extends EnergyFunction2[DiagonalEllipticLike, DiagonalEllipticLike] {
//  override def valueAndGradient(v1: DiagonalEllipticLike, v2: DiagonalEllipticLike): (Double, EnergyGradient2) = {
//    val (value, m1grad, c1grad, m2grad, c2grad) = getValueAndGradient(v1.mean.value, v1.variance.value, v2.mean.value, v2.variance.value)
//    val v1grad = new WeightsMap(_.newBlankTensor)
//    val v2grad = new WeightsMap(_.newBlankTensor)
//    v1grad(v1.mean) = m1grad
//    v1grad(v1.variance) = c1grad
//    v2grad(v2.mean) = m2grad
//    v2grad(v2.variance) = c2grad
//    (value, EnergyGradient2(v1grad, v2grad))
//  }
//
//  def getValueAndGradient(m1: Tensor1, c1: Tensor1, m2: Tensor1, c2: Tensor1): (Double, Tensor1, Tensor1, Tensor1, Tensor1) = {
//    val dim = m1.dim1
//    val m1grad = new DenseTensor1(dim)
//    val m2grad = new DenseTensor1(dim)
//    val cgrad = new DenseTensor1(dim)
//    var value = 0.0
//    var i = 0
//    while (i < dim) {
//      val csum = c1(i) + c2(i)
//      val diff = m2(i) - m1(i)
//      val diffSq = diff * diff
//      val ratio = diff / csum
//      value += -0.5 * (diffSq / csum + lambda * math.log(csum))
//      m1grad(i) = ratio
//      m2grad(i) = -ratio
//      cgrad(i) = 0.5 * (diffSq - lambda * csum) / (csum * csum)
//      i += 1
//    }
//    (value, m1grad, cgrad, m2grad, cgrad)
//  }
//}
//
//class CBOWvalMarginTripletExample[Relation](energy: EnergyFunction2[Relation, Relation], posRel: Relation, negRels: Seq[Relation], epRels: Seq[Relation], margin: Double, neighborhood: String) extends Example {
//  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
//    neighborhood match {
//      case "max" =>
//        // find the max score between the pos relation and all relations in eprels
//        val ((posenergy, EnergyGradient2(posingrad, posoutgrad)), otherRel) = epRels.map(rel => (energy.valueAndGradient(rel, posRel), rel)).maxBy(_._1._1)
//        for (negRel <- negRels) {
//          val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negRel)
//          accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
//        }
//      case "cbow" =>
//        for (otherRel <- epRels) {
//          val (posenergy, EnergyGradient2(posingrad, posoutgrad)) = energy.valueAndGradient(otherRel, posRel)
//          for (negRel <- negRels) {
//            val (negenergy, EnergyGradient2(negingrad, negoutgrad)) = energy.valueAndGradient(otherRel, negRel)
//            accumulate(value, gradient, posenergy, posingrad, posoutgrad, negenergy, negingrad, negoutgrad)
//          }
//        }
//      case _ => throw new NotImplementedError(s"$neighborhood is not a valid value for neighborhood")
//    }
//  }
//
//  def accumulate(value: DoubleAccumulator, gradient: WeightsMapAccumulator, posenergy: Double, posingrad: WeightsMap, posoutgrad: WeightsMap, negenergy: Double, negingrad: WeightsMap, negoutgrad: WeightsMap): Unit = {
//    if (posenergy - negenergy < margin) {
//      if (value != null)
//        value.accumulate(posenergy - negenergy - margin)
//      if (gradient != null) {
//        gradient.accumulate(posingrad, 1.0)
//        gradient.accumulate(posoutgrad, 1.0)
//        gradient.accumulate(negingrad, -1.0)
//        gradient.accumulate(negoutgrad, -1.0)
//      }
//    }
//  }
//}
//
//
//
//sealed trait OvalType
//case object DiagonalGaussian extends OvalType
//case object DiagonalCauchy extends OvalType
//case object DiagonalNull extends OvalType
//case object SphericalGaussian extends OvalType
//case object SphericalCauchy extends OvalType
//
//trait EnergyFunction2[V1, V2] {
//  def valueAndGradient(v1: V1, v2: V2): (Double, EnergyGradient2)
//  def value(v1: V1, v2: V2): Double = valueAndGradient(v1, v2)._1
//}
//trait EnergyFunction3[V1, V2, V3] {
//  def valueAndGradient(v1: V1, v2: V2, v3: V3): (Double, EnergyGradient3)
//  def value(v1: V1, v2: V2, v3: V3): Double = valueAndGradient(v1, v2, v3)._1
//}
//
//case class EnergyGradient2(v1grad: WeightsMap, v2grad: WeightsMap)
//case class EnergyGradient3(v1grad: WeightsMap, v2grad: WeightsMap, v3grad: WeightsMap)
//
//trait DiagonalEllipticLike {
//  def variance: Weights1
//  def mean: Weights1
//}
//
//class DiagonalElliptic(params: Parameters, dims: Int)(implicit r: Random) extends DiagonalEllipticLike {
//  var mean = params.Weights(new DenseTensor1(dims))
//  mean.value := Array.fill(dims)(r.nextDouble() / dims / 10 - 0.5 / dims / 10)
//  var variance = params.Weights(new DenseTensor1(dims))
//  variance.value := Array.fill(dims)(1.0)
//}
//
//class DiagonalEllipticPrecomputed(params: Parameters, __mean: DenseTensor1, __variance: DenseTensor1)(implicit r: Random) extends DiagonalEllipticLike {
//  val mean = params.Weights(__mean)
//  val variance = params.Weights(__variance)
//}
//
//trait SphericalEllipticLike {
//  // this should be 1-dimensional with variance in 0th position
//  def variance: Weights1
//  def mean: Weights1
//}
