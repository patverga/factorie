package cc.factorie.epistemodb

import com.mongodb.DB
import cc.factorie.la.{Tensor, Tensor1, DenseTensor1}
import cc.factorie.optimize.OptimizableObjectives.UnivariateLinkFunction
import cc.factorie.optimize.{UnivariateOptimizableObjective, OptimizableObjectives}
import scala.collection.mutable
import scala.util.Random
import cc.factorie.model.{Parameters, Weights}
import com.google.common.collect.BiMap
import scala.collection.JavaConversions._


abstract class MatrixModel {
  def similarity01(row: Int, col: Int): Double

  /**
   * Compute test scores for all cells that are not positive entries in the train and development sets.
   * These cells form the basis for evaluation. Additionally to this score, a indicator whether the scored cell is
   * positive in the test data is output.
   *
   * The positive test matrix cells, and all negative cells (from both training and test matrix) form the basis for the
   * test scores.
   *
   * The test scores can be passed to the Evaluator object, to compute evaluation measures such as mean average
   * precision and f1 score.
   */
  def similaritiesAndLabels(trainDevMatrix: CoocMatrix, testMatrix: CoocMatrix, testCols: Option[Set[Int]] = None):
  Map[Int, Seq[(Double, Boolean)]] = {
    val columns = testCols match {
      case Some(cols) => cols
      case None => testMatrix.nonZeroCols()
    }

    columns.par.map(col => {
      val scores = {for (row <- (0 until testMatrix.numRows());
                         if trainDevMatrix.get(row, col) == 0) yield {
        val sim = similarity01(row, col)
        val isTrueTest = testMatrix.get(row, col) != 0
        (sim, isTrueTest)
      }}.toSeq
      (col, scores)
    }).toMap.seq
  }


  /**
   * Compute uschema scores for all cells in test (target) Columns.
   * Rows that contain an entry for the specific target column are ignored by default.
   */
  def colToRowToScore(matrix: CoocMatrix, columns: Set[Int], ignoreObservedCells : Boolean = true):
  Map[Int, Seq[(Int, Double)]] = {
    columns.par.map(col => {
      val scores = rowsToScore(matrix, col, ignoreObservedCells)
      (col, scores)
    }).toMap.seq
  }

  def rowsToScore(matrix: CoocMatrix, col: Int, ignoreObservedCells : Boolean = true):
  Seq[(Int, Double)] = {
    val scores = {for (row <- (0 until matrix.numRows());
                       if !ignoreObservedCells || matrix.get(row, col) == 0) yield {
      val sim = similarity01(row, col)
      (row, sim)
    }}.toSeq
    scores
  }


  def similaritiesAndLables(testMatrix: CoocMatrix, cells: Map[Int, Seq[Int]]):
  Map[Int, Seq[(Double, Boolean)]] = {
    {for (col <- cells.keys) yield {
      val scores = {for (row <- cells.get(col).get) yield {
        val sim = similarity01(row, col)
        val isTrueTest = testMatrix.get(row, col) != 0
        (sim, isTrueTest)
      }}.toSeq
      (col, scores)
    }}.toMap
  }

  def getScoredColumns(v: DenseTensor1): Iterable[(Int, Double)] = ???

  def getScoredRows(v: DenseTensor1): Iterable[(Int, Double)] = ???
}


/**
 * Created by beroth on 2/18/15.
 */
class UniversalSchemaModel(val rowVectors: IndexedSeq[DenseTensor1], val colVectors: IndexedSeq[DenseTensor1]) extends MatrixModel {
  //val objective: UnivariateOptimizableObjective[Int] = OptimizableObjectives.logBinary
  //val linkFunction: UnivariateLinkFunction = OptimizableObjectives.logisticLinkFunction
  //def predict(rowIdx: Int, colIdx: Int) = linkFunction(score(i,j))

  def similarity(vec1: DenseTensor1, vec2: DenseTensor1): Double = vec1.cosineSimilarity(vec2)
  // cosine similarity normalized to lie between 0 and one
  def similarity01(vec1: DenseTensor1, vec2: DenseTensor1): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

  def similarity01(row: Int, col: Int): Double = {
    val rowVec = rowVectors(row)
    val colVec = colVectors(col)
    similarity01(rowVec, colVec)
  }

  def score(rowIdx: Int, colIdx: Int): Double = rowVectors(rowIdx).dot(colVectors(colIdx))


  def columnToExpectedRankingGain(matrix: CoocMatrix, targetColumn: Int, antecedentMinFreq: Int): Map[Int,Double] = {
    val colToExpectedGain = new mutable.HashMap[Int, Double]()
    matrix.colToRows.foreach(colEntries => {
      val antecedent = colEntries._1
      val antecendentRows = colEntries._2
      if (antecendentRows.size >= antecedentMinFreq) {
        val targetScores: Seq[(Int, Double)] = rowsToScore(matrix, targetColumn, ignoreObservedCells = true)
        val targetAndRerankedScores: Seq[(Double, Double)] = targetScores.map(rowScore => {
          val row = rowScore._1
          val oldScore = rowScore._2
          val antecedentHasEntry = antecendentRows.contains(row)
          // Since ignoreObservedCells = true, target has no entry
          // Increase the value for the target column here => annotated cells get ranked higher.
          val newScore = if (antecedentHasEntry) oldScore + 1.0 else oldScore
          (oldScore, newScore)
        })
        val ranks = Evaluator.ranksFromValues(targetAndRerankedScores)
        val rankingDifference = 0.5 + Evaluator.spearmansRankCorrelation(ranks) / 2.0
        // TODO: sigmoid
        val pCorrect = similarity01(colVectors(antecedent), colVectors(targetColumn))
        val expectedGain = pCorrect * rankingDifference
        colToExpectedGain.put(antecedent, expectedGain)
      }
    })
    colToExpectedGain.toMap
  }



/*
  def columnToScore(matrix: CoocMatrix, targetColumn: Int, antecedentMinFreq: Int): Map[Int,Double] = {
    val colToExpectedGain = new mutable.HashMap[Int, Double]()
    matrix.colToRows.filterNot(_ == targetColumn).foreach(colEntries => {
      val antecedent = colEntries._1
      val antecendentRows = colEntries._2
      if (antecedent != targetColumn && antecendentRows.size >= antecedentMinFreq) {
        val pCorrect = similarity01(colVectors(antecedent), colVectors(targetColumn))
        colToExpectedGain.put(antecedent, pCorrect)
      }
    })
    colToExpectedGain.toMap
  }

  def columnToFreq(matrix: CoocMatrix, targetColumn: Int, antecedentMinFreq: Int): Map[Int,Double] = {
    val colToExpectedGain = new mutable.HashMap[Int, Double]()
    matrix.colToRows.foreach(colEntries => {
      val antecedent = colEntries._1
      val antecendentRows = colEntries._2
      if (antecedent != targetColumn && antecendentRows.size >= antecedentMinFreq) {
        colToExpectedGain.put(antecedent, antecendentRows.size)
      }
    })
    colToExpectedGain.toMap
  }
*/



  override def getScoredColumns(v: DenseTensor1): Iterable[(Int, Double)] = {
    colVectors.indices.map(i => (i, similarity01(v, colVectors(i)) ))
  }

  override def getScoredRows(v: DenseTensor1): Iterable[(Int, Double)] = {
    throw new UnsupportedOperationException
  }

  def writeToMongo(mongoDb: DB, dropCollections: Boolean = true) {
    throw new UnsupportedOperationException
  }

}
object UniversalSchemaModel {
  type Row = (Int, mutable.HashMap[Int, Double])

  def fromMongo(mongoDb: DB): UniversalSchemaModel = {
    throw new UnsupportedOperationException
  }

  def randomModel(numRows: Int, numCols:Int, dim: Int, random: Random = new Random(0)): UniversalSchemaModel = {
    val scale = 1.0 / dim
    def initVector(): Array[Double] = Array.fill[Double](dim)(scale * random.nextGaussian())
    //def initVector(i: Int): Array[Double] = Array.fill[Double](latentDimensionality)(2*random.nextDouble() - 1.0)
    val rowVectors = (0 until numRows).map(i => new DenseTensor1(initVector))
    val colVectors = (0 until numCols).map(i => new DenseTensor1(initVector))
    new UniversalSchemaModel(rowVectors, colVectors)
  }

  def calculateProb(theta: Double): Double = {
    1.0 / (1.0 + math.exp(-theta))
  }
}


//class TransEModel(val entityVectors: IndexedSeq[DenseTensor1], val colVectors: IndexedSeq[DenseTensor1], rowToEnts: Int => (Int, Int))
class TransEModel(__entityVectors: IndexedSeq[DenseTensor1], __colVectors: IndexedSeq[DenseTensor1], val rowToEnts: Int => (Int, Int), val numEnts : Int)
extends MatrixModel with Parameters {

  val entityVectors : IndexedSeq[Weights] = __entityVectors.map(this.Weights(_))
  val colVectors : IndexedSeq[Weights] = __colVectors.map(this.Weights(_))

  def gradient(e1: Int, e2:Int, col: Int): Tensor = {
    val e1vec = entityVectors(e1).value
    val e2vec = entityVectors(e2).value
    val colVec = colVectors(col).value
    e2vec - e1vec - colVec
  }

  def similarity01(e1: Int, e2:Int, col: Int) : Double = {
//    1.0 - gradient(e1, e2, col).twoNorm
//    -gradient(e1, e2, col).twoNorm
    val e1vec = entityVectors(e1).value
    val e2vec = entityVectors(e2).value
    val colVec = colVectors(col).value
    -(e1vec + colVec - e2vec).twoNorm
  }

  def similarity01(row: Int, col: Int) : Double = {
    val ents = rowToEnts(row)
    similarity01(ents._1, ents._2, col)
  }

}


object TransEModel {
  def randomModel(numCols:Int, entityMap: BiMap[Int, (Int, Int)], dim: Int, random: Random = new Random(0)): TransEModel = {
    val scale = 1.0 / dim
    def initVector(): Array[Double] = Array.fill[Double](dim)(scale * random.nextGaussian())
    //def initVector(i: Int): Array[Double] = Array.fill[Double](latentDimensionality)(2*random.nextDouble() - 1.0)

    // Get maximums of value tuples.
    val numEnts = entityMap.map(kv => {
      math.max(kv._2._1, kv._2._2)
    }).max+1

    val entVectors = (0 until numEnts).map(i => new DenseTensor1(initVector))
    val colVectors = (0 until numCols).map(i => new DenseTensor1(initVector))

    new TransEModel(entVectors, colVectors, entityMap, numEnts)
  }
}


class UniversalSchemaAdaGradModel(__rowVectors: IndexedSeq[DenseTensor1], __colVectors: IndexedSeq[DenseTensor1])
  extends MatrixModel with Parameters {

  val rowVectors : IndexedSeq[Weights] = __rowVectors.map(this.Weights(_))
  val colVectors : IndexedSeq[Weights] = __colVectors.map(this.Weights(_))

  def similarity(vec1: Tensor, vec2: Tensor): Double = vec1.cosineSimilarity(vec2)
  // cosine similarity normalized to lie between 0 and one
  def similarity01(vec1: Tensor, vec2: Tensor): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

  def similarity01(row: Int, col: Int): Double = {
    val rowVec = rowVectors(row).value
    val colVec = colVectors(col).value
    similarity01(rowVec, colVec)
  }

  def score(rowIdx: Int, colIdx: Int): Double = rowVectors(rowIdx).value.dot(colVectors(colIdx).value)

  def cosSimilarity01(vec1: Tensor, vec2: Tensor): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

}
object UniversalSchemaAdaGradModel{
  def randomModel(numRows: Int, numCols:Int, dim: Int, random: Random = new Random(0)): UniversalSchemaAdaGradModel = {
    val scale = 1.0 / dim
    def initVector(): Array[Double] = Array.fill[Double](dim)(scale * random.nextGaussian())
    //def initVector(i: Int): Array[Double] = Array.fill[Double](latentDimensionality)(2*random.nextDouble() - 1.0)
    val rowVectors = (0 until numRows).map(i => new DenseTensor1(initVector()))
    val colVectors = (0 until numCols).map(i => new DenseTensor1(initVector()))
    new UniversalSchemaAdaGradModel(rowVectors, colVectors)
  }
}

class ColumnAverageModel(val rowToCols : Map[Int, Seq[Int]], __colVectors: IndexedSeq[DenseTensor1], val numCols : Int, val scoreType : String = "cbow")
  extends MatrixModel with Parameters {

  val colVectors : IndexedSeq[Weights] = __colVectors.map(this.Weights(_))

  def similarity(vec1: Tensor, vec2: Tensor): Double = vec1.cosineSimilarity(vec2)
  // cosine similarity normalized to lie between 0 and one
  def similarity01(vec1: Tensor, vec2: Tensor): Double =
    (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

  def similarity01(row: Int, col: Int): Double = {
    score(colVectors(col).value, rowToCols(row).map(colVectors(_).value))
  }

  def score(targetCol : Tensor, otherCols : Seq[Tensor]): Double = {
    val values = otherCols.map(otherCol => targetCol.dot(otherCol))
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

  def scoreAndMax(targetCol : Tensor, otherCols : Seq[Tensor]): (Double, Int) = {
    val values = otherCols.map(otherCol => targetCol.dot(otherCol))
    if (values.isEmpty)
      (0.0, -1)
    else
      values.zipWithIndex.maxBy(_._1)
  }

  def cosSimilarity01(vec1: Tensor, vec2: Tensor): Double = (1.0 + vec1.cosineSimilarity(vec2)) / 2.0

}
object ColumnAverageModel{
  def randomModel(rowToCols: Map[Int, Seq[Int]], numCols:Int, dim: Int, random: Random = new Random(0), scoreType : String = "cbow")
  : ColumnAverageModel = {
    val scale = 1.0 / dim
    def initVector(): Array[Double] = Array.fill[Double](dim)(scale * random.nextGaussian())
    //def initVector(i: Int): Array[Double] = Array.fill[Double](latentDimensionality)(2*random.nextDouble() - 1.0)
    val colVectors = (0 until numCols).map(i => new DenseTensor1(initVector))
    new ColumnAverageModel(rowToCols, colVectors, numCols, scoreType)
  }
}