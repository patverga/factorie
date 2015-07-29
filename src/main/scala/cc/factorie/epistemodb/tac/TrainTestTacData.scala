package cc.factorie.epistemodb.tac

import java.io.{File, PrintWriter}

import cc.factorie.la.{DenseTensor1, DenseTensor}
import com.google.common.collect.HashBiMap

import scala.io.Source
import scala.util.Random
import cc.factorie.epistemodb._

/**
 * Created by beroth on 2/23/15.
 */

class TrainTestTacDataOptions extends cc.factorie.util.DefaultCmdOptions {
  val tacData = new CmdOption("tac-data", "", "FILE", "tab separated file with TAC training data")
  val dim = new CmdOption("dim", 100, "INT", "dimensionality of data")
  val stepsize = new CmdOption("stepsize", 0.1, "DOUBLE", "step size")
  val maxNorm =  new CmdOption("max-norm", 3.0, "DOUBLE", "maximum l2-norm for vectors")
  val margin =  new CmdOption("margin", 1.0, "DOUBLE", "size of margin to use for training")
  val useMaxNorm =  new CmdOption("use-max-norm", true, "BOOLEAN", "whether to use maximum l2-norm for vectors")
  val regularizer = new CmdOption("regularizer", 0.01, "DOUBLE", "regularizer")
  val patternsOut = new CmdOption("patterns-out", "", "FILE", "Top-scored columns, for test columns.")
  val exportData = new CmdOption("export-data", false, "BOOLEAN", "export data and embeddings.")
  val loadModel = new CmdOption("load-model", false, "BOOLEAN", "load embeddings from file")
}


class
TrainTestTacData {

  val opts = new TrainTestTacDataOptions

  val testCols = Set("org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:date_dissolved",
    "org:date_founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees_members",
    "org:parents",
    "org:political_religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members_employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_or_member_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:statesorprovinces_of_residence",
    "per:title")

  def evaluate(model : MatrixModel, trainer : BprTrainer, trainKb : CoocMatrix, testKb: CoocMatrix) : Unit = {
    var result = model.similaritiesAndLabels(trainKb, testKb)
    println("Initial MAP: " + Evaluator.meanAveragePrecision(result))

    trainer.train(10)

    result = model.similaritiesAndLabels(trainKb, testKb)
    println("MAP after 10 iterations: " + Evaluator.meanAveragePrecision(result))

    trainer.train(40)

    result = model.similaritiesAndLabels(trainKb, testKb)
    println("MAP after 50 iterations: " + Evaluator.meanAveragePrecision(result))

    trainer.train(50)

    result = model.similaritiesAndLabels(trainKb, testKb)
    println("MAP after 100 iterations: " + Evaluator.meanAveragePrecision(result))

    trainer.train(100)

    result = model.similaritiesAndLabels(trainKb, testKb)
    println("MAP after 200 iterations: " + Evaluator.meanAveragePrecision(result))
  }

  def exportTrainMatrix(trainKb : StringStringKBMatrix): Unit ={
    // non zero row / col cells as ints
    var writer = new PrintWriter("train.mtx")
    trainKb.matrix.getNnzCells().foreach(t => writer.println(s"${t._1}\t${t._2}"))
    writer.close()
    // row -> string map
    writer = new PrintWriter("row-map.tsv")
    trainKb.__rowMap.keyIterator.foreach(key => writer.println(s"${trainKb.__rowMap.keyToIndex(key)}\t$key"))
    writer.close()
    // col -> string map
    writer = new PrintWriter("col-map.tsv")
    trainKb.__colMap.keyIterator.foreach(key => writer.println(s"${trainKb.__colMap.keyToIndex(key)}\t$key"))
    writer.close()
  }

  def exportTestMatrix(trainDevMatrix: StringStringKBMatrix, testMatrix: StringStringKBMatrix, testCols: Option[Set[Int]] = None) ={
    val columns = testCols match {
      case Some(cols) => cols
      case None => testMatrix.matrix.nonZeroCols()
    }
    new File("test-mtx").mkdir()
    columns.par.foreach(col => {
      val writer = new PrintWriter(s"test-mtx/$col-test.mtx")
      for (row <- 0 until testMatrix.numRows();
           if trainDevMatrix.matrix.get(row, col) == 0) yield {
        val isTrueTest = testMatrix.matrix.get(row, col) != 0
        // convert test row / col to train row/col
        writer.write(s"$row\t$col\t${if (isTrueTest) 1 else 0}\n")
      }
      writer.close()
    })
  }

  def exportEmbeddings(model : UniversalSchemaModel): Unit ={
    var writer = new PrintWriter("col.embeddings")
    model.colVectors.zipWithIndex.foreach{ case (vector, i) => writer.println(s"${vector.mkString(" ")}") }
    writer.close()
    writer = new PrintWriter("row.embeddings")
    model.rowVectors.zipWithIndex.foreach{ case (vector, i) => writer.println(s"${vector.mkString(" ")}") }
    writer.close()
  }

  def loadEmbeddings(): (IndexedSeq[DenseTensor1], IndexedSeq[DenseTensor1]) = {
    val rowEmbeddings = Source.fromFile("row.embeddings").getLines().map(line => {
      new DenseTensor1(line.split(" ").map(_.toDouble))
    }).toIndexedSeq
    val colEmbeddings = Source.fromFile("col.embeddings").getLines().map(line => {
      new DenseTensor1(line.split(" ").map(_.toDouble))
    }).toIndexedSeq

    (rowEmbeddings, colEmbeddings)
  }
}


object ExportData  extends TrainTestTacData {
  def main(args: Array[String]): Unit = {
    opts.parse(args)

    val kb = StringStringKBMatrix.fromTsv(opts.tacData.value).prune(2, 1)
    println("Stats:")
    println("Num Rows:" + kb.numRows())
    println("Num Cols:" + kb.numCols())
    println("Num cells:" + kb.nnz())

    val random = new Random(0)
    val numDev = 0
    val numTest = 10000
    val (trainKb, _, testKb) = kb.randomTestSplit(numDev, numTest, None, Some(testCols), random)

    // export training matrix
    exportTrainMatrix(trainKb)
    // export test matrix
    exportTestMatrix(trainKb, testKb)
  }
}

object TrainTestTacData  extends TrainTestTacData{
  def main(args: Array[String]) : Unit = {
      opts.parse(args)

      val tReadStart = System.currentTimeMillis
//      val kb = EntityRelationKBMatrix.fromTsv(opts.tacData.value).prune(2,1)
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

      val model = if (opts.loadModel.value) {
          val (rowEmbeddings, colEmbeddings) = loadEmbeddings()
          new UniversalSchemaModel(rowEmbeddings, colEmbeddings)
        }
        else UniversalSchemaModel.randomModel(kb.numRows(), kb.numCols(), opts.dim.value, random)

      val trainer = if(opts.useMaxNorm.value) {
        println("use norm constraint")
        new NormConstrainedBprUniversalSchemaTrainer(opts.maxNorm.value, opts.stepsize.value, opts.dim.value,
          trainKb.matrix, model, random)
      } else {
        println("use regularization")
        new RegularizedBprUniversalSchemaTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
          trainKb.matrix, model, random)
      }

      evaluate(model, trainer, trainKb.matrix, testKb.matrix)

      if (!opts.patternsOut.value.isEmpty) {
        kb.writeTopPatterns(testCols, model, 0.5, opts.patternsOut.value)
      }
      if (opts.exportData.value) {
        exportTrainMatrix(trainKb)
        exportTestMatrix(trainKb, testKb)
        exportEmbeddings(model)
      }
    }
}

object TrainTestTacDataAdaGrad  extends TrainTestTacData{
  def main(args: Array[String]) : Unit = {
    opts.parse(args)

    val tReadStart = System.currentTimeMillis
    //      val kb = EntityRelationKBMatrix.fromTsv(opts.tacData.value).prune(2,1)
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

    val model = UniversalSchemaAdaGradModel.randomModel(kb.numRows(), kb.numCols(), opts.dim.value, random)

    val trainer = new AdaGradUniversalSchemaTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
      opts.margin.value, trainKb.matrix, model, random)

    evaluate(model, trainer, trainKb.matrix, testKb.matrix)
  }
}

object TrainTestTacDataColAverage extends TrainTestTacData{
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
    val model = ColumnAverageModel.randomModel(rowToCols, kb.numCols(), opts.dim.value, random)
    val trainer = new ColumnAverageTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
      opts.margin.value, trainKb.matrix, model, random)

    evaluate(model, trainer, trainKb.matrix, testKb.matrix)

    //    if (!opts.patternsOut.value.isEmpty) {
//      kb.writeTopPatterns(testCols, model, 0.5, opts.patternsOut.value)
//    }

  }
}

object TrainTestTacDataTransE extends TrainTestTacData{
  def main(args: Array[String]) : Unit = {
    opts.parse(args)

    val tReadStart = System.currentTimeMillis
    //      val kb = EntityRelationKBMatrix.fromTsv(opts.tacData.value).prune(2,1)
    val kb = TransEKBMatrix.fromTsv(opts.tacData.value)//.pruneWithEntities(2,1)
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

    val rowToEnts = kb.matrix.rowEntsBimap

    val model = TransEModel.randomModel(kb.numCols(), rowToEnts, opts.dim.value, random)
    val trainer = new TransETrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value, opts.margin.value, trainKb.matrix, model, random)

    evaluate(model, trainer, trainKb.matrix, testKb.matrix)


    //    if (!opts.patternsOut.value.isEmpty) {
    //      kb.writeTopPatterns(testCols, model, 0.5, opts.patternsOut.value)
    //    }

  }
}