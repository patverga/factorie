package cc.factorie.epistemodb.tac

import cc.factorie.epistemodb._
import scala.util.Random
import scala.Some
import com.mongodb.{DB, MongoClient}

class TrainTestEventOptions extends cc.factorie.util.DefaultCmdOptions {
  val tacData = new CmdOption("matrix", "", "FILE", "tab separated file with training data")
  val dim = new CmdOption("dim", 100, "INT", "dimensionality of data")
  val stepsize = new CmdOption("stepsize", 0.1, "DOUBLE", "step size")
  val regularizer = new CmdOption("regularizer", 0.01, "DOUBLE", "regularizer")
  val testCols = new CmdOption("test-columns", "", "FILE", "file with test columns, line-by-line")
  val nnzTest = new CmdOption("nnz-test", 0.1, "DOUBLE", "non-zero ratio (within test columns) to be used for held out testing")
  val pruning = new CmdOption("pruning", 2, "INT", "pruning threshold: 0: only graph component selection; >=1: pruning with thresholds")
  val prunedMatrixOut = new CmdOption("matrix-out", "", "FILE", "Tab separated file with pruned data, to be written out.")
  val patternsOut = new CmdOption("patterns-out", "", "FILE", "Top-scored columns, for test columns.")
  val patternsThreshold = new CmdOption("patterns-threshold", 0.5, "DOUBLE", "Threshold for writing out patterns.")

  //val mongoHost = new CmdOption("mongo-host","localhost","STRING","host with running mongo db")
  //val mongoPort = new CmdOption("mongo-port", 27017, "INT", "port mongo db is running on")
  val dbname = new CmdOption("db-name", "event_tmp", "STRING", "name of mongo db to write data into")
}

/**
 * Created by beroth on 3/18/15.
 */
object TrainTestEvent {
  val opts = new TrainTestEventOptions


  def readColSetFromFile(filename: String): Set[String] = {
      scala.io.Source.fromFile(filename).getLines.toSet
  }

  def main(args: Array[String]) : Unit = {
    opts.parse(args)

    val testCols: Set[String] = readColSetFromFile(opts.testCols.value)

    println("Number of test rows: " + testCols.size)

    //val mongoClient = new MongoClient( opts.mongoHost.value , opts.mongoPort.value )
    //val db:DB = mongoClient.getDB( opts.dbname.value )
    //db.dropDatabase()

    val tReadStart = System.currentTimeMillis
//    val kb = StringStringKBMatrix.fromTsvMongoBacked(db, opts.tacData.value, 1).prune(2,1)
    val kb = StringStringKBMatrix.fromTsv(opts.tacData.value, 1).prune(opts.pruning.value, opts.pruning.value)
    val tRead = (System.currentTimeMillis - tReadStart)/1000.0
    println(f"Reading from file and pruning took $tRead%.2f s")

    println("Stats:")
    println("Num Rows:" + kb.numRows())
    println("Num Cols:" + kb.numCols())
    println("Num cells:" + kb.nnz())

    println("Number of potential test cells:" + kb.nnzForCols(testCols))

    if (!opts.prunedMatrixOut.value.isEmpty) {
      kb.writeToTsvFile(opts.prunedMatrixOut.value)
    }

    val random = new Random(0)
    val numDev = 0
    val numTest = (kb.nnzForCols(testCols) * opts.nnzTest.value).toInt

    val (trainKb, devKb, testKb) = kb.randomTestSplit(numDev, numTest, None, Some(testCols), random)

    val model = UniversalSchemaModel.randomModel(kb.numRows(), kb.numCols(), opts.dim.value, random)

    val trainer = new RegularizedBprUniversalSchemaTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
        trainKb.matrix, model, random)

    var result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
    println("Initial MAP: " + Evaluator.meanAveragePrecision(result))

    trainer.train(10)

    result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
    println("MAP after 10 iterations: " + Evaluator.meanAveragePrecision(result))

    trainer.train(40)

    //result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
    //println("MAP after 50 iterations: " + Evaluator.meanAveragePrecision(result))

    trainer.train(50)

    result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
    println("MAP after 100 iterations: " + Evaluator.meanAveragePrecision(result))

    //trainer.train(100)

    //result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
    //println("MAP after 200 iterations: " + Evaluator.meanAveragePrecision(result))

    // TODO:
    //val thresholds = Evaluator.tuneThresholdsMicroAvgF1Score(result)
    kb.writeTopPatterns(testCols, model, opts.patternsThreshold.value, opts.patternsOut.value)

    //db.dropDatabase()
  }
}
