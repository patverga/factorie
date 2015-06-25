package cc.factorie.epistemodb.activelearning

import scala.util.Random
import cc.factorie.epistemodb._

/**
 * Created by beroth on 2/23/15.
 */

class ActiveLearningOptions extends cc.factorie.util.DefaultCmdOptions {
  val tacData = new CmdOption("tac-data", "", "FILE", "tab separated file with TAC training data")
  val dim = new CmdOption("dim", 100, "INT", "dimensionality of data")
  val stepsize = new CmdOption("stepsize", 0.1, "DOUBLE", "step size")
  val regularizer = new CmdOption("regularizer", 0.01, "DOUBLE", "regularizer")

  val patternsOut = new CmdOption("patterns-out", "", "FILE", "Top-scored columns, for test columns.")
}


object ActiveLearning {

  val opts = new ActiveLearningOptions

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

      val model = UniversalSchemaModel.randomModel(kb.numRows(), kb.numCols(), opts.dim.value, random)

      val trainer = new RegularizedBprUniversalSchemaTrainer(opts.regularizer.value, opts.stepsize.value, opts.dim.value,
          trainKb.matrix, model, random)

      var result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
      println("Initial MAP: " + Evaluator.meanAveragePrecision(result))

      trainer.train(100)

      showAndAnnotate(model, trainKb, testCols, 10)


      result = model.similaritiesAndLabels(trainKb.matrix, testKb.matrix)
      println("MAP after 200 iterations: " + Evaluator.meanAveragePrecision(result))

      if (!opts.patternsOut.value.isEmpty) {
        kb.writeTopPatterns(testCols, model, 0.5, opts.patternsOut.value)
      }

    }

  def askUser(antecedentStr: String, targetStr:String): Boolean = ???

  def showAndAnnotate(model: UniversalSchemaModel, trainKb: StringStringKBMatrix, testCols: Set[String], numAnno: Int) {
    for (targetStr <- testCols) {
      val targetIdx = trainKb.__colMap.keyToIndex(targetStr)
      val bestAntecedents = model.columnToExpectedRankingGain(trainKb.matrix, targetIdx, 10).toSeq.sortBy(-_._2).slice(0,10)
      for (antecedentIdx <- bestAntecedents.map(_._1)) {
        val antecedentStr = trainKb.__colMap.indexToKey(antecedentIdx)
        //val targetStr = trainKb.__colMap.indexToKey(targetIdx)
        val relationHolds = askUser(antecedentStr, targetStr) // TODO: Log output
        if (relationHolds) {
          for(row <- trainKb.matrix.colToRows.get(antecedentIdx).get) {
            // Get all nnz rows for antecedent
            val antecedentVal = trainKb.matrix.get(row, antecedentIdx)
            if (antecedentVal > trainKb.matrix.get(row, targetIdx)) {
              trainKb.matrix.set(row, targetIdx, antecedentVal)
            }
          }
        }
      }
    }
  }



}
