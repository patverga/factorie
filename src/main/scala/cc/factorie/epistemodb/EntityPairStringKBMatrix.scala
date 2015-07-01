package cc.factorie.epistemodb

import java.util

import com.google.common.collect.{BiMap, HashBiMap}
import com.mongodb._
import java.io.{Writer, File}
import scala.Predef._
import scala.Some

/**
 * Created by beroth on 2/6/15.
 */
/**
 * Holds a knowledge-base with an underlying matrix.
 * I.e. additionally to matrix information, it also stores information about entities, relations etc.
 */



//class TransEKBMatrix {
//  val coocMatrix = new EntityPairStringKBMatrix
//
//  val entityIndexMap = new MemoryIndexMap[String]
//
//  def listEntityIndexMap: Map[Int, (Int, Int)] = {
//    coocMatrix.__rowMap.keyIterator.map(key => {
//      entityIndexMap.add(key.e1)
//      entityIndexMap.add(key.e2)
//      val id1 = entityIndexMap.keyToIndex(key.e1)
//      val id2 = entityIndexMap.keyToIndex(key.e2)
//      val pairId = coocMatrix.__rowMap.keyToIndex(key)
//      (pairId -> (id1, id2))
//    }).toMap
//  }
//}

object TransEKBMatrix {
  private def entitiesAndRelFromLine(line: String, colsPerEnt: Int): (EntityPair, String, Double) = {
    val parts = line.split("\t")
    if (parts.length < 2 * colsPerEnt + 2) {
      throw new IllegalArgumentException("Line specifying matrix cell needs columns for 2 entities, relation, and count.")
    }
    val e1: String = parts.slice(0, colsPerEnt).mkString("\t")
    val e2: String = parts.slice(colsPerEnt, 2 * colsPerEnt).mkString("\t")
    val rel: String = parts.slice(2 * colsPerEnt, parts.length - 1).mkString("\t")
    val cellVal: Double = parts(parts.length - 1).toDouble
    (EntityPair(e1, e2), rel, cellVal)
  }

  // Loads a matrix from a tab-separated file
  def fromTsv(filename: String, colsPerEnt: Int = 2): TransEKBMatrix = {
    val kb = new TransEKBMatrix()
    val tReadStart = System.currentTimeMillis
    var numRead = 0
    scala.io.Source.fromFile(filename).getLines.foreach(line => {
      val (ep, rel, cellVal) = entitiesAndRelFromLine(line, colsPerEnt)
      kb.set(ep, rel, cellVal)

      numRead += 1
      if (numRead % 100000 == 0) {
        val tRead = numRead / (System.currentTimeMillis - tReadStart).toDouble
        println(f"cells read per millisecond: $tRead%.4f")
        println(f"Last row: (${ep.e1}s, ${ep.e2}s)")
        println(f"Last column: (${rel}s)")
        println(f"Last cell value: $cellVal%.4f")
      }
    })
    println(kb.matrix._numEnts, kb.matrix._numRows, kb.matrix._numCols)
    kb
  }
}

class TransEKBMatrix(val matrix:EntityPairCoocMatrix = new EntityPairCoocMatrix(0,0,0),
                             val __rowMap: MatrixIndexMap[EntityPair] = new EntityPairMemoryMap(collectionPrefix = MongoWritable.ENTITY_ROW_MAP_PREFIX),
                             val __colMap: MatrixIndexMap[String] = new StringMemoryIndexMap(collectionPrefix = MongoWritable.ENTITY_COL_MAP_PREFIX),
                             val __entityMap: MatrixIndexMap[String] = new StringMemoryIndexMap(collectionPrefix = "ENITIIES")
                              ) extends KBMatrix[TransEKBMatrix, EntityPair, String]  {


  override def set(ep: EntityPair, rel: String, cellVal: Double) {
    __entityMap.add(ep.e1)
    __entityMap.add(ep.e2)
    val rowNr = __rowMap.add(ep)
    val colNr = __colMap.add(rel)
    matrix.set(__entityMap.keyToIndex(ep.e1), __entityMap.keyToIndex(ep.e2), __colMap.keyToIndex(rel), cellVal)

  }

  def cloneWithNewCells(cells: CoocMatrix): TransEKBMatrix = {
    val epMatrix = new EntityPairCoocMatrix(numRows, numCols, __entityMap.size)
    epMatrix.rowEntsBimap = this.matrix.rowEntsBimap
    epMatrix.rowToColAndVal = cells.rowToColAndVal
    epMatrix.colToRows = cells.colToRows
    new TransEKBMatrix(matrix = epMatrix, __rowMap = this.__rowMap, __colMap = this.__colMap)

  }

  def createEmptyMatrix: TransEKBMatrix = {
    new TransEKBMatrix()
  }

  def pruneWithEntities(tRow: Int = 2, tCol: Int = 2): TransEKBMatrix = {
    val (prunedMatrix, oldToNewRow, oldToNewCol) = matrix.pruneWithEntities(tRow, tCol)
    val newKb: TransEKBMatrix = this.createEmptyMatrix

    val newToOldCol = oldToNewCol.map(_ swap)
    val newToOldRow = oldToNewRow.map(_ swap)

    for (rowNr <- 0 until prunedMatrix.numRows()) {
      for((colNr, cellVal) <- prunedMatrix.getRow(rowNr)) {
        val rowKey = __rowMap.indexToKey(newToOldRow.get(rowNr).get)
        val colKey = __colMap.indexToKey(newToOldCol.get(colNr).get)
        newKb.set(rowKey, colKey, cellVal)
      }
    }
    newKb
  }

}




class EntityPairStringKBMatrix(val matrix:CoocMatrix = new CoocMatrix(0,0),
               val __rowMap: MatrixIndexMap[EntityPair] with MongoWritable = new EntityPairMemoryMap(collectionPrefix = MongoWritable.ENTITY_ROW_MAP_PREFIX),
               val __colMap: MatrixIndexMap[String] with MongoWritable = new StringMemoryIndexMap(collectionPrefix = MongoWritable.ENTITY_COL_MAP_PREFIX)
                              ) extends KBMatrix[EntityPairStringKBMatrix, EntityPair, String] with MongoWritable {

  def cloneWithNewCells(cells: CoocMatrix): EntityPairStringKBMatrix = {
    new EntityPairStringKBMatrix(matrix = cells, __rowMap = this.__rowMap, __colMap = this.__colMap)
  }

  def createEmptyMatrix(): EntityPairStringKBMatrix = {
    new EntityPairStringKBMatrix()
  }

  def writeToMongo(mongoDb: DB) {
    matrix.writeToMongo(mongoDb)
    __rowMap.writeToMongo(mongoDb)
    __colMap.writeToMongo(mongoDb)
  }

  def populateFromMongo(mongoDb: DB) {
    matrix.populateFromMongo(mongoDb)
    __rowMap.populateFromMongo(mongoDb)
    __colMap.populateFromMongo(mongoDb)
  }
}


class StringStringKBMatrix(val matrix:CoocMatrix = new CoocMatrix(0,0),
                             val __rowMap: MatrixIndexMap[String] with MongoWritable = new StringMemoryIndexMap(collectionPrefix = MongoWritable.ENTITY_ROW_MAP_PREFIX),
                             val __colMap: MatrixIndexMap[String] with MongoWritable = new StringMemoryIndexMap(collectionPrefix = MongoWritable.ENTITY_COL_MAP_PREFIX)
                              ) extends KBMatrix[StringStringKBMatrix, String, String] with MongoWritable {

  def cloneWithNewCells(cells: CoocMatrix): StringStringKBMatrix = {
    new StringStringKBMatrix(matrix = cells, __rowMap = this.__rowMap, __colMap = this.__colMap)
  }

  def createEmptyMatrix(): StringStringKBMatrix = {
    new StringStringKBMatrix()
  }

  def writeToMongo(mongoDb: DB) {
    matrix.writeToMongo(mongoDb)
    __rowMap.writeToMongo(mongoDb)
    __colMap.writeToMongo(mongoDb)
  }

  def populateFromMongo(mongoDb: DB) {
    matrix.populateFromMongo(mongoDb)
    __rowMap.populateFromMongo(mongoDb)
    __colMap.populateFromMongo(mongoDb)
  }

  def writeToTsvFile(filename: String) {
    val pw = new java.io.PrintWriter(new File(filename))
    for (rowStr <- this.__rowMap.keyIterator) {
      for (colStr <- this.getColsForRow(rowStr)) {
        val count = this.get(rowStr, colStr)
        pw.println(f"$rowStr%s\t$colStr%s\t$count%.4f")
      }
    }
    pw.close()
  }

  def writeTopPatterns(testCols: Set[String], model: UniversalSchemaModel, threshold: Double, filename: String) {
    val pw = new java.io.PrintWriter(new File(filename))
    for (testColStr <- testCols;
    if (__colMap.containsKey(testColStr) &&
      matrix.nonZeroCols().contains(__colMap.keyToIndex(testColStr)))
    ) {
      val testColIdx = this.__colMap.keyToIndex(testColStr)
      val testColVec = model.colVectors(testColIdx)
      model.getScoredColumns(testColVec).
        filter(_._2 > threshold).
        map(idxScore => (this.__colMap.indexToKey(idxScore._1), idxScore._2)). // map col index to surface form
        filter(strScore => !testCols.contains(strScore._1)).foreach(strScore => {
        val pattern = strScore._1
        val score = strScore._2
        pw.println(f"$score%.4f\t$testColStr%s\t$pattern%s")
      })
    }
    pw.close()
  }

  def writeColumnEmbeddings(model: UniversalSchemaModel, writer: Writer, constrainTo: Option[Iterable[String]] = None,
                            dontWrite: Set[String] = Set()) {
    val colIds: Iterable[Int] = constrainTo match {
      case Some(ids) => ids.map(id => __colMap.keyToIndex(id))
      case None => Range(0, __colMap.size)
    }
    for (colId <- colIds) {
      val relStr = __colMap.indexToKey(colId)
      if (!dontWrite.contains(relStr)) {
        val vecStr = model.colVectors(colId).mkString(" ")
        writer.write(relStr + "\t" + vecStr + "\n")
      }
    }
  }
}



object StringStringKBMatrix {

  private def entitiesAndRelFromLine(line: String, colsPerEnt:Int): (String, String, Double) = {
    val parts = line.split("\t")
    if (parts.length < 2 * colsPerEnt + 2) {
      throw new IllegalArgumentException("Line specifying matrix cell needs columns for 2 entities, relation, and count.")
    }
    val ep : String = parts.slice(0, 2 * colsPerEnt).mkString("\t")
    val rel : String = parts.slice(2 * colsPerEnt, parts.length - 1).mkString("\t")
    val cellVal : Double = parts(parts.length - 1).toDouble
    (ep, rel, cellVal)
  }

  def fromTsvMongoBacked(mongoDb: DB, filename:String, colsPerEnt:Int = 2) : StringStringKBMatrix = {
    val rowMap = new StringMongoMap(mongoDb = mongoDb, collectionPrefix = MongoWritable.ENTITY_ROW_MAP_PREFIX)
    val colMap = new StringMongoMap(mongoDb = mongoDb, collectionPrefix = MongoWritable.ENTITY_COL_MAP_PREFIX)

    val kb = new StringStringKBMatrix(__rowMap = rowMap, __colMap = colMap)

    val tReadStart = System.currentTimeMillis
    var numRead = 0
    scala.io.Source.fromFile(filename).getLines.foreach(line => {
      val (ep, rel, cellVal) = entitiesAndRelFromLine(line, colsPerEnt)
      kb.set(ep, rel, cellVal)

      numRead += 1
      if (numRead % 100000 == 0) {
        val tRead = numRead / (System.currentTimeMillis - tReadStart).toDouble
        println(f"cells read per millisecond: $tRead%.4f")
        println(f"Last row: (${ep}s)")
        println(f"Last column: (${rel}s)")
        println(f"Last cell value: $cellVal%.4f")
      }
    })
    kb
  }


  def fromTsv(filename:String, colsPerEnt:Int = 2) : StringStringKBMatrix = {
    val kb = new StringStringKBMatrix()

    val tReadStart = System.currentTimeMillis
    var numRead = 0
    scala.io.Source.fromFile(filename).getLines.foreach(line => {
      val (ep, rel, cellVal) = entitiesAndRelFromLine(line, colsPerEnt)
      kb.set(ep, rel, cellVal)

      numRead += 1
      if (numRead % 100000 == 0) {
        val tRead = numRead / (System.currentTimeMillis - tReadStart).toDouble
        println(f"cells read per millisecond: $tRead%.4f")
        println(f"Last row: (${ep}s)")
        println(f"Last column: (${rel}s)")
        println(f"Last cell value: $cellVal%.4f")
      }
    })
    kb
  }
}




object EntityPairStringKBMatrix {

  private def entitiesAndRelFromLine(line: String, colsPerEnt:Int): (EntityPair, String, Double) = {
    val parts = line.split("\t")
    if (parts.length < 2 * colsPerEnt + 2) {
      throw new IllegalArgumentException("Line specifying matrix cell needs columns for 2 entities, relation, and count.")
    }
    val e1 : String = parts.slice(0, colsPerEnt).mkString("\t")
    val e2 : String = parts.slice(colsPerEnt, 2 * colsPerEnt).mkString("\t")
    val rel : String = parts.slice(2 * colsPerEnt, parts.length - 1).mkString("\t")
    val cellVal : Double = parts(parts.length - 1).toDouble
    (EntityPair(e1, e2), rel, cellVal)
  }

  // Loads a matrix from a tab-separated file
  def fromTsv(filename:String, colsPerEnt:Int = 2) : EntityPairStringKBMatrix = {
    val kb = new EntityPairStringKBMatrix()
    val tReadStart = System.currentTimeMillis
    var numRead = 0
    scala.io.Source.fromFile(filename).getLines.foreach(line => {
      val (ep, rel, cellVal) = entitiesAndRelFromLine(line, colsPerEnt)
      kb.set(ep, rel, cellVal)

      numRead += 1
      if (numRead % 100000 == 0) {
        val tRead = numRead / (System.currentTimeMillis - tReadStart).toDouble
        println(f"cells read per millisecond: $tRead%.4f")
        println(f"Last row: (${ep.e1}s, ${ep.e2}s)")
        println(f"Last column: (${rel}s)")
        println(f"Last cell value: $cellVal%.4f")
      }
    })
    kb
  }


  def fromTsvMongoBacked(mongoDb: DB, filename:String, colsPerEnt:Int = 2) : EntityPairStringKBMatrix = {

    val rowMap = new EntityPairMongoMap(mongoDb = mongoDb, collectionPrefix = MongoWritable.ENTITY_ROW_MAP_PREFIX)
    val colMap = new StringMongoMap(mongoDb = mongoDb, collectionPrefix = MongoWritable.ENTITY_COL_MAP_PREFIX)

    val kb = new EntityPairStringKBMatrix(__rowMap = rowMap, __colMap = colMap)

    val tReadStart = System.currentTimeMillis
    var numRead = 0
    scala.io.Source.fromFile(filename).getLines.foreach(line => {
      val (ep, rel, cellVal) = entitiesAndRelFromLine(line, colsPerEnt)
      kb.set(ep, rel, cellVal)

      numRead += 1
      if (numRead % 100000 == 0) {
        val tRead = numRead / (System.currentTimeMillis - tReadStart).toDouble
        println(f"cells read per millisecond: $tRead%.4f")
        println(f"Last row: (${ep.e1}s, ${ep.e2}s)")
        println(f"Last column: (${rel}s)")
        println(f"Last cell value: $cellVal%.4f")
      }
    })
    kb
  }
}
