package ge

import java.util.Random

import ge.PSUtils._
import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ps.storage.matrix.{PartitionSourceArray, PartitionSourceMap}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.models.PSMatrix
import com.tencent.angel.ml.matrix.RowType
import com.tencent.angel.ml.matrix.psf.get.base.{GetFunc, GetResult}
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateFunc, VoidResult}
import com.tencent.angel.spark.ml.psf.embedding.NEModelRandomize
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapred.TextOutputFormat
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import samplers.BaseSampler
import ge.basics.{DistributionMeta, HashMapModel}
import  com.tencent.angel.spark.ml.embedding.NEModel
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import com.tencent.angel.spark.ml.psf.embedding.NEModelRandomize.{RandomizePartitionUpdateParam, RandomizeUpdateParam}
import utils.FastMath

abstract class GEModel (sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta], dimension: Int) extends Serializable {
	// init the word model and context model

	def initOneDimModel(size: Int, bandwidth: Float, random: Random): Array[Float] = {
		val array = Array.ofDim[Float](size)
		for (i <- 0 until (array.length)) {
			array(i) = (random.nextFloat() - 0.5.toFloat) * 2 * bandwidth
		}
		array
	}

	/**
	  * save one RDD[HashMap] model.
	  * @param model
	  * @param bcDict
	  * @param output
	  */
	def saveModel(model: RDD[HashMapModel],
				  bcDict: Broadcast[Array[String]], output: String): Unit = {

		// adapted from hadoop saveAsText, only save the source vector
		val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
		val textClassTag = implicitly[ClassTag[Text]]


		val arrayModel = model.mapPartitions(x => {
			// one element in each partition
			val int2String = bcDict.value
			val vertexEntryIter = x.next().toEntryterator()
			val arrayBuffer = new ArrayBuffer[(String, Array[Float])](1000)
			while (vertexEntryIter.hasNext()) {
				val entry = vertexEntryIter.next()
				arrayBuffer.append((int2String(entry.getKey), entry.getValue))
			}
			arrayBuffer.toIterator
		})

		val r = arrayModel.mapPartitions(iter => {
			val text = new Text()
			iter.map(x => {
				text.set(x._1 + " " + x._2.mkString(" "))
				(NullWritable.get(), text)
			})
		})
		RDD.rddToPairRDDFunctions(r)(nullWritableClassTag, textClassTag, null)
			.saveAsHadoopFile[TextOutputFormat[NullWritable, Text]](output)
	}

	def destory(): Unit
	def saveSrcModel(path: String, bcDict: Broadcast[Array[String]]): Unit
	def saveDstModel(path: String, bcDict: Broadcast[Array[String]]): Unit

}

class GEPSModel (sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta],
				 dimension: Int, numNodePerRow: Int, numParts: Int) extends GEModel(sampler, bcMeta, dimension){

	val srcModel: RDD[(Int, HashMapModel)] = {
		val trainSet = sampler.trainset
		val model: RDD[(Int, HashMapModel)] = {
			// should be barrier mode, because data rdd is barrier mode.
			trainSet.mapPartitionsWithIndex((pid, iter) => {
				val srcMeta: Array[Int] = bcMeta.value.srcMeta

				val srcModel: HashMapModel = new HashMapModel()
				val random = new Random(pid)
				for (i <- 0 until (srcMeta.length)) {
					if (srcMeta(i) == pid) {
						srcModel.put(i, initOneDimModel(dimension, 0.5.toFloat / dimension, random))
					}
				}
				Iterator.single((pid, srcModel))
			})
		}
		model
	}

	val dstModel: PSMatrix = {
		val trainSet = sampler.trainset
		val context: PSContext = PSContext.getOrCreate(trainSet.context)
		val numNode = bcMeta.value.srcMeta.length
		val numRow = numNode / numNodePerRow + 1
		val numCol = numNodePerRow * dimension
		val rowsInBlock = numRow / numParts + 1
		val colsInBlock = numCol
		val dstModel = PSMatrix.dense(numRow, numCol, rowsInBlock, colsInBlock, RowType.T_FLOAT_DENSE,
			Map(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS -> classOf[PartitionSourceMap].getName))
		dstModel.psfUpdate(new GERandom(new GERandomParam(dstModel.id, dimension))).get()
		dstModel
	}
	srcModel.setName("srcModel")
	srcModel.cache()

	override def saveSrcModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
		saveModel(srcModel.map(x => x._2), bcDict, path)
	}

	override def saveDstModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
		// to be finished
	}

	override def destory(): Unit ={
		srcModel.unpersist()
		dstModel.destroy()
		PSContext.stop()
	}

}

class GERDDModel (sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta],
				  dimension: Int )extends GEModel(sampler, bcMeta, dimension){

	val srcDstModel: RDD[(Int, (HashMapModel, HashMapModel))] = {

		val trainSet = sampler.trainset
		val model: RDD[(Int, (HashMapModel, HashMapModel))] =
			trainSet.mapPartitionsWithIndex((pid, iter) => {
				val srcMeta: Array[Int] = bcMeta.value.srcMeta
				val dstMeta: Array[Int] = bcMeta.value.dstMeta

				val srcModel: HashMapModel = new HashMapModel()
				val dstModel: HashMapModel = new HashMapModel()
				val random = new Random(pid)
				for (i <- 0 until (srcMeta.length)) {
					if (srcMeta(i) == pid) {
						srcModel.put(i, initOneDimModel(dimension, 0.5.toFloat / dimension, random))
					}
					if (dstMeta(i) == pid) {
						dstModel.put(i, initOneDimModel(dimension, 0.5.toFloat / dimension, random))
					}
				}
				Iterator.single((pid, (srcModel, dstModel)))
			})
		model
	}

	srcDstModel.setName("srcDstModel")
	srcDstModel.cache()

	override def saveSrcModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
		saveModel(srcDstModel.map(x => x._2._1), bcDict, path)
	}

	override def saveDstModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
		saveModel(srcDstModel.map(x => x._2._2), bcDict, path)
	}

	override def destory(): Unit = {
		srcDstModel.unpersist()
	}
}

class GEWORKERModel (sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta],
                 dimension: Int, numNodePerRow: Int, numParts: Int) extends GEModel(sampler, bcMeta, dimension){
  val srcModel: PSMatrix = {
    val trainSet = sampler.trainset
    val context: PSContext = PSContext.getOrCreate(trainSet.context)
    val numNode = bcMeta.value.srcMeta.length
    val numRow = numNode / numNodePerRow + 1
    val numCol = numNodePerRow * dimension
    val rowsInBlock = numRow / numParts + 1
    val colsInBlock = numCol
    val srcModel = PSMatrix.dense(numRow, numCol, rowsInBlock, colsInBlock, RowType.T_FLOAT_DENSE,
      Map(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS -> classOf[PartitionSourceMap].getName))
    srcModel.psfUpdate(new GERandom(new GERandomParam(srcModel.id, dimension))).get()
    println("srcModel.id"+srcModel.id)
    srcModel
  }


  val dstModel: PSMatrix = {
    val trainSet = sampler.trainset
    val context: PSContext = PSContext.getOrCreate(trainSet.context)
    val numNode = bcMeta.value.srcMeta.length
    val numRow = numNode / numNodePerRow + 1
    val numCol = numNodePerRow * dimension
    val rowsInBlock = numRow / numParts + 1
    val colsInBlock = numCol
    val dstModel = PSMatrix.dense(numRow, numCol, rowsInBlock, colsInBlock, RowType.T_FLOAT_DENSE,
      Map(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS -> classOf[PartitionSourceMap].getName))
    dstModel.psfUpdate(new GERandom(new GERandomParam(dstModel.id, dimension))).get()
    println("dstModel.id"+dstModel.id)
    dstModel
  }


  override def saveSrcModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
    // to be finished
  }

  override def saveDstModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
    // to be finished
  }

  override def destory(): Unit ={
    srcModel.destroy()
    dstModel.destroy()
    PSContext.stop()
  }

}

class GEDOTModel (sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta],
                     dimension: Int, numNodePerRow: Int, numParts: Int) extends GEModel(sampler, bcMeta, dimension){
  val psModel: PSMatrix = {
    val trainSet = sampler.trainset
    val context: PSContext = PSContext.getOrCreate(trainSet.context)
    val numNode = bcMeta.value.srcMeta.length
		val partDim = dimension / numParts
    val order = 2
		val psModel = createPSMatrix(partDim * order, numNode, numParts, numNodePerRow)
    psModel
  }


  private def createPSMatrix(sizeOccupiedPerNode: Int,
                             numNode: Int,
                             numPart: Int,
                             numNodesPerRow: Int = -1): PSMatrix = {
    require(numNodesPerRow <= Int.MaxValue / sizeOccupiedPerNode,
      s"size exceed Int.MaxValue, $numNodesPerRow * $sizeOccupiedPerNode > ${Int.MaxValue}")

    val rowCapacity = if (numNodesPerRow > 0) numNodesPerRow else Int.MaxValue / sizeOccupiedPerNode
    val numRow = (numNode - 1) / rowCapacity + 1
    val numNodesEachRow = if (numNodesPerRow > 0) numNodesPerRow else (numNode - 1) / numRow + 1
    val numCol = numPart.toLong * numNodesEachRow * sizeOccupiedPerNode
    val rowsInBlock = numRow
    val colsInBlock = numNodesEachRow * sizeOccupiedPerNode
    println(s"matrix meta:\n" +
      s"colNum: $numCol\n" +
      s"rowNum: $numRow\n" +
      s"colsInBlock: $colsInBlock\n" +
      s"rowsInBlock: $rowsInBlock\n" +
      s"numNodesPerRow: $numNodesEachRow\n" +
      s"sizeOccupiedPerNode: $sizeOccupiedPerNode"
    )
    // create ps matrix
    val begin = System.currentTimeMillis()
    val psMatrix = PSMatrix
      .dense(numRow, numCol, rowsInBlock, colsInBlock, RowType.T_FLOAT_DENSE,
        Map(AngelConf.ANGEL_PS_PARTITION_SOURCE_CLASS -> classOf[PartitionSourceArray].getName))
    println(s"Model created, takes ${(System.currentTimeMillis() - begin) / 1000.0}s")
    psMatrix
  }

	protected def psfUpdate(func: UpdateFunc): VoidResult = {
		psModel.psfUpdate(func).get()
	}

	protected def psfGet(func: GetFunc): GetResult = {
		psModel.psfGet(func)
	}

	def doGrad(dots: Array[Float], negative: Int, alpha: Float): Double = {
		var loss = 0.0
		for (i <- dots.indices) {
			val prob = FastMath.sigmoid(dots(i))
			if (i % (negative + 1) == 0) {
				dots(i) = alpha * (1 - prob)
				loss -= FastMath.log(prob)
			} else {
				dots(i) = -alpha * FastMath.sigmoid(dots(i))
				loss -= FastMath.log(1 - prob)
			}
		}
		loss
	}


  override def saveSrcModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
    // to be finished
  }

  override def saveDstModel(path: String, bcDict: Broadcast[Array[String]]): Unit = {
    // to be finished
  }

  override def destory(): Unit ={
    psModel.destroy()
    PSContext.stop()
  }

}