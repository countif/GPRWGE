package ge


import com.tencent.angel.ml.matrix.psf.get.base.{GetFunc, GetResult}
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateFunc, VoidResult}
import com.tencent.angel.spark.models.PSMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import ge.basics.{ChunkDataset, DistributionMeta, HashMapModel, PairsDataset}
import ge.PSUtils._
import it.unimi.dsi.fastutil.ints.{Int2IntOpenHashMap, IntOpenHashSet}
import org.apache.spark.TaskContext
import com.tencent.angel.ml.matrix.psf.get.base.{GetFunc, GetResult}
import com.tencent.angel.spark.ml.psf.embedding.{Init, InitParam, NEModelRandomize}
import org.apache.spark.sql.SparkSession
import samplers.{APP, BaseSampler, DeepWalk, LINE}
import utils.FastMath
import com.tencent.angel.spark.ml.psf.embedding.NEDot.NEDotResult
import com.tencent.angel.spark.ml.psf.embedding.line.{Adjust, AdjustParam, Dot, DotParam}

import scala.util.Random


//import ge.DotAdjust.{Adjust, AdjustParam, Dot, DotParam}

class GEDOT(params: Map[String, String]) extends GraphEmbedding(params: Map[String, String]){

  val numParts = params.getOrElse(EmbeddingConf.NUMPARTS, "10").toInt
  val numNodePerRow = params.getOrElse(EmbeddingConf.NUMNODEPERROW, "10000").toInt
  var psMatrix:PSMatrix =  _
  val partDim: Int = dimension / numParts
  val seed = Random.nextInt()
  private val rand = new Random(seed)

  override def initModel(sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta]): GEModel = {
    new GEDOTModel(sampler, bcMeta, dimension, numNodePerRow, numParts)
  }


  def getDotFunc(data: PairsDataset, batchSeed: Int, ns: Int, partitionId: Int): GetFunc = {
    val pairData = data.asInstanceOf[PairsDataset]
    val param = new DotParam(psMatrix.id, batchSeed, partitionId, pairData.src.toArray, pairData.dst.toArray)
    new Dot(param)
  }

  protected def psfUpdate(func: UpdateFunc): VoidResult = {
    psMatrix.psfUpdate(func).get()
  }

  protected def psfGet(func: GetFunc): GetResult = {
    psMatrix.psfGet(func)
  }

  def getAdjustFunc(data: PairsDataset,
                             batchSeed: Int,
                             ns: Int,
                             grad: Array[Float],
                             partitionId: Int): UpdateFunc = {
    val pairData = data.asInstanceOf[PairsDataset]
    val param = new AdjustParam(psMatrix.id, batchSeed, ns, partitionId, grad, pairData.src.toArray, pairData.dst.toArray)
    new Adjust(param)
  }

  def doGrad(dots: Array[Float], negative: Int, alpha: Float): Float = {
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
    loss.toFloat
  }

  private def randomInitialize(seed: Int): Unit = {
    val beforeRandomize = System.currentTimeMillis()
    psfUpdate(new NEModelRandomize(psMatrix.id, dimension / numParts, dimension, 2, seed))
    logInfo(s"Model successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
    System.err.println(s"Model successfully Randomized, cost ${(System.currentTimeMillis() - beforeRandomize) / 1000.0}s")
  }

  override def train(batchData: RDD[(Int, PairsDataset)], bcMeta: Broadcast[DistributionMeta],
                     gEModel: GEModel, vertexNum: Int, iterationId: Int): (Float, Int) = {
    val numPartititions = batchData.getNumPartitions
    val gePSModel: GEDOTModel = gEModel.asInstanceOf[GEDOTModel]

    val psModel: PSMatrix = gePSModel.psModel
    var seed = 5
    val (batchLoss, batchCnt) = batchData.mapPartitionsWithIndex((partitionId,iterator) => {
      //batch data
      val pairsDataset = iterator.next()._2
      val srcIds = pairsDataset.src.toArray

      var (start, end) = (0L, 0L)
      // dot
      start = System.currentTimeMillis()
      val dots = psfGet(getDotFunc(pairsDataset, seed, negative, partitionId))
        .asInstanceOf[NEDotResult].result
      end = System.currentTimeMillis()
      val dotTime = end - start
      // gradient
      start = System.currentTimeMillis()
      val loss = doGrad(dots, negative, stepSize)
      end = System.currentTimeMillis()
      val gradientTime = end - start
      // adjust
      start = System.currentTimeMillis()
      psfUpdate(getAdjustFunc(pairsDataset, seed, negative, dots, partitionId))
      end = System.currentTimeMillis()
      val adjustTime = end - start
      //       return loss

      logInfo(s"dotTime=$dotTime gradientTime=$gradientTime adjustTime=$adjustTime")
      System.err.println(s"dotTime=$dotTime gradientTime=$gradientTime adjustTime=$adjustTime")


      // push back and compute loss
      Iterator.single((loss, srcIds.length))
    },preservesPartitioning = true).reduce((x, y) => (x._1 + y._1, x._2 + y._2))


    logInfo(s"*ghand*batch finished, batchLoss: ${batchLoss / batchCnt}, batchCnt:${batchCnt}")
    System.err.println(s"*ghand*batch finished, batchLoss: ${batchLoss / batchCnt}, batchCnt:${batchCnt}")
    (batchLoss, batchCnt)
  }

  def getInitFunc(numPartitions: Int, maxIndex: Int, maxLength: Int, negative:Int, window: Int): UpdateFunc = {
    val param = new InitParam(psMatrix.id, numPartitions, maxIndex, maxLength, negative, 2, partDim, window)
    new Init(param)
  }

  override  def run(): Unit = {
    System.out.println("hello this is GEDOT!!!")
    System.err.println("hello this is GEDOT!!!")

    val spark: SparkSession= SparkSession.builder().appName("GraphEmbedding-" + platForm).getOrCreate()
    val sparkContext = spark.sparkContext

    /* loading data */
    var start_time = System.currentTimeMillis()
    val (chunkedDataset, bcDict, vertexNum): (RDD[(ChunkDataset, Int)],
      Broadcast[Array[String]], Int) = loadData(sparkContext)
    chunkedDataset.setName("chunkedDatasetRDD").cache()
    val numChunks: Int = chunkedDataset.count().toInt
    logInfo(s"*ghand*finished loading data, n" +
      s"um of chunks:${numChunks}, num of vertex:${vertexNum}, " +
      s"time cost: ${(System.currentTimeMillis() - start_time) / 1000.0f}")
    System.err.println(s"*ghand*finished loading data, n" +
      s"um of chunks:${numChunks}, num of vertex:${vertexNum}, " +
      s"time cost: ${(System.currentTimeMillis() - start_time) / 1000.0f}")
    start_time = System.currentTimeMillis()
    val sampler: BaseSampler = samplerName match {
      case "APP" =>
        new APP(chunkedDataset, vertexNum, stopRate)
      case "DEEPWALK" =>
        new DeepWalk(chunkedDataset, vertexNum, window)
      case "LINE" =>
        new LINE(chunkedDataset, vertexNum)
    }

    /**
      * for now we don't split vertex. Later will modify this for power-law.
      */
    val meta: DistributionMeta = sampler.hashDestinationForNodes(sampler.trainset.getNumPartitions)
    val bcMeta = sparkContext.broadcast(meta)
    val geModel: GEModel = initModel(sampler, bcMeta)
    psMatrix = geModel.asInstanceOf[GEDOTModel].psModel
    val batchIter = sampler.batchIterator() // barrierRDD.
    val trainStart = System.currentTimeMillis()


    psMatrix.psfUpdate(getInitFunc(chunkedDataset.getNumPartitions, vertexNum, -1, negative = negative, -1))
    randomInitialize(rand.nextInt)

    for (epochId <- 0 until (numEpoch)) {
      var trainedLoss = 0.0
      var trainedPairs = 0
      var batchId = 0
      while (batchId < numChunks) {
        val batchData: RDD[PairsDataset] = batchIter.next() // barrierRDD
        val shuffledData = shuffleDataBySource(batchData, bcMeta)
        //shuffledData.cache()
        val (batchLoss, batchCnt) = train(shuffledData, bcMeta, geModel, vertexNum, batchId)
        //shuffledData.unpersist()
        trainedLoss += batchLoss
        trainedPairs += batchCnt
        batchId += 1

        System.err.println(s"*ghand*epochId:${epochId} trainedPairs:${trainedPairs}")

        System.err.println(s"*ghand*epochId:${epochId} batchId:${batchId} " +
          s"batchPairs:${batchCnt} loss:${batchLoss / batchCnt}")

        logInfo(s"*ghand*epochId:${epochId} trainedPairs:${trainedPairs}")

        logInfo(s"*ghand*epochId:${epochId} batchId:${batchId} " +
          s"batchPairs:${batchCnt} loss:${batchLoss / batchCnt}")
      }
      logInfo(s"*ghand*epochId:${epochId} trainedPairs:${trainedPairs} " +
        s"loss:${trainedLoss / trainedPairs}")
      System.err.println(s"*ghand*epochId:${epochId} trainedPairs:${trainedPairs} " +
        s"loss:${trainedLoss / trainedPairs}")

      if (((epochId + 1) % checkpointInterval) == 0) {
        // checkpoint the model
        geModel.saveSrcModel(output + "_src_" + epochId, bcDict)
        geModel.saveDstModel(output + "_dst_" + epochId, bcDict)
      }
    }
    logInfo(s"*ghand* training ${numEpoch} epochs takes: " +
      s"${(System.currentTimeMillis() - trainStart) / 1000.0} seconds")
    System.err.println(s"*ghand* training ${numEpoch} epochs takes: " +
      s"${(System.currentTimeMillis() - trainStart) / 1000.0} seconds")
    geModel.destory()
  }

}

