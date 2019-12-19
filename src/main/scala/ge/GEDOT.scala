package ge

import java.util.Random

import com.tencent.angel.spark.models.PSMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import ge.basics.{ChunkDataset, DistributionMeta, HashMapModel, PairsDataset}
import ge.PSUtils._
import it.unimi.dsi.fastutil.ints.{Int2IntOpenHashMap, IntOpenHashSet}
import org.apache.spark.TaskContext
import org.apache.spark.sql.SparkSession
import samplers.{APP, BaseSampler, DeepWalk, LINE}
import utils.FastMath
class GEDOT(params: Map[String, String]) extends GraphEmbedding(params: Map[String, String]){

  val numParts = params.getOrElse(EmbeddingConf.NUMPARTS, "10").toInt
  val numNodePerRow = params.getOrElse(EmbeddingConf.NUMNODEPERROW, "10000").toInt

  override def initModel(sampler: BaseSampler, bcMeta: Broadcast[DistributionMeta]): GEModel = {
    new GEDOTModel(sampler, bcMeta, dimension, numNodePerRow, numParts)
  }

  override def train(batchData: RDD[(Int, PairsDataset)], bcMeta: Broadcast[DistributionMeta],
                     gEModel: GEModel, vertexNum: Int, iterationId: Int): (Float, Int) = {
    val numPartititions = batchData.getNumPartitions
    val gePSModel: GEDOTModel = gEModel.asInstanceOf[GEDOTModel]
    val srcModel: PSMatrix = gePSModel.srcModel
    val dstModel: PSMatrix = gePSModel.dstModel

    val (batchLoss, batchCnt) = batchData.mapPartitions((iterator) => {
      var (startTime, endTime) = (0L, 0L)
      val pairsDataset = iterator.next()._2
      val srcIds = pairsDataset.src.toArray
      val dstIds = pairsDataset.dst.toArray

      // get negative sampling nodes
      val random: Random = new Random(iterationId)
      val negArray: Array[Int] = Array.ofDim(negative)
      for (i <- 0 until (negArray.length)) {
        negArray(i) = random.nextInt(vertexNum)
      }

      // get srcModel indices to pull
      val srcindicesSet: IntOpenHashSet = new IntOpenHashSet()

      for(i <- 0 until(negArray.length)){
        srcindicesSet.add(srcIds(i))
      }


      // get dstModel indices to pull
      val indicesSet: IntOpenHashSet = new IntOpenHashSet()
      for(i <- 0 until(negArray.length)){
        indicesSet.add(negArray(i))
      }
      for(i <- 0 until(dstIds.length)){
        indicesSet.add(dstIds(i))
      }
      logInfo(s"*ghand*dstIds.length:${dstIds.length}")
      val dstindices: Array[Int] = indicesSet.toIntArray()
      val srcindices: Array[Int] = indicesSet.toIntArray()

      startTime = System.currentTimeMillis()
      // pull them and train
      val srcresult  = srcModel.psfGet(new GEPull(
        new GEPullParam(srcModel.id,
          srcindices,
          numNodePerRow,
          dimension)))
        .asInstanceOf[GEPullResult]

      val dstresult  = dstModel.psfGet(new GEPull(
        new GEPullParam(dstModel.id,
          dstindices,
          numNodePerRow,
          dimension)))
        .asInstanceOf[GEPullResult]

      endTime = System.currentTimeMillis()
      logInfo(s"*ghand*worker ${TaskContext.getPartitionId()} pulls ${srcindices.length+dstindices.length} vectors " +
        s"from PS, takes ${(endTime - startTime) / 1000.0} seconds.")

      startTime = System.currentTimeMillis()
      val dstindex2offset = new Int2IntOpenHashMap() //
      dstindex2offset.defaultReturnValue(-1)
      for (i <- 0 until dstindices.length) dstindex2offset.put(dstindices(i), i)

      val srcindex2offset = new Int2IntOpenHashMap()
      srcindex2offset.defaultReturnValue(-1)
      for (i <- 0 until srcindices.length) srcindex2offset.put(srcindices(i),i)

      val dstdeltas = new Array[Float](dstresult.layers.length)
      // deep copy for deltas, we do asgd, for shared Negative sampling, it is not asgd
      val srcdeltas = new Array[Float](srcresult.layers.length)
      for(i <- 0 until(srcresult.layers.length)){
        srcdeltas(i) = srcresult.layers(i)
      }
      for(i <- 0 until(dstresult.layers.length)){
        dstdeltas(i) = dstresult.layers(i)
      }
      // deep copy for src vec

      var loss = 0.0f
      val sharedNegativeUpdate: Array[Float] = Array.ofDim(negative * dimension)
      for(i <- 0 until(srcIds.length)){
        val srcId = srcIds(i)
        val dstId = dstIds(i)
        loss += psTrainOnePair(srcdeltas,srcId, dstId, negArray, sharedNegativeUpdate,srcindex2offset, dstindex2offset, dstdeltas)
      }

      for(i <- 0 until(srcdeltas.length)){
        srcdeltas(i) -= srcresult.layers(i)
      }
      for(i <- 0 until(dstdeltas.length)){
        dstdeltas(i) -= dstresult.layers(i)
      }


      // update the shared negative sampling nodes
      for(i <- 0 until(negArray.length)){
        val ioffset = dstindex2offset.get(negArray(i)) * dimension
        for(j <- 0 until(dimension)) {
          dstdeltas(ioffset + j) += sharedNegativeUpdate(i * dimension + j)
        }
      }
      endTime = System.currentTimeMillis()
      // TODO: some penalty to frequently updates, i.e., stale updates
      logInfo(s"*ghand*worker ${TaskContext.getPartitionId()} finished training, " +
        s"takes ${(endTime - startTime) / 1000.0} seconds.")
      startTime = System.currentTimeMillis()
      dstModel.psfUpdate(new GEPush(
        new GEPushParam(dstModel.id,
          dstindices, dstdeltas, numNodePerRow, dimension)))
        .get()
      srcModel.psfUpdate(new GEPush(
        new GEPushParam(srcModel.id,
          srcindices, srcdeltas, numNodePerRow, dimension)))
        .get()
      endTime = System.currentTimeMillis()
      logInfo(s"*ghand*worker ${TaskContext.getPartitionId()} pushes ${dstindices.length} vectors " +
        s"to PS, takes ${(endTime - startTime) / 1000.0} seconds.")

      // push back and compute loss
      Iterator.single((loss, srcIds.length))
    }).reduce((x, y) => (x._1 + y._1, x._2 + y._2))


    logInfo(s"*ghand*batch finished, batchLoss: ${batchLoss / batchCnt}, batchCnt:${batchCnt}")
    (batchLoss, batchCnt)
  }

  /**
    *
    * @param srcVec
    * @param srcId
    * @param dstId
    * @param negArray
    * @param sharedNegativeUpdate
    * @param srcindex2offset
    * @param dstindex2offset
    * @param ctxVecs
    * @return
    */
  def psTrainOnePair(srcVec: Array[Float],srcId:Int, dstId: Int, negArray: Array[Int],
                     sharedNegativeUpdate: Array[Float],srcindex2offset: Int2IntOpenHashMap,
                     dstindex2offset: Int2IntOpenHashMap,ctxVecs: Array[Float]): Float = {
    //					   dstOffset: Int, negative: Int, sharedNegativeUpdate: Array[Float]): Float = {
    var loss: Float = 0.0f
    val srcUpdate: Array[Float] = Array.ofDim(dimension)

    //src offset
    val srcindex = srcindex2offset.get(srcId)
    val srcoffset = srcindex*dimension

    // positive pair
    var label = 1
    val dstOffset = dstindex2offset.get(dstId) * dimension
    var sum = 0.0f
    for(i <- 0 until(dimension)) {
      sum += srcVec(i+srcoffset) * ctxVecs(i + dstOffset)
    }
    val prob = FastMath.sigmoid(sum)
    val g = -(label - prob) * stepSize

    for (i <- 0 until (dimension)) {
      srcUpdate(i) -= g * ctxVecs(i + dstOffset)
      ctxVecs(i + dstOffset) -= g * srcVec(i+srcoffset)
    }
    loss += -FastMath.log(prob)

    // negative pairs
    for (negId <- 0 until (negArray.length)) {
      label = 0
      val dstOffset = dstindex2offset.get(negArray(negId)) * dimension
      var sum = 0.0f
      for(i <- 0 until(dimension)){
        sum += srcVec(i+srcoffset) * ctxVecs(i + dstOffset)
      }
      val prob = FastMath.sigmoid(sum)
      val g = -(label - prob) * stepSize

      for (i <- 0 until (dimension)) {
        srcUpdate(i) -= g * ctxVecs(dstOffset + i)
        sharedNegativeUpdate(negId * dimension + i) -= g * srcVec(i+srcoffset)
      }
      loss += -FastMath.log(1 - prob)
    }

    for (i <- 0 until (srcVec.length)) {
      srcVec(i+srcoffset) += srcUpdate(i)
    }
    loss
  }


  override  def run(): Unit = {
    System.out.println("hello this is GEDOT!!!")
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

    val batchIter = sampler.batchIterator() // barrierRDD.
    val trainStart = System.currentTimeMillis()

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
        logInfo(s"*ghand*epochId:${epochId} batchId:${batchId} " +
          s"batchPairs:${batchCnt} loss:${batchLoss / batchCnt}")
      }
      logInfo(s"*ghand*epochId:${epochId} trainedPairs:${trainedPairs} " +
        s"loss:${trainedLoss / trainedPairs}")

      if (((epochId + 1) % checkpointInterval) == 0) {
        // checkpoint the model
        geModel.saveSrcModel(output + "_src_" + epochId, bcDict)
        geModel.saveDstModel(output + "_dst_" + epochId, bcDict)
      }
    }
    logInfo(s"*ghand* training ${numEpoch} epochs takes: " +
      s"${(System.currentTimeMillis() - trainStart) / 1000.0} seconds")
    geModel.destory()
  }

}

