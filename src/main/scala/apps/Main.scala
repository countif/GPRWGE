package apps

import ge.{EmbeddingConf, GEPS,GEWORKER,GEDOT}
import utils.ArgsUtil


object Main{
	def main(args: Array[String]): Unit = {
		val params = ArgsUtil.parse(args)
		val platForm = params.getOrElse(EmbeddingConf.PLATFORM, "PS").toUpperCase()
		platForm match{

			case "PS" => new GEPS(params).run()
			case "GEWORKER" => new GEWORKER(params).run()
			case "GEDOT" => new GEDOT(params).run()
			case _ =>
			{
				System.out.println("need platform paramters")
				System.exit(-1)
			}
		}

	}
}