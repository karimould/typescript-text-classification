import { classify } from "./classify"
import { startTraining } from "./train"
process.argv[2] === "train" &&
  startTraining(process.argv[3] === "stopwords" ? true : false)
process.argv[2] === "classify" && classify(process.argv[3])
