import fs from "fs"
import nj from "numjs"
import { config } from "../nnConfig"
import { bow } from "./util/nlp"

// load model, words and classes from model file
const model = "model.json"
const synapse = JSON.parse(fs.readFileSync(model, "utf8"))

const synapse0 = nj.array(synapse.synapse0)
const synapse1 = nj.array(synapse.synapse1)

const words = synapse.words
const classes = synapse.classes

export const runThroughModel = (sentence: string) => {
  const backOfWords = bow(sentence, words, [], 1)
  const l0 = backOfWords
  const l1 = nj.sigmoid(nj.dot(l0, synapse0))
  const l2 = nj.sigmoid(nj.dot(l1, synapse1))

  return l2
}

export const classify = (sentence: string) => {
  const results = runThroughModel(sentence)
  const resultArray = results.tolist()

  let resultsTrimmed = []
  for (let i in resultArray) {
    if (resultArray[i] > config.errorTreshold) {
      resultsTrimmed.push([i, resultArray[i]])
    }
  }

  resultsTrimmed = resultsTrimmed.sort((a, b) => {
    if (a[0] === b[0]) {
      return 0
    } else {
      return a[0] > b[0] ? -1 : 1
    }
  })

  const returnResult = []
  // tslint:disable-next-line: forin
  for (let i in resultsTrimmed) {
    const r = resultsTrimmed[i]
    const returnJSON = {
      class: classes[r[0]],
      probability: r[1],
    }
    returnResult.push(returnJSON)
  }

  if (!returnResult.length) {
    returnResult.push(["could not classify", "-1"])
  }
  console.log("--------------------------------")
  console.log("sentence:", sentence)
  console.log("output", returnResult)
  return returnResult
}
