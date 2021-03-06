import fs from "fs"
import nj from "numjs"
import { binaryArray, curve, rand } from "./util/math"
import { config } from "../nnConfig"
import { IDocument } from "./types/types"
import { removeDuplicateWords, tokenize } from "./util/nlp"
import { stopWords } from "../data/stopwords"
import { trainigData } from "../data/data"

export const prepareData = (removeStopWords: boolean = false) => {
  let words: string[] = []
  const classes: string[] = []
  const documents: IDocument[] = []
  console.log(trainigData.length, "sentences in training data")
  // tokenize sentencens
  // all words > 0 letter
  // no stop words right now
  trainigData.forEach((data) => {
    const wordArray = tokenize(data.sentence, [], 1)
    wordArray.forEach((word) => {
      words.push(word)
    })
    documents.push({ words: wordArray, class: data.class })
    // add class if not allready in array
    !classes.some((className) => className.includes(data.class)) &&
      classes.push(data.class)
  })

  // remove duplicates
  words = removeDuplicateWords(words)

  // remove stopWords if flag is set
  if (removeStopWords) {
    console.log("removing stop words ...")
    words = words.filter((w, i, self) => {
      return !stopWords.includes(w) && self.indexOf(w) === i
    })
  }

  console.log(documents.length, "documents")
  console.log(classes.length, "classes")
  console.log(words.length, "unique words")

  const training: number[][] = []
  const output: number[][] = []
  // create empty array with outpu for each class
  const outputEmpty = new Array(classes.length).fill(0)

  // training set, bagOfWords of words for each sentence
  documents.forEach((document) => {
    const bagOfWords: number[] = []
    const docuentWords = document.words
    words.forEach((word) => {
      docuentWords.includes(word) ? bagOfWords.push(1) : bagOfWords.push(0)
    })
    training.push(bagOfWords)
    const outputRow = outputEmpty.slice(0)
    outputRow[classes.indexOf(document.class)] = 1
    output.push(outputRow)
  })
  return {
    classes,
    output,
    training,
    words,
  }
}

export const trainModel = (
  words: string[],
  classes: string[],
  trainingArray: nj.NdArray<number[]>,
  outputArray: nj.NdArray<number[]>,
  hiddenNeurons: number,
  alpha: number,
  epochs: number
) => {
  const STARTTIME = new Date()

  const inputMatrix = trainingArray.tolist()
  console.log(hiddenNeurons, "neurons")
  console.log(alpha, "alpha")
  console.log("input matrix", inputMatrix.length + "x" + inputMatrix[0].length)
  console.log("output matrix: 1x", classes.length)
  let error = [1]
  // randomly initialize our weights with mean 0
  let synapse0 = nj.array(rand(inputMatrix[0].length, hiddenNeurons))
  let synapse1 = nj.array(rand(hiddenNeurons, classes.length))

  let prevSynapseWeightUpdate0:
    | nj.NdArray<number[]>
    | nj.NdArray<number> = nj.zeros(synapse0.shape)
  let prevSynapseWeightUpdate1 = nj.zeros(synapse1.shape)

  let synapseDirectionCount0 = nj.zeros(synapse0.shape)
  let synapseDirectionCount1 = nj.zeros(synapse1.shape)

  for (let j = 0; j < epochs + 1; j++) {
    const layer0 = trainingArray
    const layer1 = nj.sigmoid(nj.dot(layer0, synapse0))
    const layer2 = nj.sigmoid(nj.dot(layer1, synapse1))
    const layer2Error = outputArray.subtract(layer2)

    if (j % 10000 === 0 && j > 5000) {
      if (nj.mean(nj.abs(layer2Error)) < error) {
        console.log(
          "iteration: " + j + " - delta: " + nj.mean(nj.abs(layer2Error))
        )
        error = nj.mean(nj.abs(layer2Error))
      } else {
        console.log("break:" + nj.mean(nj.abs(layer2Error)) + ">" + error)
        break
      }
    }

    const layer2Delta = layer2Error.multiply(curve(layer2))
    const layer1Error = layer2Delta.dot(synapse1.T)
    const layer1Delta = layer1Error.multiply(curve(layer1))
    const synapse1WeightUpdate = layer1.T.dot(layer2Delta)
    const synapse0WeightUpdate = layer0.T.dot(layer1Delta)

    if (j > 0) {
      synapseDirectionCount0 = synapseDirectionCount0.add(
        nj.abs(
          binaryArray(synapse0WeightUpdate).subtract(
            binaryArray(prevSynapseWeightUpdate0)
          )
        )
      )

      synapseDirectionCount1 = synapseDirectionCount1.add(
        nj.abs(
          binaryArray(synapse1WeightUpdate).subtract(
            binaryArray(prevSynapseWeightUpdate1)
          )
        )
      )
    }

    synapse1 = synapse1.add(synapse1WeightUpdate.multiply(alpha))
    synapse0 = synapse0.add(synapse0WeightUpdate.multiply(alpha))

    prevSynapseWeightUpdate0 = synapse0WeightUpdate
    prevSynapseWeightUpdate1 = synapse1WeightUpdate
  }
  // Saving trained model to file
  const model = "model.json"
  const synapse = JSON.stringify(
    {
      classes,
      synapse0: synapse0.tolist(),
      synapse1: synapse1.tolist(),
      words,
    },
    null,
    4
  )

  fs.writeFileSync(model, synapse, "utf8")
  console.log("model:", model)

  // Calculating trining time
  const ENDTIME = new Date()
  const TRAININGTIME = (Number(ENDTIME) - Number(STARTTIME)) / 1000
  if (TRAININGTIME > 60) {
    const min = Math.floor(TRAININGTIME / 60)
    const sec = Math.floor(TRAININGTIME % 60)
    console.log("training completed in: " + min + " min " + sec + " sec")
  } else {
    console.log("training completed in: " + TRAININGTIME + " sec")
  }
}

export const startTraining = (useStopWord?: boolean) => {
  const { classes, output, training, words } = prepareData(useStopWord)
  trainModel(
    words,
    classes,
    nj.array(training),
    nj.array(output),
    config.hiddenNeurons,
    config.alpha,
    config.epochs
  )
}
