import nj from "numjs"

// Tokenize sentence
export const tokenize = (
  sentence: string,
  stopwords: string[],
  minWordLength: number
) => {
  const result = sentence.split(/\W+/).filter((token) => {
    token = token.toLowerCase()
    return token.length >= minWordLength && stopwords.indexOf(token) === -1
  })
  return result
}

export const removeDuplicateWords = (words: string[]) => {
  return words.filter((w, i, self) => {
    return self.indexOf(w) === i
  })
}

// Building syllable matrix
export const bow = (
  sentence: string,
  words: string[],
  stopwords: string[],
  minWordLength: number
) => {
  const sentenceWords = tokenize(sentence, stopwords, minWordLength)
  const bag = []
  words.forEach(() => {
    bag.push(0)
  })

  sentenceWords.forEach((singleWordFromSentence) => {
    words.forEach((word, j) => {
      if (word === singleWordFromSentence) {
        bag[j] = 1
      }
    })
  })

  return nj.array(bag)
}
