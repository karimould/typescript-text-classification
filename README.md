# TypeScript Text Classification

Text classification using a neuronal network and TypeScript.

Right now the app does not use any module for the NLP util functions, this may change.
For the matrix multiplications and the sigmoid function it uses [numjs](hhttps://github.com/nicolaspanel/numjs.git).

The data used for this example is intended to simulate a simple query to a home assistant.

---

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Node.js](https://nodejs.org/en/download/) (which comes with [npm](http://npmjs.com)) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/karimould/typescript-text-classification

# Go into the repository
$ cd typescript-text-classification

# Install dependencies
$ npm install
```

### How To Train The Model

If you have installed all the node modules, you can train the model from your command line with:

```bash
$ npm start train
```

If you want to remove stop words from your training data before trainig the model you can use:

```bash
$ npm start train stopwords
```

### How To Classify

If you have trained your model, you can run the classification from the terminal with:

```bash
$ npm start classify 'your sentence'
```

Examples:

```bash
$ npm start classify 'thank you, but i dont need your help'
```

```bash
$ npm start classify 'please tell me the time'
```

```bash
$ npm start classify 'is it going to rain today or can i leave the umbrella at home'
```

---

## How To Add Training Data And Stopwords

To set your training data you can change the JSON in the data.ts at:

```bash
$ ./data/data.ts
```

To change the stop words you can change the stopwords.ts at:

```bash
$ ./data/stopwords.ts
```

The data with wich the neuronal network is trained is an array of JSON's with a class and a sentence.

```typescript
interface ITrainingData {
  class: string
  sentence: string
}
```

The data gets preprocessed and parsed in to documents.
A document contains the words of the sentence in an array of strings and the class.

```typescript
interface IDocument {
  words: string[]
  class: string
}
```

## How To Change The Config

You can change the config of the neuronal network in the nnConfig.ts at:

```bash
$ ./nnConfig.ts
```

Here you can set:

- alpha
- epochs
- errorTreshold
- hiddenNeurons

---

## License

MIT

---

## ToDo

- write about config parameters
- put example outputs in readme
