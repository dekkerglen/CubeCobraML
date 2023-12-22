const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const indexToOracle = JSON.parse(fs.readFileSync("indexToOracleMap.json"));
const oracleToIndex = Object.fromEntries(
  Object.entries(indexToOracle).map(([key, value]) => [value, key])
);

const numOracles = Object.keys(oracleToIndex).length;

let encoder;
let recommend_decoder;
let deckbuilder_decoder;
let draft_decoder;

tf.loadGraphModel('file://../../model/tfjs_model/encoder/model.json').then((model) => {
  encoder = model;
  console.log('encoder loaded');
}).catch((err) => {
  console.log(err);
});

tf.loadGraphModel('file://../../model/tfjs_model/cube_decoder/model.json').then((model) => {
  recommend_decoder = model;
  console.log('recommend_decoder loaded');
}).catch((err) => {
  console.log(err);
});

tf.loadGraphModel('file://../../model/tfjs_model/deck_build_decoder/model.json').then((model) => {
  deckbuilder_decoder = model;
  console.log('deck_build_decoder loaded');
}).catch((err) => {
  console.log(err);
});

tf.loadGraphModel('file://../../model/tfjs_model/draft_decoder/model.json').then((model) => {
  draft_decoder = model;
  console.log('draft_decoder loaded');
}).catch((err) => {
  console.log(err);
});

const embeddings = JSON.parse(fs.readFileSync("embeddings.json"));

const softmax = (array) => {
  const max = Math.max(...array);
  const exps = array.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sum);
}

const encodeIndeces = (indeces) => {
  const tensor = new Array(numOracles).fill(0);

  indeces.forEach((index) => {
    tensor[index] = 1;
  });

  return tensor;
}

const recommend = (oracles) => {

  if (!encoder || !recommend_decoder) {
    return {
      adds: [],
      removes: []
    }
  }

  const vector = [encodeIndeces(oracles.map(oracle => oracleToIndex[oracle]))];
  const tensor = tf.tensor(vector);

  const encoded = encoder.predict(tensor);
  const recommendations = recommend_decoder.predict([encoded]);

  const array = recommendations.dataSync();

  const res = [];

  for (let i = 0; i < numOracles; i++) {
    res.push({
      oracle: indexToOracle[i],
      rating: array[i]
    });
  }  
  

  const adds =  res.sort((a, b) => b.rating - a.rating).filter((card) => !oracles.includes(card.oracle)).slice(0, 100);
  const removes =  res.sort((a, b) => a.rating - b.rating).filter((card) => oracles.includes(card.oracle)).slice(0, 100);
  
  return {
    adds,
    removes
  }
}

const deckbuild = (oracles) => {

  if (!encoder || !deckbuilder_decoder) {
    return {
      mainboard: [],
      sideboard: []
    }
  }

  const vector = [encodeIndeces(oracles.map(oracle => oracleToIndex[oracle]))];
  const tensor = tf.tensor(vector);

  const encoded = encoder.predict(tensor);
  const recommendations = deckbuilder_decoder.predict([encoded]);

  const array = recommendations.dataSync();

  const res = [];

  for (let i = 0; i < numOracles; i++) {
    const oracle = indexToOracle[i];

    if (oracles.includes(oracle)) {
      res.push({
        oracle: indexToOracle[i],
        rating: array[i]
      });
    }
  }

  const sorted = res.sort((a, b) => b.rating - a.rating);
  const mainboard = sorted.slice(0, 24);
  const sideboard = sorted.slice(24, oracles.length).reverse();

  return {
    mainboard,
    sideboard
  }
}

const draft = (pack, pool) => {
  const vector = [encodeIndeces(pool.map(oracle => oracleToIndex[oracle]))]; 
  const tensor = tf.tensor(vector);

  const encoded = encoder.predict(tensor);
  const recommendations = draft_decoder.predict([encoded]);

  const array = recommendations.dataSync();

  const packVector = encodeIndeces(pack.map(oracle => oracleToIndex[oracle]));
  const mask = packVector.map((x) => 1e9 * (1 - x));
  
  const softmaxed = softmax(array.map((x, i) => x * packVector[i] - mask[i]));

  const res = [];

  for (let i = 0; i < numOracles; i++) {
    const oracle = indexToOracle[i];
    if (pack.includes(oracle)) {
      res.push({
        oracle: indexToOracle[i],
        rating: softmaxed[i]
      });
    }
  }
  
  return res.sort((a, b) => b.rating - a.rating);
}

const encode = (one_hot_encodings) => {
  return tf.tidy(() => {
    const tensor = tf.tensor(one_hot_encodings);
    return encoder.predict(tensor).arraySync();
  });
};

const cosineSimilarity = (a, magA, b, magB) => {
  const dotProduct = a.reduce((acc, val, index) => acc + val * b[index], 0);
  return dotProduct / (magA * magB);
};

const synergies = (oracle) => {
  const currentEmbedding = embeddings[oracleToIndex[oracle]];
  const currentMagnitude = Math.sqrt(currentEmbedding.reduce((acc, val) => acc + val * val, 0));

  const res = [];

  for (let i = 0; i < numOracles; i++) {
    const oracle = indexToOracle[i];
    const embedding = embeddings[i];
    const magnitude = Math.sqrt(embedding.reduce((acc, val) => acc + val * val, 0));

    const rating = cosineSimilarity(currentEmbedding, currentMagnitude, embedding, magnitude);

    res.push({
      oracle,
      rating
    });
  }

  return res.sort((a, b) => b.rating - a.rating).slice(1, 101);
}

module.exports = {
  recommend,
  deckbuild,
  draft,
  encode,
  synergies
}
