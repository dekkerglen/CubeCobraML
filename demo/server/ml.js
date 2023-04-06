const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const indexToOracle = JSON.parse(fs.readFileSync("indexToOracleMap.json"));
const oracleToIndex = Object.fromEntries(
  Object.entries(indexToOracle).map(([key, value]) => [value, key])
);

const numOracles = Object.keys(oracleToIndex).length;
console.log(numOracles);

let encoder;
let recommend_decoder;
let deckbuilder_decoder;

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
  console.log(encoded.dataSync());
  const recommendations = recommend_decoder.predict([encoded]);

  const array = recommendations.dataSync();
  console.log(array);

  const res = [];

  for (let i = 0; i < numOracles; i++) {
    res.push({
      oracle: indexToOracle[i],
      rating: array[i]
    });
  }  
  

  const sorted = res.sort((a, b) => b.rating - a.rating);

  const adds = sorted.filter((card) => !oracles.includes(card.oracle)).slice(0, 100);
  const removes = sorted.filter((card) => oracles.includes(card.oracle)).slice(0, 100);
  

  return {
    adds,
    removes
  }
}

module.exports = {
  recommend
}
