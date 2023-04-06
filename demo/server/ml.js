const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const indexToOracle = JSON.parse(fs.readFileSync("indexToOracleMap.json"));
const oracleToIndex = Object.fromEntries(
  Object.entries(indexToOracle).map(([key, value]) => [value, key])
);

let encoder;
let recommend_decoder;

tf.loadLayersModel('file://../../model/model/encoder_js/model/model.json').then((model) => {
  encoder = model;
  console.log('encoder loaded');
}).catch((err) => {
  console.log(err);
});

tf.loadLayersModel('file://../../model/model/cube_decoder_js/model/model.json').then((model) => {
  recommend_decoder = model;
  console.log('recommend_decoder loaded');
}).catch((err) => {
  console.log(err);
});

const recommend = (oracles) => {

  if (!encoder || !recommend_decoder) {
    return {
      adds: [],
      removes: []
    }
  }

  const tensor = tf.tensor(oracles.map(oracle => oracleToIndex[oracle]));

  const encoded = encoder.predict(tensor);
  const recommendations = recommend_decoder.predict(encoded);

  const sorted = recommendations.dataSync().map((rating, index) => ({
    oracle: indexToOracle[index],
    rating
  })).sort((a, b) => b.rating - a.rating);

  const adds = sorted.filter((card) => !oracles.includes(card.oracle)).slice(0, 100);
  const removes = sorted.filter((card) => oracles.includes(card.oracle)).slice(0, 100);

  return {
    adds: adds,
    removes: removes
  }
}

module.exports = {
  recommend
}
