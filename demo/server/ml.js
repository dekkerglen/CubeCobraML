const tf = require('@tensorflow/tfjs-node');

const inbdexToOracle = JSON.parse(fs.readFileSync("indexToOracleMap.json"));
const oracleToIndex = Object.fromEntries(
  Object.entries(indexToOracle).map(([key, value]) => [value, key])
);

let encoder;
let recommend_decoder;

tf.loadLayersModel('file://../../model/model/encoder/model').then((model) => {
  encoder = model;
  console.log('encoder loaded');
});

tf.loadLayersModel('file://../../model/model/recommend_decoder/model').then((model) => {
  recommend_decoder = model;
  console.log('recommend_decoder loaded');
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
    oracle: inbdexToOracle[index],
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
