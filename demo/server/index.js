const express = require("express");
const fs = require("fs");
const ml = require('./ml.js');
const PORT = process.env.PORT || 3001;

const app = express();

// json parser middleware
app.use(express.json());

const cardDict = JSON.parse(fs.readFileSync("simpleCardDict.json"));
const oracleByName = Object.fromEntries(
  Object.entries(cardDict).map(([key, value]) => [value.name.toLowerCase(), key])
);


const translateCard = (card) => {
  if (card.toLowerCase() in oracleByName) {
    return cardDict[oracleByName[card.toLowerCase()]];
  }

  // magic card back
  return {
    name: card
  }
}

app.post("/api/cards", (req, res) => {
  const { cards } = req.body;

  res.json({
    cards: cards.map(translateCard)
  });  
});

app.post("/api/recommend", (req, res) => {
  const { cards } = req.body;

  const oracles = cards.filter(card => card.toLowerCase() in oracleByName).map(card => oracleByName[card.toLowerCase()]);

  const recommendations = ml.recommend(oracles);

  res.json({
    adds: recommendations.adds.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    })),
    removes: recommendations.removes.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    }))
  });
});

app.post("/api/deckbuild", (req, res) => {
  const { cards } = req.body;

  const oracles = cards.filter(card => card.toLowerCase() in oracleByName).map(card => oracleByName[card.toLowerCase()]);

  const recommendations = ml.deckbuild(oracles);

  res.json({
    mainboard: recommendations.mainboard.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    })),
    sideboard: recommendations.sideboard.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    }))
  });
});

app.post("/api/draft", (req, res) => {
  const { pack, pool } = req.body;

  const packOracles = pack.filter(card => card.toLowerCase() in oracleByName).map(card => oracleByName[card.toLowerCase()]);
  const poolOracles = pool.filter(card => card.toLowerCase() in oracleByName).map(card => oracleByName[card.toLowerCase()]);

  const recommendations = ml.draft(packOracles, poolOracles);

  res.json({
    picks: recommendations.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    }))
  });
});

app.post("/api/synergies", (req, res) => {
  console.log(req.body);

  const { card } = req.body;

  const oracle = oracleByName[card.toLowerCase()];

  const recommendations = ml.synergies(oracle);

  res.json({
    cards: recommendations.map((card) => ({
      ...cardDict[card.oracle],
      rating: card.rating
    }))
  });
});

app.listen(PORT, () => {
  console.log(`Server listening on ${PORT}`);
});
