const fs = require('fs');

const sourceDir = 'raw_data';
const destDir = 'data';


function writeFile(filepath, data) {
  
  const fd = fs.openSync(filepath, 'w');
  
  fs.writeSync(fd, '[');

  const batchSize = 10000;

  for (let i = 0; i < data.length; i += batchSize) {
    const serialized = JSON.stringify(data.slice(i, i + batchSize));

    // trim the brackets
    fs.writeSync(fd, serialized.substring(1, serialized.length - 1));

    if (i + batchSize < data.length) {
      fs.writeSync(fd, ',');
    }
  }

  
  fs.writeSync(fd, ']');
  fs.closeSync(fd);
}

const processCubes = (numOracles) => {
  console.log('\tLoading cubes...');

  const cubes = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'))
    .filter((cube) => cube.cards.length > 0)
    .map((cube) => cube.cards);

  const oracleFrequency = new Array(numOracles).fill(0);

  for (let i = 0; i < cubes.length; i++) {
    for (let j = 0; j < cubes[i].length; j++) {
      oracleFrequency[cubes[i][j]]++;
    }
  }

  console.log(`\tLoaded ${cubes.length} cubes.`);

  fs.writeFileSync(`${destDir}/cubes.json`, JSON.stringify(cubes));
  fs.writeFileSync(`${destDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  
  console.log('\tDone processing cubes.');
}

const processDecks = (numOracles) => {
  console.log('\tLoading decks...');

  // enumurate src/decks
  const deckFiles = fs.readdirSync(`${sourceDir}/decks`);

  console.log(`\tLoaded ${deckFiles.length} deck files.`);

  const decks = [];

  for (let i = 0; i < deckFiles.length; i++) {
    decks.push(...JSON.parse(fs.readFileSync(`${sourceDir}/decks/${deckFiles[i]}`, 'utf8'))
      .filter((deck) => deck.mainboard.length > 0 || deck.sideboard.length > 0)
      .map((deck) => ({
        mainboard: deck.mainboard.filter((card) => !(deck.basics || []).includes(card)),
        sideboard: deck.sideboard.filter((card) => !(deck.basics || []).includes(card)),
      })));
  }

  fs.writeFileSync(`${destDir}/decks.json`, JSON.stringify(decks));

  console.log(`\tDone processing ${decks.length} decks.`);

}

const processPicks =  (numOracles) => {
  console.log('\tLoading picks...');

  // enumurate src/picks
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`);

  console.log(`\tLoaded ${pickFiles.length} pick files.`);
  console.log("\tWriting picks to file...");

  
  const fd = fs.openSync(`${destDir}/picks.json`, 'w');
  fs.writeSync(fd, '[');

  for (let i = 0; i < pickFiles.length; i++) {
    const picks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'))
      .filter((pick) => pick.pack.length > 3 && pick.pool.length > 3 && pick.pack.includes(pick.picked))
      .map((pick) => ({
        pool: pick.pool.filter((card) => card !== pick.picked),
        pick: pick.picked,
        pack: pick.pack,
      }));

    const serialized = JSON.stringify(picks);
      
    // trim the brackets
    fs.writeSync(fd, serialized.substring(1, serialized.length - 1));

    if (i < pickFiles.length - 1) {
      fs.writeSync(fd, ',');
    }
      
    console.log(`\t\tProcessed ${i} / ${pickFiles.length}`);
  }

  fs.writeSync(fd, ']');
  fs.closeSync(fd);
  

  console.log(`\tDone processing ${pickFiles.length} pick files.`);
}

const processOracleDict = () => {
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  fs.writeFileSync(`${destDir}/oracleDict.json`, JSON.stringify(indexToOracle));

  return indexToOracle.length;
}

const run =  () => {  
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir);
  }

  const numOracles = processOracleDict();

  console.log('Processing cubes...');
  processCubes(numOracles);

  console.log('Processing decks...');
  processDecks(numOracles);

  console.log('Processing picks...');
  processPicks(numOracles);

  console.log('Done!');
  process.exit(0);
}

run();
