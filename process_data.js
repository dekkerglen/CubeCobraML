const fs = require('fs');

const sourceDir = 'raw_data';
const trainDir = 'data/train';
const testDir = 'data/test';

const TEST_PERCENT = 0.1;


// this is for saving files larger than the string buffer limit
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

  fs.writeFileSync(`${trainDir}/cubes.json`, JSON.stringify(cubes.slice(0, Math.floor(cubes.length * (1-TEST_PERCENT)))));
  fs.writeFileSync(`${testDir}/cubes.json`, JSON.stringify(cubes.slice(Math.floor(cubes.length * (1-TEST_PERCENT)))));
  fs.writeFileSync(`${trainDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  fs.writeFileSync(`${testDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  
  console.log('\tDone processing cubes.');
}

const incrementCorrelation = (correlations, oracleIndex1, oracleIndex2, oracleCount) => {
  if (oracleIndex1 === oracleIndex2) {
    return;
  }

  const index1 = oracleIndex1 * oracleCount + oracleIndex2;
  const index2 = oracleIndex2 * oracleCount + oracleIndex1;

  correlations[index1]++;
  correlations[index2]++;
}

const processDecks = (oracleCount) => {
  console.log('\tLoading decks...');

  const correlations = new Int32Array(oracleCount * oracleCount);

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

    for (let j = 0; j < decks.length; j++) {
      for (let k = 0; k < decks[j].mainboard.length; k++) {
        for (let l = k + 1; l < decks[j].mainboard.length; l++) {
          incrementCorrelation(correlations, decks[j].mainboard[k], decks[j].mainboard[l], oracleCount);
        }
      }
    }

    console.log(`\tProcessed ${i + 1} of ${deckFiles.length} deck files.`);
  }

  fs.writeFileSync(`${trainDir}/decks.json`, JSON.stringify(decks.slice(0, Math.floor(decks.length * (1-TEST_PERCENT)))));
  fs.writeFileSync(`${testDir}/decks.json`, JSON.stringify(decks.slice(Math.floor(decks.length * (1-TEST_PERCENT)))));
  writeFile(`${trainDir}/correlations.json`, correlations);

  console.log(`\tDone processing ${decks.length} decks.`);
}

const processPicks =  (numOracles) => {
  console.log('\tLoading picks...');

  // enumurate src/picks
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`);

  console.log(`\tLoaded ${pickFiles.length} pick files.`);
  console.log("\tWriting picks to file...");

  
  const trainFile = fs.openSync(`${trainDir}/picks.json`, 'w');
  const testFile = fs.openSync(`${testDir}/picks.json`, 'w');
  fs.writeSync(trainFile, '[');
  fs.writeSync(testFile, '[');

  let wroteTrain = false;
  let wroteTest = false;

  for (let i = 0; i < pickFiles.length; i++) {
    const picks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'))
      .filter((pick) => 
        pick.pack.filter((index) => index !== -1).filter((index) => index).length > 1
         && pick.pool.filter((index) => index !== -1) 
         && pick.pack.includes(pick.picked) 
         && pick.picked !== -1
         && pick.picked
      )
      .map((pick) => ({
        pool: pick.pool.filter((card) => card !== pick.picked).filter((index) => index !== -1).filter((index) => index),
        pick: pick.picked,
        pack: pick.pack.filter((index) => index !== -1).filter((index) => index),
      }));

    const serialized = JSON.stringify(picks);
      
    if (i / pickFiles.length > TEST_PERCENT) {
      if (wroteTest) {
        fs.writeSync(trainFile, ',');
      } else {
        wroteTest = true;
      }
      fs.writeSync(trainFile, serialized.substring(1, serialized.length - 1));

    } else {
      if (wroteTrain) {
        fs.writeSync(testFile, ',');
      } else {
        wroteTrain = true;
      }
      fs.writeSync(testFile, serialized.substring(1, serialized.length - 1));
    }
    
    console.log(`\t\tProcessed ${i} / ${pickFiles.length}`);
  }

  fs.writeSync(trainFile, ']');
  fs.writeSync(testFile, ']');
  fs.closeSync(trainFile);
  fs.closeSync(testFile);
  

  console.log(`\tDone processing ${pickFiles.length} pick files.`);
}

const processOracleDict = () => {
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  fs.writeFileSync(`${trainDir}/oracleDict.json`, JSON.stringify(indexToOracle));
  fs.writeFileSync(`${testDir}/oracleDict.json`, JSON.stringify(indexToOracle));

  const simpleCardDict = JSON.parse(fs.readFileSync(`${sourceDir}/simpleCardDict.json`, 'utf8'));

  const elos = [];

  for (let i = 0; i < indexToOracle.length; i++) {
    const elo = simpleCardDict[indexToOracle[i]].elo || 1200;
    elos.push(Math.log(elo / 600));
  }

  // normalize elos
  const maxElo = Math.max(...elos);

  for (let i = 0; i < elos.length; i++) {
    elos[i] = elos[i] / maxElo;
  }

  fs.writeFileSync(`${testDir}/elos.json`, JSON.stringify(elos));
  fs.writeFileSync(`${trainDir}/elos.json`, JSON.stringify(elos));

  return indexToOracle.length;
}

const run =  () => {  
  if (!fs.existsSync(trainDir)) {
    fs.mkdirSync(trainDir);
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
