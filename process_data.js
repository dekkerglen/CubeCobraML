const fs = require('fs');

const sourceDir = 'raw_data';
const trainDir = 'data/train';
const testDir = 'data/test';

const TEST_PERCENT = 0.0;
const WRITE_BATCH_SIZE = 10000;

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }
}

const padLeft = (number, length) => {
  let str = '' + number;
  while (str.length < length) {
    str = '0' + str;
  }
  return str;
}

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

function writeLargeArray(filepath, arr) {  
  const fd = fs.openSync(filepath, 'w');
  
  fs.writeSync(fd, '[');

  const batchSize = 10000;

  for (let i = 0; i < arr.length; i += batchSize) {

    const slice = Array.from(arr.slice(i, i + batchSize));

    // need to make sure this is a list, not a weird map
    const serialized = JSON.stringify(slice);

    // trim the brackets
    fs.writeSync(fd, serialized.substring(1, serialized.length - 1));

    if (i + batchSize < arr.length) {
      fs.writeSync(fd, ',');
    }
  }

  
  fs.writeSync(fd, ']');
  fs.closeSync(fd);
}

const processCubes = (numOracles) => {
  console.log('\tLoading cubes...');

  let cubeCount = 0;

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

  const train =cubes.slice(0, Math.floor(cubes.length * (1-TEST_PERCENT)));
  const test = cubes.slice(Math.floor(cubes.length * (1-TEST_PERCENT)));

  ensureDir(`${trainDir}/cubes`);
  ensureDir(`${testDir}/cubes`);


  for (let i = 0; i < train.length / WRITE_BATCH_SIZE; i++) {
    cubeCount += train.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE).length;
    writeFile(`${trainDir}/cubes/${padLeft(i, 4)}.json`, train.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
  }

  for (let i = 0; i < test.length / WRITE_BATCH_SIZE; i++) {
    writeFile(`${testDir}/cubes/${padLeft(i, 4)}.json`, test.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
  }
  
  fs.writeFileSync(`${trainDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  fs.writeFileSync(`${testDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  
  console.log('\tDone processing cubes.');

  return cubeCount;
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

  let numDecks = 0;

  const correlations = new Int32Array(oracleCount * oracleCount).fill(0);

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

  const train = decks.slice(0, Math.floor(decks.length * (1-TEST_PERCENT)));
  const test = decks.slice(Math.floor(decks.length * (1-TEST_PERCENT)));

  ensureDir(`${trainDir}/decks`);
  ensureDir(`${testDir}/decks`);

  for (let i = 0; i < train.length / WRITE_BATCH_SIZE; i++) {
    numDecks += train.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE).length;
    writeFile(`${trainDir}/decks/${padLeft(i, 4)}.json`, train.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
  }

  for (let i = 0; i < test.length / WRITE_BATCH_SIZE; i++) {
    writeFile(`${testDir}/decks/${padLeft(i, 4)}.json`, test.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
  }

  writeLargeArray(`${trainDir}/correlations.json`, correlations);

  console.log(`\tDone processing ${decks.length} decks.`);

  return numDecks;
}

const processPicks =  (numOracles) => {
  console.log('\tLoading picks...');

  let numPicks = 0;

  // enumurate src/picks
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`);

  console.log(`\tLoaded ${pickFiles.length} pick files.`);
  console.log("\tWriting picks to file...");

  ensureDir(`${trainDir}/picks`);
  ensureDir(`${testDir}/picks`);

  let trainIndex = 0;
  let testIndex = 0;

  let trainSize = 0;
  let testSize = 0;

  const nextTrainFile = () => {
    const trainFile = fs.openSync(`${trainDir}/picks/${padLeft(trainIndex, 4)}.json`, 'w');
    fs.writeSync(trainFile, '[');
    return trainFile;
  }

  const nextTestFile = () => {
    const testFile = fs.openSync(`${testDir}/picks/${padLeft(testIndex, 4)}.json`, 'w');
    fs.writeSync(testFile, '[');
    return testFile;
  }

  const closeFile = (file) => {
    fs.writeSync(file, ']');
    fs.closeSync(file);
  } 

  let trainFile = nextTrainFile();
  let testFile = nextTestFile();

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

    const train = picks.slice(0, Math.floor(picks.length * (1-TEST_PERCENT)));
    const test = picks.slice(Math.floor(picks.length * (1-TEST_PERCENT)));
    
    numPicks += train.length;

    if (train.length > 0) {
      if (trainSize > 0) {
        fs.writeFileSync(trainFile, ',');
      }

      if (trainSize + train.length > WRITE_BATCH_SIZE) {
        const toAppendSerialized = JSON.stringify(train.slice(0, WRITE_BATCH_SIZE - trainSize));
        const toWrite = train.slice(WRITE_BATCH_SIZE - trainSize);
        const toWriteSerialized = JSON.stringify(toWrite);        

        fs.writeFileSync(trainFile, toAppendSerialized.substring(1, toAppendSerialized.length - 1));
        closeFile(trainFile);

        trainIndex++;
        trainFile = nextTrainFile();
        fs.writeFileSync(trainFile, toWriteSerialized.substring(1, toWriteSerialized.length - 1));
        trainSize = toWrite.length;


      } else {
        const serialized = JSON.stringify(train);

        fs.writeFileSync(trainFile, serialized.substring(1, serialized.length - 1));
        trainSize += train.length;
      }
    }
    
    if (test.length > 0) {
      if (testSize + test.length > WRITE_BATCH_SIZE) {
        const toAppendSerialized = JSON.stringify(test.slice(0, WRITE_BATCH_SIZE - testSize));
        const toWrite = test.slice(WRITE_BATCH_SIZE - testSize);
        const toWriteSerialized = JSON.stringify(toWrite);

        fs.writeFileSync(testFile, toAppendSerialized.substring(1, toAppendSerialized.length - 1));
        closeFile(testFile);

        testIndex++;
        testFile = nextTestFile();
        fs.writeFileSync(testFile, toWriteSerialized.substring(1, toWriteSerialized.length - 1));
        testSize = toWrite.length;

        if (testSize < WRITE_BATCH_SIZE) {
          fs.writeFileSync(testFile, ',');
        }
      } else {
        const serialized = JSON.stringify(test);

        fs.writeFileSync(testFile, serialized.substring(1, serialized.length - 1));
        testSize += test.length;

        if (testSize < WRITE_BATCH_SIZE) {
          fs.writeFileSync(testFile, ',');
        }
      }
    }

    console.log(`\t\tProcessed ${i} / ${pickFiles.length}`);
  }

  closeFile(trainFile);
  closeFile(testFile);

  console.log(`\tDone processing ${pickFiles.length} pick files.`);

  return numPicks;
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

  const metadata = {};

  metadata.numOracles = processOracleDict();

  console.log("We have " + metadata.numOracles + " oracles.")

  // console.log('Processing cubes...');
  // metadata.numCubes = processCubes(metadata.numOracles);

  // console.log('Processing decks...');
  // metadata.numDecks = processDecks(metadata.numOracles);

  console.log('Processing picks...');
  metadata.numPicks = processPicks(metadata.numOracles);

  fs.writeFileSync(`${trainDir}/metadata.json`, JSON.stringify(metadata));

  console.log('Done!');
  process.exit(0);
}

run();
