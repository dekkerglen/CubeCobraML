const fs = require('fs');

const trainDir = 'data/train';

const processOracleDict = () => {
  const oracleDict = JSON.parse(fs.readFileSync(`${trainDir}/oracleDict.json`));
  return oracleDict.length;
}

const processCubes = () => {
  let numCubes = 0;

  const cubeFiles = fs.readdirSync(`${trainDir}/cubes`);

  for (let i = 0; i < cubeFiles.length; i++) {
    const cube = JSON.parse(fs.readFileSync(`${trainDir}/cubes/${cubeFiles[i]}`));
    numCubes += cube.length;

    console.log(`Processing cube ${i + 1} of ${cubeFiles.length}...`);
  }

  return numCubes;
}

const processDecks = () => {
  let numDecks = 0;

  const deckFiles = fs.readdirSync(`${trainDir}/decks`);

  for (let i = 0; i < deckFiles.length; i++) {
    const deck = JSON.parse(fs.readFileSync(`${trainDir}/decks/${deckFiles[i]}`));
    numDecks += deck.length;

    console.log(`Processing deck ${i + 1} of ${deckFiles.length}...`);
  }

  return numDecks;
}

const processPicks = () => {
  let numPicks = 0;

  const pickFiles = fs.readdirSync(`${trainDir}/picks`);

  for (let i = 0; i < pickFiles.length; i++) {
    const pick = JSON.parse(fs.readFileSync(`${trainDir}/picks/${pickFiles[i]}`));
    numPicks += pick.length;

    console.log(`Processing pick ${i + 1} of ${pickFiles.length}...`);
  }

  return numPicks;
}

const run =  () => {  
  if (!fs.existsSync(trainDir)) {
    fs.mkdirSync(trainDir);
  }

  const metadata = {};

  metadata.numOracles = processOracleDict();

  console.log('Processing cubes...');
  metadata.numCubes = processCubes();

  console.log('Processing decks...');
  metadata.numDecks = processDecks();

  console.log('Processing picks...');
  metadata.numPicks = processPicks();

  fs.writeFileSync(`${trainDir}/metadata.json`, JSON.stringify(metadata));

  console.log('Done!');
  process.exit(0);
}

run();
