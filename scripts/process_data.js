const fs = require('fs');

const sourceDir = 'raw_data';
const trainDir = 'data/train';
const testDir = 'data/test';

const TEST_PERCENT = 0.0;
const WRITE_BATCH_SIZE = 10000;

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
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

/**
 * Process cubes from raw_data/cubes.json.
 * Returns { numCubes, cubeMap } where cubeMap is { cubeId: cardIndexArray }
 * so that deck processing can look up the cube's card list.
 */
const processCubes = (numOracles) => {
  console.log('\tLoading cubes...');

  let cubeCount = 0;

  const rawCubes = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'))
    .filter((cube) => cube.cards.length > 0);

  // Build a map from cube UUID → card list for deck processing
  const cubeMap = {};
  for (const cube of rawCubes) {
    cubeMap[cube.id] = cube.cards;
  }

  const cubes = rawCubes.map((cube) => cube.cards);

  const oracleFrequency = new Array(numOracles).fill(0);

  for (let i = 0; i < cubes.length; i++) {
    for (let j = 0; j < cubes[i].length; j++) {
      oracleFrequency[cubes[i][j]]++;
    }
  }

  console.log(`\tLoaded ${cubes.length} cubes.`);

  const train = cubes.slice(0, Math.floor(cubes.length * (1 - TEST_PERCENT)));
  const test = cubes.slice(Math.floor(cubes.length * (1 - TEST_PERCENT)));

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

  return { numCubes: cubeCount, cubeMap };
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

/**
 * Process decks from raw_data/decks/.
 * Each deck now includes cube_cards — the full card list of the cube it was
 * drafted from, looked up via the deck's `cube` UUID in cubeMap.
 */
const processDecks = (oracleCount, cubeMap) => {
  console.log('\tLoading decks...');

  let numDecks = 0;

  const correlations = new Int32Array(oracleCount * oracleCount).fill(0);

  // enumerate src/decks
  const deckFiles = fs.readdirSync(`${sourceDir}/decks`);

  console.log(`\tLoaded ${deckFiles.length} deck files.`);

  const decks = [];

  // Minimum mainboard size to accept a deck. Smaller than this and it's
  // garbage (token decks, art-card-only entries, etc.) that pollutes the
  // adjacency matrix enough to break the cube recommender's regularization.
  const MIN_MAINBOARD = 20;

  for (let i = 0; i < deckFiles.length; i++) {
    const newDecks = JSON.parse(fs.readFileSync(`${sourceDir}/decks/${deckFiles[i]}`, 'utf8'))
      .map((deck) => {
        const basics = deck.basics ? new Set(deck.basics) : null;
        // Strip both basics and the -1 sentinels cobra's export uses for empty slots.
        // Leaving -1 in caused the correlation loop to silently write to negative
        // indices (dropped) and mirrored cells one row off — producing ghost rows
        // and breaking the matrix's symmetry.
        const keep = (card) => card !== -1 && !(basics && basics.has(card));
        return {
          mainboard: deck.mainboard.filter(keep),
          sideboard: deck.sideboard.filter(keep),
          cube_cards: cubeMap[deck.cube] || [],
        };
      })
      .filter((deck) => deck.mainboard.length >= MIN_MAINBOARD);

    for (let j = 0; j < newDecks.length; j++) {
      const mb = newDecks[j].mainboard;
      for (let k = 0; k < mb.length; k++) {
        const a = mb[k];
        const rowA = a * oracleCount;
        for (let l = k + 1; l < mb.length; l++) {
          const b = mb[l];
          if (a === b) continue;
          correlations[rowA + b]++;
          correlations[b * oracleCount + a]++;
        }
      }
    }

    decks.push(...newDecks);

    console.log(`\tProcessed ${i + 1} of ${deckFiles.length} deck files.`);
  }

  const train = decks.slice(0, Math.floor(decks.length * (1 - TEST_PERCENT)));
  const test = decks.slice(Math.floor(decks.length * (1 - TEST_PERCENT)));

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

/**
 * Process picks from raw_data/picks/ alongside raw_data/cubeInstances/.
 *
 * To avoid duplicating cube context ~45× per draft, we keep a separate
 * cubeInstances/ directory that parallels picks/.  Each pick stores only a
 * small integer `cube_cards_idx` referencing into the cubeInstances array
 * for the same batch file.
 *
 * Output structure:
 *   picks/{n}.json       - array of { pool, pick, pack, cube_cards_idx }
 *   cubeInstances/{n}.json - array of card-index arrays (deduplicated per file)
 */
const processPicks = (numOracles) => {
  console.log('\tLoading picks...');

  let numPicks = 0;

  // enumerate src/picks
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`);

  console.log(`\tLoaded ${pickFiles.length} pick files.`);
  console.log("\tWriting picks to file...");

  ensureDir(`${trainDir}/picks`);
  ensureDir(`${testDir}/picks`);
  ensureDir(`${trainDir}/cubeInstances`);
  ensureDir(`${testDir}/cubeInstances`);

  // We collect picks into a flat train/test array, then batch-write them
  // along with a deduplicated cubeInstances array per output batch.
  let trainPicks = [];
  let trainCubeInstances = [];  // deduplicated cube card lists for current batch
  let trainCubeMap = new Map(); // "fileIdx:srcCubeIdx" → index into trainCubeInstances
  let trainBatchIdx = 0;

  let testPicks = [];
  let testCubeInstances = [];
  let testCubeMap = new Map();
  let testBatchIdx = 0;

  const flushTrain = () => {
    if (trainPicks.length === 0) return;
    writeFile(`${trainDir}/picks/${padLeft(trainBatchIdx, 4)}.json`, trainPicks);
    writeFile(`${trainDir}/cubeInstances/${padLeft(trainBatchIdx, 4)}.json`, trainCubeInstances);
    trainBatchIdx++;
    trainPicks = [];
    trainCubeInstances = [];
    trainCubeMap = new Map();
  };

  const flushTest = () => {
    if (testPicks.length === 0) return;
    writeFile(`${testDir}/picks/${padLeft(testBatchIdx, 4)}.json`, testPicks);
    writeFile(`${testDir}/cubeInstances/${padLeft(testBatchIdx, 4)}.json`, testCubeInstances);
    testBatchIdx++;
    testPicks = [];
    testCubeInstances = [];
    testCubeMap = new Map();
  };

  const getCubeIdx = (map, instances, key, cubeCards) => {
    const existing = map.get(key);
    if (existing !== undefined) return existing;
    const idx = instances.length;
    instances.push(cubeCards);
    map.set(key, idx);
    return idx;
  };

  for (let i = 0; i < pickFiles.length; i++) {
    // Load the corresponding cubeInstances file (parallel batch)
    const cubeInstancesPath = `${sourceDir}/cubeInstances/${pickFiles[i]}`;
    const cubeInstances = fs.existsSync(cubeInstancesPath)
      ? JSON.parse(fs.readFileSync(cubeInstancesPath, 'utf8'))
      : [];

    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    const picks = [];
    for (let p = 0; p < rawPicks.length; p++) {
      const pick = rawPicks[p];
      if (!pick.picked || pick.picked === -1) continue;
      if (!pick.pack.includes(pick.picked)) continue;

      const cleanPack = [];
      for (let q = 0; q < pick.pack.length; q++) {
        const v = pick.pack[q];
        if (v && v !== -1) cleanPack.push(v);
      }
      if (cleanPack.length <= 1) continue;

      const picked = pick.picked;
      const cleanPool = [];
      for (let q = 0; q < pick.pool.length; q++) {
        const v = pick.pool[q];
        if (v && v !== -1 && v !== picked) cleanPool.push(v);
      }

      picks.push({
        pool: cleanPool,
        pick: picked,
        pack: cleanPack,
        _srcCubeIdx: pick.cubeCards,
      });
    }

    const splitIdx = Math.floor(picks.length * (1 - TEST_PERCENT));
    numPicks += splitIdx;

    for (let t = 0; t < picks.length; t++) {
      const p = picks[t];
      const srcIdx = p._srcCubeIdx;
      const cubeCards = cubeInstances[srcIdx] || [];
      const key = `${i}:${srcIdx}`;

      if (t < splitIdx) {
        const cubeIdx = getCubeIdx(trainCubeMap, trainCubeInstances, key, cubeCards);
        trainPicks.push({ pool: p.pool, pick: p.pick, pack: p.pack, cube_cards_idx: cubeIdx });
        if (trainPicks.length >= WRITE_BATCH_SIZE) flushTrain();
      } else {
        const cubeIdx = getCubeIdx(testCubeMap, testCubeInstances, key, cubeCards);
        testPicks.push({ pool: p.pool, pick: p.pick, pack: p.pack, cube_cards_idx: cubeIdx });
        if (testPicks.length >= WRITE_BATCH_SIZE) flushTest();
      }
    }

    console.log(`\t\tProcessed ${i} / ${pickFiles.length}`);
  }

  // flush remaining
  flushTrain();
  flushTest();

  console.log(`\tDone processing ${pickFiles.length} pick files.`);

  return numPicks;
}

const processOracleDict = () => {
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  fs.writeFileSync(`${trainDir}/oracleDict.json`, JSON.stringify(indexToOracle));
  fs.writeFileSync(`${testDir}/oracleDict.json`, JSON.stringify(indexToOracle));

  const simpleCardDict = JSON.parse(fs.readFileSync(`${sourceDir}/simpleCardDict.json`, 'utf8'));

  const elos = [];
  const isLand = [];

  for (let i = 0; i < indexToOracle.length; i++) {
    const card = simpleCardDict[indexToOracle[i]] || {};
    const elo = card.elo || 1200;
    elos.push(Math.log(elo / 600));
    // Flag any card whose printed type contains "Land" (covers basics, non-basics,
    // gates, dual-faced lands, etc.). Used by the training loss to penalize the
    // draft head for picking a land when the human's true pick wasn't a land.
    const type = (card.type || '').trim();
    isLand.push(type.includes('Land') ? 1 : 0);
  }

  // normalize elos
  let maxElo = -Infinity;
  for (let i = 0; i < elos.length; i++) {
    if (elos[i] > maxElo) maxElo = elos[i];
  }

  for (let i = 0; i < elos.length; i++) {
    elos[i] = elos[i] / maxElo;
  }

  fs.writeFileSync(`${testDir}/elos.json`, JSON.stringify(elos));
  fs.writeFileSync(`${trainDir}/elos.json`, JSON.stringify(elos));
  fs.writeFileSync(`${trainDir}/isLand.json`, JSON.stringify(isLand));
  fs.writeFileSync(`${testDir}/isLand.json`, JSON.stringify(isLand));

  console.log(`\tFlagged ${isLand.reduce((a, b) => a + b, 0)} of ${indexToOracle.length} cards as lands.`);

  return indexToOracle.length;
}

const run = () => {  
  ensureDir(trainDir);
  ensureDir(testDir);

  const metadata = {};

  metadata.numOracles = processOracleDict();

  console.log("We have " + metadata.numOracles + " oracles.")

  console.log('Processing cubes...');
  const { numCubes, cubeMap } = processCubes(metadata.numOracles);
  metadata.numCubes = numCubes;

  console.log('Processing decks...');
  metadata.numDecks = processDecks(metadata.numOracles, cubeMap);

  console.log('Processing picks...');
  metadata.numPicks = processPicks(metadata.numOracles);

  fs.writeFileSync(`${trainDir}/metadata.json`, JSON.stringify(metadata));

  console.log('Done!');
  process.exit(0);
}

run();
