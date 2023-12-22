const fs = require('fs');
const util = require('util');
const ml = require('./ml.js');

// Convert fs.writeFile into Promise version to use with async/await
const writeFile = util.promisify(fs.writeFile);

const one_hot_encoding = (index, numOracles) => {
  const tensor = new Array(numOracles).fill(0);

  tensor[index] = 1;

  return tensor;
}

async function compileEmbeddings() {
  // load indexToOracleMap.json
  const indexToOracle = JSON.parse(fs.readFileSync("indexToOracleMap.json"));

  const numOracles = Object.keys(indexToOracle).length;

  console.log(`Compiling encodings for ${numOracles} oracles`);

  // wait 10 seconds
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Initialize array to hold encodings
  let embeddings = [];

  // Process in batches
  const batchSize = 100; // Adjust batch size as needed
  for (let i = 0; i < numOracles; i += batchSize) {
    console.log(`Processing batch ${i / batchSize + 1} of ${Math.ceil(numOracles / batchSize)}`);

    const encodings = [];
    for (let j = i; j < Math.min(i + batchSize, numOracles); j++) {
      encodings.push(one_hot_encoding(j, numOracles));
    }
    const batchEncodings = await ml.encode(encodings);

    embeddings = embeddings.concat(batchEncodings);
  }

  // Save encodings to JSON file
  await writeFile('./embeddings.json', JSON.stringify(embeddings));

  console.log('Encodings compiled and saved to encodings.json');
}

// Run the function
compileEmbeddings().catch(console.error);
