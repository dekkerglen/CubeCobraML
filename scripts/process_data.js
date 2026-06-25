const fs = require('fs');
const crypto = require('crypto');
const { sampleValDrafts: sampleValDraftsCore } = require('./lib/val_sampler');

const sourceDir = 'raw_data';
const dataDir = 'data';
const trainDir = 'data/train';
const testDir = 'data/test';

// Stable 12-hex-char id from any string. Used to mint synthetic draft_ids.
const shortHash = (s) => crypto.createHash('sha1').update(s).digest('hex').slice(0, 12);

// =============================================================================
// Configuration
// =============================================================================
//
// Split granularity is PER-DRAFT, identified by (fileIdx, owner, cube, cubeCards).
// A single (owner, cube) pair often hosts multiple drafts distinguished by
// cubeCards (cube snapshot index, file-local).
//
// Cubes are asymmetric: train/cubes = ALL cubes; test/cubes = cubes referenced
// by val drafts only.
//
// Decks are routed by CONTENT MATCHING: each deck's (mainboard ∪ sideboard) is
// a superset of its source draft's picked_set, so we content-match per file to
// find the deck's true source draft. Yields val_decks ≈ 1-2× val_drafts.
//
// The val sampler aims for TWO goals simultaneously:
//   (1) DISTRIBUTION: per-card pick rate in val ≈ per-card pick rate in full
//       dataset (so val metrics generalize).
//   (2) COVERAGE: every card with totalPicks ≥ COVERAGE_THRESHOLD must have
//       at least one pick in val (so we can measure model behavior on it).
//
// Algorithm — Random baseline + coverage top-up:
//
//   Phase A — Random baseline (gives goal 1 for sufficiently common cards):
//     Shuffle drafts deterministically. Walk and accept until total picks reach
//     RANDOM_TARGET_PICKS. By construction, val pick rate matches full pick
//     rate up to sampling noise. Cards with totalPicks < ~500 are likely
//     missed.
//
//   Phase B — Coverage top-up (achieves goal 2 with minimum bias):
//     For each uncovered eligible card in ASCENDING totalPicks order, pick the
//     unselected draft containing it whose marginal contribution to OTHER
//     uncovered cards is largest. Each Phase B draft naturally cascades
//     coverage to many rare cards at once, minimizing the total drafts added.
//
//     Hard-cap total at HARD_CAP_PICKS as a safety net.
//
// Per-card "needed" is just 1 — presence is enough for the eval metric.
// Distributional match is the job of Phase A; Phase B fills the long tail.

const RANDOM_TARGET_PICKS = parseInt(process.env.RANDOM_TARGET_PICKS ?? '300000', 10);
const HARD_CAP_PICKS = parseInt(process.env.HARD_CAP_PICKS ?? '500000', 10);
// Cards with totalPicks below this threshold are NOT force-included. The math:
// forcing a single pick of a card with totalPicks=T into a val of V picks
// gives ratio = full_total / (V * T). For default V≈300k, threshold=400 keeps
// the per-card distortion at <= ~1.1x. Lowering this floods val with rare
// cards and inflates the "rare" band ratio drastically. Cards below this floor
// are statistically uninformative anyway — totalPicks < 400 → fewer than 2 val
// picks even with proportional sampling.
const COVERAGE_THRESHOLD = parseInt(process.env.COVERAGE_THRESHOLD ?? '400', 10);
const BOMB_COUNT = parseInt(process.env.BOMB_COUNT ?? '100', 10);
const RANDOM_SEED = parseInt(process.env.RANDOM_SEED ?? '42', 10);
const WRITE_BATCH_SIZE = 10000;

// Cube-size filter: cubes outside [CUBE_SIZE_MIN, CUBE_SIZE_MAX] cards are
// treated as garbage and EVERYTHING tied to their UUID is dropped — picks,
// decks, cubeInstances. The encoder is fed inputs ranging from binary masks
// of multi-thousand-card "cubes" (likely all-sets test data) down to 1-card
// stubs; that variance destabilizes the encoder. Bounding here propagates
// down: train/test data, val sampler, derived sidecars all see only valid
// cubes.
const CUBE_SIZE_MIN = parseInt(process.env.CUBE_SIZE_MIN ?? '180', 10);
const CUBE_SIZE_MAX = parseInt(process.env.CUBE_SIZE_MAX ?? '1080', 10);

/**
 * Read raw_data/cubes.json once and return:
 *   - validCubeIds: Set<UUID> of cubes whose `cards.length` is in
 *     [CUBE_SIZE_MIN, CUBE_SIZE_MAX]
 *   - stats: { total, kept, droppedSmall, droppedLarge }
 *
 * Must be called BEFORE pickPass1 / processCubes / processDecks so that
 * every downstream stage filters consistently on the same set.
 */
const loadValidCubeIds = () => {
  const rawCubes = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'));
  const valid = new Set();
  let droppedSmall = 0;
  let droppedLarge = 0;
  for (const cube of rawCubes) {
    const n = cube.cards ? cube.cards.length : 0;
    if (n < CUBE_SIZE_MIN) { droppedSmall++; continue; }
    if (n > CUBE_SIZE_MAX) { droppedLarge++; continue; }
    valid.add(cube.id);
  }
  return {
    validCubeIds: valid,
    stats: { total: rawCubes.length, kept: valid.size, droppedSmall, droppedLarge },
  };
};

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
};

// Re-create a directory empty. Used for output subdirs (picks/, decks/, etc.)
// so stale files from previous runs (which may have produced different file
// counts) don't pollute the new output.
const resetDir = (dir) => {
  if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
};

const padLeft = (number, length) => {
  let str = '' + number;
  while (str.length < length) str = '0' + str;
  return str;
};

// Large-array writers — file-by-file streaming avoids hitting V8's string buffer
// limit on JSON.stringify of >500MB payloads.
function writeFile(filepath, data) {
  const fd = fs.openSync(filepath, 'w');
  fs.writeSync(fd, '[');
  const batchSize = 10000;
  for (let i = 0; i < data.length; i += batchSize) {
    const serialized = JSON.stringify(data.slice(i, i + batchSize));
    fs.writeSync(fd, serialized.substring(1, serialized.length - 1));
    if (i + batchSize < data.length) fs.writeSync(fd, ',');
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
    const serialized = JSON.stringify(slice);
    fs.writeSync(fd, serialized.substring(1, serialized.length - 1));
    if (i + batchSize < arr.length) fs.writeSync(fd, ',');
  }
  fs.writeSync(fd, ']');
  fs.closeSync(fd);
}

// Validate a raw pick record. Returns the picked card id if valid, else null.
// Shared between pass 1 (counting) and pass 2 (writing) so the two passes see
// identical record acceptance rules.
const validatePick = (pick) => {
  if (!pick.picked || pick.picked === -1) return null;
  if (!pick.pack.includes(pick.picked)) return null;
  let packSize = 0;
  for (let q = 0; q < pick.pack.length; q++) {
    const v = pick.pack[q];
    if (v && v !== -1) packSize++;
  }
  if (packSize <= 1) return null;
  return pick.picked;
};

// =============================================================================
// Pass 1: gather pickCount + per-draft summaries + card→drafts inverted index
// =============================================================================

/**
 * Walk every raw pick file once. For each valid pick:
 *   - increment pickCount[card]
 *   - find or create the draft summary for (file, owner, cube, cubeCards)
 *   - increment that draft's per-card pick count
 *
 * After all files processed, build cardToDrafts[c] = Int32Array of draft IDs
 * (indices into draftSummaries) that contain at least one pick of card c.
 * This inverted index is essential for the card-by-card coverage sampler — we
 * need O(1) access to "which drafts can cover this rare card?"
 *
 * Memory footprint:
 *   - draftSummaries: ~2.5M drafts × ~30 unique cards in Map = ~600MB
 *   - cardToDrafts:   ~75M total (card, draftId) edges × 4 bytes = ~300MB
 * Total well under 16GB heap.
 *
 * Returns { pickCount, draftSummaries, cardToDrafts }.
 */
const pickPass1 = (numOracles, validCubeIds) => {
  console.log('\tPick pass 1: scanning for pickCount + per-draft summaries...');
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`).sort();
  const pickCount = new Array(numOracles).fill(0);
  const draftsByKey = new Map(); // key → draft summary

  let totalValid = 0;
  let droppedByCubeFilter = 0;
  for (let i = 0; i < pickFiles.length; i++) {
    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    for (let p = 0; p < rawPicks.length; p++) {
      const pick = rawPicks[p];
      const picked = validatePick(pick);
      if (picked === null) continue;

      const owner = pick.owner || '_anon';
      const cube = pick.cube || '_nocube';
      // Cube filter: drop picks tied to invalid cubes BEFORE accounting.
      // pickCount, draftSummaries, and the inverted index must reflect only
      // post-filter data so the sampler / coverage math is correct.
      if (!validCubeIds.has(cube)) { droppedByCubeFilter++; continue; }

      pickCount[picked]++;
      totalValid++;

      const key = `${i}::${owner}::${cube}::${pick.cubeCards}`;
      let draft = draftsByKey.get(key);
      if (draft === undefined) {
        draft = { key, owner, cube, nPicks: 0, cardCounts: new Map() };
        draftsByKey.set(key, draft);
      }
      draft.nPicks++;
      draft.cardCounts.set(picked, (draft.cardCounts.get(picked) || 0) + 1);
    }
    if ((i + 1) % 100 === 0 || i + 1 === pickFiles.length) {
      console.log(`\t\tPass 1: ${i + 1}/${pickFiles.length} files, ` +
        `${draftsByKey.size} drafts, ${totalValid} valid picks so far ` +
        `(${droppedByCubeFilter} dropped by cube filter)`);
    }
  }

  const draftSummaries = Array.from(draftsByKey.values());
  console.log(`\tPass 1: ${totalValid} valid picks across ${draftSummaries.length} drafts ` +
    `(${droppedByCubeFilter} picks dropped by cube-size filter).`);

  // Build inverted index card → draft IDs. Two-pass: first count incidences per
  // card to size typed arrays exactly, then fill.
  console.log('\tBuilding card→drafts inverted index...');
  const counts = new Int32Array(numOracles); // per-card incidence count
  for (let d = 0; d < draftSummaries.length; d++) {
    for (const card of draftSummaries[d].cardCounts.keys()) counts[card]++;
  }
  const cardToDrafts = new Array(numOracles);
  for (let c = 0; c < numOracles; c++) {
    if (counts[c] > 0) cardToDrafts[c] = new Int32Array(counts[c]);
  }
  const fillIdx = new Int32Array(numOracles); // current fill cursor per card
  let totalEdges = 0;
  for (let d = 0; d < draftSummaries.length; d++) {
    for (const card of draftSummaries[d].cardCounts.keys()) {
      cardToDrafts[card][fillIdx[card]++] = d;
      totalEdges++;
    }
  }
  console.log(`\tInverted index built: ${totalEdges} card→draft edges.`);

  return { pickCount, draftSummaries, cardToDrafts };
};

// =============================================================================
// Sampler: pick which drafts go to val
// =============================================================================

// The validation-split sampler lives in ./lib/val_sampler.js. This thin
// wrapper binds the module-level config consts so the call sites below
// stay unchanged. See that module for the algorithm.
const sampleValDrafts = (draftSummaries, cardToDrafts, pickCount) =>
  sampleValDraftsCore(draftSummaries, cardToDrafts, pickCount, {
    coverageThreshold: COVERAGE_THRESHOLD,
    randomTargetPicks: RANDOM_TARGET_PICKS,
    hardCapPicks: HARD_CAP_PICKS,
    randomSeed: RANDOM_SEED,
    bombCount: BOMB_COUNT,
  });

// =============================================================================
// Pass 2: re-read raw picks and split by valDraftKeys
// =============================================================================

/**
 * Walk every raw pick file again. For each valid pick, route to train or test
 * based on whether its (owner, cube) draft is in valDraftKeys. Writes batched
 * picks/*.json + cubeInstances/*.json files for both splits.
 *
 * Output pick records DROP landCount/nonlandCount (not used by the new
 * architecture). Shape is { pool, pick, pack, cube_cards_idx }.
 *
 * Returns { numTrainPicks, numTestPicks, valCubeIds: Set<string> }.
 * valCubeIds is needed for cube splitting downstream.
 */
const pickPass2 = (numOracles, valDraftKeys, validCubeIds) => {
  console.log('\tPick pass 2: routing + writing picks...');
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`).sort();

  resetDir(`${trainDir}/picks`);
  resetDir(`${testDir}/picks`);
  resetDir(`${trainDir}/cubeInstances`);
  resetDir(`${testDir}/cubeInstances`);

  let numTrainPicks = 0;
  let numTestPicks = 0;
  const valCubeIds = new Set();

  // Phase J / Phase S: per-session accumulation of pick global indices.
  // Session key = `${draftKey}::start=${firstPickGlobalIdx}` so the same
  // (file, owner, cube, cubeCards) tuple can host N back-to-back anonymous
  // drafters as N distinct sessions, split at pool-growth boundaries. For
  // named single-drafter draftKeys this produces one session per draftKey.
  // Memory peak for train ~1.5GB at 3M drafts × ~500B — needs ≥24GB Node heap
  // (see --max-old-space-size).
  const valDraftSessions = new Map();    // sessionKey -> {cube_uuid, owner, picks}
  const trainDraftSessions = new Map();
  // Per-draftKey state for the pool-growth boundary detector. lastPoolSize
  // tracks the previous pick's pool size; sessionKey points to the currently
  // open session in *DraftSessions. A new session starts when pool size
  // returns to 0 after being >0 (= a new drafter's first pick).
  const valOpenSessions = new Map();   // draftKey -> {sessionKey, lastPoolSize}
  const trainOpenSessions = new Map();

  let trainPicks = [], trainCubeInstances = [], trainCubeMap = new Map(), trainBatchIdx = 0;
  let testPicks = [], testCubeInstances = [], testCubeMap = new Map(), testBatchIdx = 0;

  const flushTrain = () => {
    if (trainPicks.length === 0) return;
    writeFile(`${trainDir}/picks/${padLeft(trainBatchIdx, 4)}.json`, trainPicks);
    writeFile(`${trainDir}/cubeInstances/${padLeft(trainBatchIdx, 4)}.json`, trainCubeInstances);
    trainBatchIdx++;
    trainPicks = []; trainCubeInstances = []; trainCubeMap = new Map();
  };
  const flushTest = () => {
    if (testPicks.length === 0) return;
    writeFile(`${testDir}/picks/${padLeft(testBatchIdx, 4)}.json`, testPicks);
    writeFile(`${testDir}/cubeInstances/${padLeft(testBatchIdx, 4)}.json`, testCubeInstances);
    testBatchIdx++;
    testPicks = []; testCubeInstances = []; testCubeMap = new Map();
  };

  const getCubeIdx = (map, instances, key, cubeCards) => {
    const existing = map.get(key);
    if (existing !== undefined) return existing;
    const idx = instances.length;
    instances.push(cubeCards);
    map.set(key, idx);
    return idx;
  };

  // Pool-growth boundary detector — returns the open session object for the
  // current pick, opening a new one if this pick is the first of a fresh
  // drafter. Caller appends the global pick index to the returned `picks`.
  const openSessionForPick = (draftKey, cube, owner, poolSize, globalIdx,
                              sessionsMap, openMap) => {
    let open = openMap.get(draftKey);
    const isNewSession = open === undefined || (poolSize === 0 && open.lastPoolSize > 0);
    if (isNewSession) {
      const sessionKey = `${draftKey}::start=${globalIdx}`;
      const sess = { cube_uuid: cube, owner: owner, picks: [] };
      sessionsMap.set(sessionKey, sess);
      open = { sessionKey, lastPoolSize: poolSize };
      openMap.set(draftKey, open);
      return sess;
    }
    open.lastPoolSize = poolSize;
    return sessionsMap.get(open.sessionKey);
  };
  const openValSession = (draftKey, cube, owner, poolSize, globalIdx, sessionsMap, openMap) =>
    openSessionForPick(draftKey, cube, owner, poolSize, globalIdx, sessionsMap, openMap);
  const openTrainSession = (draftKey, cube, owner, poolSize, globalIdx, sessionsMap, openMap) =>
    openSessionForPick(draftKey, cube, owner, poolSize, globalIdx, sessionsMap, openMap);

  for (let i = 0; i < pickFiles.length; i++) {
    const cubeInstancesPath = `${sourceDir}/cubeInstances/${pickFiles[i]}`;
    const cubeInstances = fs.existsSync(cubeInstancesPath)
      ? JSON.parse(fs.readFileSync(cubeInstancesPath, 'utf8'))
      : [];

    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    for (let p = 0; p < rawPicks.length; p++) {
      const pick = rawPicks[p];
      const picked = validatePick(pick);
      if (picked === null) continue;

      // Must use the SAME key formula as Pass 1 so draft membership is
      // consistent: (file index, owner, cube, cubeCards).
      const owner = pick.owner || '_anon';
      const cube = pick.cube || '_nocube';
      // Cube filter: must match Pass 1 exactly or draftKey membership
      // diverges and picks get misrouted.
      if (!validCubeIds.has(cube)) continue;
      const draftKey = `${i}::${owner}::${cube}::${pick.cubeCards}`;
      const isVal = valDraftKeys.has(draftKey);

      const cleanPack = [];
      for (let q = 0; q < pick.pack.length; q++) {
        const v = pick.pack[q];
        if (v && v !== -1) cleanPack.push(v);
      }
      const cleanPool = [];
      for (let q = 0; q < pick.pool.length; q++) {
        const v = pick.pool[q];
        if (v && v !== -1 && v !== picked) cleanPool.push(v);
      }

      const srcIdx = pick.cubeCards;
      const cubeCards = cubeInstances[srcIdx] || [];
      const ciKey = `${i}:${srcIdx}`;

      const record = { pool: cleanPool, pick: picked, pack: cleanPack };
      // Pool-growth split: anonymous drafters of the same cube snapshot share a
      // draftKey but their picks are concatenated rather than interleaved. A
      // new session begins whenever pool size returns to 0 after being non-zero
      // (= a new drafter's first pick). For named single-drafter cases this is
      // a no-op since pool resets only at the very start.
      const poolSize = cleanPool.length;
      if (isVal) {
        record.cube_cards_idx = getCubeIdx(testCubeMap, testCubeInstances, ciKey, cubeCards);
        const sess = openValSession(draftKey, cube, owner, poolSize, numTestPicks, valDraftSessions, valOpenSessions);
        sess.picks.push(numTestPicks);
        testPicks.push(record);
        numTestPicks++;
        valCubeIds.add(cube);
        if (testPicks.length >= WRITE_BATCH_SIZE) flushTest();
      } else {
        record.cube_cards_idx = getCubeIdx(trainCubeMap, trainCubeInstances, ciKey, cubeCards);
        const sess = openTrainSession(draftKey, cube, owner, poolSize, numTrainPicks, trainDraftSessions, trainOpenSessions);
        sess.picks.push(numTrainPicks);
        trainPicks.push(record);
        numTrainPicks++;
        if (trainPicks.length >= WRITE_BATCH_SIZE) flushTrain();
      }
    }
    if ((i + 1) % 100 === 0 || i + 1 === pickFiles.length) {
      console.log(`\t\tPass 2: ${i + 1}/${pickFiles.length} files, ${numTrainPicks} train, ${numTestPicks} test`);
    }
  }
  flushTrain();
  flushTest();

  // Phase J: emit val draft sessions sidecar.
  writeDraftSessions(valDraftSessions);

  // Phase O: emit train draft sessions JSONL + idx.bin + manifest, plus the
  // small cube_drafts_index.json. Done at the end so disk I/O for the heavy
  // pass2 output finishes first.
  writeTrainDraftSidecars(trainDraftSessions);

  console.log(`\tPick pass 2 done: ${numTrainPicks} train, ${numTestPicks} test.`);
  return { numTrainPicks, numTestPicks, valCubeIds };
};

// Phase J: write data/test/draft_sessions.json from a Map<draftKey, {cube_uuid,
// owner, picks[]}>. draft_id = shortHash(draftKey) — stable across reprocesses.
// `synthetic` is false because picks were grouped from raw (file, owner, cube,
// cubeCards) keys (true draft identity). Extracted so --drafts-only bootstrap
// can also call out to a parallel pool-growth implementation that produces the
// same on-disk shape.
function writeDraftSessions(valDraftSessions) {
  const sessions = {};
  let totalPicks = 0;
  for (const [draftKey, sess] of valDraftSessions) {
    const id = shortHash(draftKey);
    sessions[id] = {
      cube_uuid: sess.cube_uuid,
      owner: sess.owner,
      picks: sess.picks,
      synthetic: false,
      // Phase S: threshold relaxed from <10 to <5 — pool-growth splitting now
      // routes mid-length anonymous sessions to their own draft_id, so a real
      // sealed pool / early-quit at 5-9 picks shouldn't be flagged.
      suspect: sess.picks.length < 5,
    };
    totalPicks += sess.picks.length;
  }
  fs.writeFileSync(`${testDir}/draft_sessions.json`, JSON.stringify({ sessions }));
  console.log(`\tDraft sessions: ${Object.keys(sessions).length} drafts (${totalPicks} picks) `
    + `written to ${testDir}/draft_sessions.json.`);
}

// Phase O: emit train sidecars for cube-scoped lookups.
//
// Three coordinated files:
//   - draft_sessions.jsonl       — one JSON object per line, sorted by draft_id
//   - draft_sessions.idx.bin     — packed (draft_id_12B, offset_u64, length_u32)
//                                  sorted by draft_id for O(log N) bisect
//   - draft_sessions.manifest.json — {n_records, jsonl_sha256, built_at, schema_version}
// Plus the small cube_drafts_index.json: {cube_uuid: [draft_id,...]}.
//
// All four are written to .tmp paths first and atomically renamed at the end.
// The backend service refuses to load a JSONL whose sha256 doesn't match the
// manifest — protects against partial bootstrap crashes serving bad data.
//
// idx.bin record layout (24 bytes each):
//   bytes  0..12  draft_id (12 ASCII hex chars)
//   bytes 12..20  offset_u64 little-endian
//   bytes 20..24  length_u32 little-endian
const DRAFT_SESSION_SCHEMA = 1;
const DRAFT_ID_BYTES = 12;
const IDX_RECORD_BYTES = DRAFT_ID_BYTES + 8 + 4;

function writeTrainDraftSidecars(trainDraftSessions) {
  if (trainDraftSessions.size === 0) {
    console.log('\tTrain draft sidecars: no train drafts found; skipping.');
    return;
  }
  console.log(`\tTrain draft sidecars: preparing ${trainDraftSessions.size} drafts...`);

  // Hash each draftKey and sort by draft_id so the binary index supports
  // bisect. Frees the original Map progressively.
  const entries = new Array(trainDraftSessions.size);
  {
    let i = 0;
    for (const [draftKey, sess] of trainDraftSessions) {
      entries[i++] = {
        draft_id: shortHash(draftKey),
        cube_uuid: sess.cube_uuid,
        owner: sess.owner,
        picks: sess.picks,
      };
    }
  }
  // Allow the original map to GC; we don't need it again.
  trainDraftSessions.clear();
  if (global.gc) global.gc();

  entries.sort((a, b) => a.draft_id < b.draft_id ? -1 : a.draft_id > b.draft_id ? 1 : 0);

  // Cube → draft_ids index (small file, JSON).
  const cubeDrafts = {};
  for (const e of entries) {
    if (!cubeDrafts[e.cube_uuid]) cubeDrafts[e.cube_uuid] = [];
    cubeDrafts[e.cube_uuid].push(e.draft_id);
  }

  const jsonlPath  = `${trainDir}/draft_sessions.jsonl`;
  const idxPath    = `${trainDir}/draft_sessions.idx.bin`;
  const manifestP  = `${trainDir}/draft_sessions.manifest.json`;
  const cubeIdxP   = `${trainDir}/cube_drafts_index.json`;
  const tmpJsonl   = jsonlPath + '.tmp';
  const tmpIdx     = idxPath + '.tmp';
  const tmpManif   = manifestP + '.tmp';
  const tmpCubeIdx = cubeIdxP + '.tmp';

  // Stream JSONL + idx records in one pass.
  const fdJsonl = fs.openSync(tmpJsonl, 'w');
  const idxBuf  = Buffer.alloc(entries.length * IDX_RECORD_BYTES);
  const hasher  = crypto.createHash('sha256');
  let offset = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    // suspect retained for parity with val (UI filter); threshold relaxed to
    // <5 (Phase S) — pool-growth splitting already isolates anonymous drafts.
    const body = {
      cube_uuid: e.cube_uuid,
      owner: e.owner,
      picks: e.picks,
      synthetic: false,
      suspect: e.picks.length < 5,
    };
    const line = e.draft_id + '\t' + JSON.stringify(body) + '\n';
    const buf = Buffer.from(line, 'utf8');
    fs.writeSync(fdJsonl, buf);
    hasher.update(buf);
    const base = i * IDX_RECORD_BYTES;
    idxBuf.write(e.draft_id, base, DRAFT_ID_BYTES, 'ascii');
    idxBuf.writeBigUInt64LE(BigInt(offset), base + DRAFT_ID_BYTES);
    idxBuf.writeUInt32LE(buf.length, base + DRAFT_ID_BYTES + 8);
    offset += buf.length;
    if ((i + 1) % 200000 === 0) {
      console.log(`\t\t...${i + 1}/${entries.length} drafts written (${(offset / 1e9).toFixed(2)} GB)`);
    }
  }
  fs.closeSync(fdJsonl);
  fs.writeFileSync(tmpIdx, idxBuf);

  const manifest = {
    n_records: entries.length,
    jsonl_sha256: hasher.digest('hex'),
    jsonl_bytes: offset,
    schema_version: DRAFT_SESSION_SCHEMA,
    built_at: new Date().toISOString(),
  };
  fs.writeFileSync(tmpManif, JSON.stringify(manifest, null, 2));
  fs.writeFileSync(tmpCubeIdx, JSON.stringify(cubeDrafts));

  // Atomic flip — anything reading the canonical paths sees either the old or
  // new versions, never a partial file.
  fs.renameSync(tmpJsonl,   jsonlPath);
  fs.renameSync(tmpIdx,     idxPath);
  fs.renameSync(tmpManif,   manifestP);
  fs.renameSync(tmpCubeIdx, cubeIdxP);

  console.log(`\tTrain draft sidecars: ${entries.length} drafts, ${(offset / 1e9).toFixed(2)} GB JSONL, `
    + `${Object.keys(cubeDrafts).length} cubes indexed.`);
}

// Canonical bootstrap (preferred): rebuild data/test/draft_sessions.json AND
// data/test/decks_by_cube_uuid.json with real cube_uuid + owner + draft_id
// by re-running pickPass1 + sampler against raw_data, then replaying just the
// val-ordering pass (no disk writes for picks/decks). Strictly slower than
// the heuristic --drafts-only, strictly more accurate. Use this after the
// initial dataset has been processed by an older version of the pipeline.
function runRebuildDrafts() {
  console.log('Rebuild-drafts: canonical pass1 + sampler + val-ordering...');
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  const numOracles = indexToOracle.length;

  console.log('\nPick pass 1 (raw → draft summaries + card→drafts index)...');
  const { pickCount, draftSummaries, cardToDrafts } = pickPass1(numOracles);

  console.log('\nSampling val drafts (deterministic; must match the original RANDOM_SEED)...');
  const { valDraftKeys } = sampleValDrafts(draftSummaries, cardToDrafts, pickCount);
  // Free the big pass1 structures — neither subsequent step needs them.
  draftSummaries.length = 0;
  for (let c = 0; c < cardToDrafts.length; c++) cardToDrafts[c] = undefined;
  if (global.gc) global.gc();

  console.log('\nPick-ordering pass (raw → draft_sessions for both splits)...');
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`).sort();
  // Same session model as pickPass2: sessionKey carries start-of-drafter idx,
  // open-session map tracks pool-growth boundaries per draftKey.
  const valDraftSessions = new Map();
  const trainDraftSessions = new Map();
  const valOpenSessions = new Map();
  const trainOpenSessions = new Map();
  let numTestPicks = 0;
  let numTrainPicks = 0;
  const openSessionForPick = (draftKey, cube, owner, poolSize, globalIdx,
                              sessionsMap, openMap) => {
    let open = openMap.get(draftKey);
    const isNewSession = open === undefined || (poolSize === 0 && open.lastPoolSize > 0);
    if (isNewSession) {
      const sessionKey = `${draftKey}::start=${globalIdx}`;
      const sess = { cube_uuid: cube, owner: owner, picks: [] };
      sessionsMap.set(sessionKey, sess);
      open = { sessionKey, lastPoolSize: poolSize };
      openMap.set(draftKey, open);
      return sess;
    }
    open.lastPoolSize = poolSize;
    return sessionsMap.get(open.sessionKey);
  };
  for (let i = 0; i < pickFiles.length; i++) {
    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    for (let p = 0; p < rawPicks.length; p++) {
      const pick = rawPicks[p];
      const picked = validatePick(pick);
      if (picked === null) continue;
      const owner = pick.owner || '_anon';
      const cube = pick.cube || '_nocube';
      const draftKey = `${i}::${owner}::${cube}::${pick.cubeCards}`;
      const isVal = valDraftKeys.has(draftKey);
      // Match pickPass2's cleanPool filter so the boundary detector sees the
      // same pool size at each step (drops 0/-1 sentinels and the picked card).
      let poolSize = 0;
      for (let q = 0; q < pick.pool.length; q++) {
        const v = pick.pool[q];
        if (v && v !== -1 && v !== picked) poolSize++;
      }
      if (isVal) {
        const sess = openSessionForPick(draftKey, cube, owner, poolSize, numTestPicks,
                                        valDraftSessions, valOpenSessions);
        sess.picks.push(numTestPicks);
        numTestPicks++;
      } else {
        const sess = openSessionForPick(draftKey, cube, owner, poolSize, numTrainPicks,
                                        trainDraftSessions, trainOpenSessions);
        sess.picks.push(numTrainPicks);
        numTrainPicks++;
      }
    }
    if ((i + 1) % 100 === 0 || i + 1 === pickFiles.length) {
      console.log(`\t${i + 1}/${pickFiles.length} files, val=${numTestPicks}p/${valDraftSessions.size}d, `
        + `train=${numTrainPicks}p/${trainDraftSessions.size}d`);
    }
  }
  // Capture the set of val cube_uuids *before* writeTrainDraftSidecars clears
  // the trainDraftSessions map; valDraftSessions remains intact.
  const valCubeIds = new Set();
  for (const sess of valDraftSessions.values()) valCubeIds.add(sess.cube_uuid);

  writeDraftSessions(valDraftSessions);
  writeTrainDraftSidecars(trainDraftSessions);

  console.log('\nCube uuids sidecar (positional with val_data.cubes)...');
  const rawCubesForSidecar = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'))
    .filter((cube) => cube.cards.length > 0);
  writeCubeUuidsSidecar(rawCubesForSidecar, valCubeIds);

  console.log('\nVal deck-matching pass (raw → decks_by_cube_uuid.json)...');
  writeDecksByCubeUuid(valDraftKeys);
}

// Replays processDecks's content-match routing but ONLY for val side, and
// only tracks (val_deck_global_idx → cube_uuid). Used by --rebuild-drafts;
// the canonical processDecks emits the same sidecar inline via the bridge
// argument below.
function writeDecksByCubeUuid(valDraftKeys) {
  const deckFiles = fs.readdirSync(`${sourceDir}/decks`).sort();
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`).sort();
  if (deckFiles.length !== pickFiles.length) {
    throw new Error(`deck/pick file count mismatch: ${deckFiles.length} vs ${pickFiles.length}`);
  }
  const MIN_MAINBOARD = 20;
  const decksByCube = {};
  const trainDecksByCube = {};
  let numTestDecks = 0;
  let numTrainDecks = 0;

  for (let i = 0; i < deckFiles.length; i++) {
    // Per-file (owner, cube) → Map<cubeCards, pickedSet>, same as processDecks.
    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    const draftGroups = new Map();
    for (const pk of rawPicks) {
      const picked = validatePick(pk);
      if (picked === null) continue;
      const owner = pk.owner || '_anon';
      const cube = pk.cube || '_nocube';
      const oc = `${owner}::${cube}`;
      let inner = draftGroups.get(oc);
      if (inner === undefined) { inner = new Map(); draftGroups.set(oc, inner); }
      let pickedSet = inner.get(pk.cubeCards);
      if (pickedSet === undefined) { pickedSet = new Set(); inner.set(pk.cubeCards, pickedSet); }
      pickedSet.add(picked);
    }

    const rawDecks = JSON.parse(fs.readFileSync(`${sourceDir}/decks/${deckFiles[i]}`, 'utf8'));
    for (const deck of rawDecks) {
      const basics = deck.basics ? new Set(deck.basics) : null;
      const keep = (card) => card !== -1 && !(basics && basics.has(card));
      const mainboard = deck.mainboard.filter(keep);
      if (mainboard.length < MIN_MAINBOARD) continue;

      const owner = deck.owner || '_anon';
      const cube = deck.cube || '_nocube';
      const inner = draftGroups.get(`${owner}::${cube}`);

      let isVal = false;
      if (inner !== undefined) {
        const deckCards = new Set();
        for (const c of mainboard) deckCards.add(c);
        for (const c of deck.sideboard.filter(keep)) deckCards.add(c);
        let bestCc = -1, bestContained = -1, bestSize = Infinity;
        for (const [cc, pickedSet] of inner) {
          let contained = 0;
          for (const p of pickedSet) if (deckCards.has(p)) contained++;
          if (contained > bestContained
              || (contained === bestContained && pickedSet.size < bestSize)) {
            bestContained = contained;
            bestSize = pickedSet.size;
            bestCc = cc;
          }
        }
        if (bestCc !== -1 && bestContained > 0) {
          const draftKey = `${i}::${owner}::${cube}::${bestCc}`;
          isVal = valDraftKeys.has(draftKey);
        }
      }

      if (isVal) {
        if (!decksByCube[cube]) decksByCube[cube] = [];
        decksByCube[cube].push(numTestDecks);
        numTestDecks++;
      } else {
        if (!trainDecksByCube[cube]) trainDecksByCube[cube] = [];
        trainDecksByCube[cube].push(numTrainDecks);
        numTrainDecks++;
      }
    }
    if ((i + 1) % 100 === 0 || i + 1 === deckFiles.length) {
      console.log(`\t${i + 1}/${deckFiles.length} deck files, `
        + `val=${numTestDecks}/${Object.keys(decksByCube).length}c, `
        + `train=${numTrainDecks}/${Object.keys(trainDecksByCube).length}c`);
    }
  }

  fs.writeFileSync(`${testDir}/decks_by_cube_uuid.json`, JSON.stringify(decksByCube));
  fs.writeFileSync(`${trainDir}/decks_by_cube_uuid.json`, JSON.stringify(trainDecksByCube));
  console.log(`\tDecks-by-cube: ${Object.keys(decksByCube).length} cubes, ${numTestDecks} decks total.`);
}

// Bootstrap entry point: rebuild ONLY data/test/draft_sessions.json from the
// existing data/test/picks/ files via the pool-growth heuristic. Used when
// adopting Phase J against an already-processed dataset (full reprocess
// retains real owner UUIDs; this bootstrap doesn't).
function runDraftsOnly() {
  console.log('Drafts-only bootstrap: rebuilding draft_sessions.json from data/test/picks/*...');
  const pickFiles = fs.readdirSync(`${testDir}/picks`).sort();
  const sessions = {};
  let globalIdx = 0;
  // cube_cards_idx → current open draft for that cube context. A new draft
  // begins when pool size is 0 (no picks recorded yet for this drafter); all
  // subsequent picks with the same cube_cards_idx whose pool size > 0 are
  // appended to that draft until the next pool=0 marker.
  let openByCC = new Map();

  // A "real" cube draft typically yields ~42 valid picks. Anything below
  // SUSPECT_MIN is almost certainly a sampling artifact (e.g., back-to-back
  // pool=0 records that pool-growth detection treats as separate drafts).
  const SUSPECT_MIN = 10;

  const closeDraft = (cur) => {
    sessions[cur.draft_id] = {
      cube_uuid: null,
      owner: null,
      picks: cur.picks,
      synthetic: true,
      suspect: cur.picks.length < SUSPECT_MIN,
    };
  };

  for (const fname of pickFiles) {
    // cube_cards_idx is per-batch — flush all open drafts at every batch
    // boundary so we don't merge unrelated drafts that happen to share a cci
    // across batches. This may split the rare draft that straddles a 10k-pick
    // boundary; those land as two consecutive sessions, marked synthetic.
    for (const cur of openByCC.values()) closeDraft(cur);
    openByCC = new Map();

    const batch = JSON.parse(fs.readFileSync(`${testDir}/picks/${fname}`, 'utf8'));
    for (const rec of batch) {
      const cci = rec.cube_cards_idx;
      const poolSize = (rec.pool || []).length;
      let cur = openByCC.get(cci);
      // New draft only when we cross from a non-empty pool back to empty —
      // sequential pool=0 picks within a draft (validation dropped the first
      // pick of pack 1) should stay together.
      const isNewDraft = cur === undefined || (poolSize === 0 && cur.lastPoolSize > 0);
      if (isNewDraft) {
        if (cur !== undefined) closeDraft(cur);
        const seed = `${fname}::cci=${cci}::start=${globalIdx}`;
        cur = { draft_id: shortHash(seed), lastPoolSize: poolSize, picks: [globalIdx] };
        openByCC.set(cci, cur);
      } else {
        cur.picks.push(globalIdx);
        cur.lastPoolSize = poolSize;
      }
      globalIdx++;
    }
  }
  for (const cur of openByCC.values()) closeDraft(cur);

  fs.writeFileSync(`${testDir}/draft_sessions.json`, JSON.stringify({ sessions }));
  console.log(`\tDraft sessions: ${Object.keys(sessions).length} drafts (${globalIdx} picks) `
    + `written to ${testDir}/draft_sessions.json.`);
}

// =============================================================================
// Cubes (asymmetric: train=all, test=val-referenced subset)
// =============================================================================

/**
 * Process cubes from raw_data/cubes.json. train/cubes/ gets ALL cubes (so the
 * recsys task has full data); test/cubes/ gets only cubes whose UUID appears in
 * valCubeIds (the cubes referenced by val draft picks). oracleFrequency is
 * computed across ALL cubes since that matches train's view.
 *
 * Returns { cubeMap, numTrainCubes, numTestCubes }. cubeMap covers all cubes
 * (used by deck processing for cube_cards embedding regardless of split).
 */
const processCubes = (numOracles, valCubeIds, validCubeIds) => {
  console.log('\tLoading cubes...');
  // Apply the same size filter used in pickPass1/pickPass2 so cubeMap and the
  // valid set agree. We filter by membership in validCubeIds — which was
  // already built via loadValidCubeIds() using identical [MIN, MAX] bounds.
  const rawCubes = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'))
    .filter((cube) => validCubeIds.has(cube.id));

  const cubeMap = {};
  const allCubeCards = [];
  const testCubeCards = [];
  // Phase S sidecar: cube_uuid in the same positional order as testCubeCards
  // (i.e., matches val_data.cubes index space). Lets the backend reverse-look
  // up cube_uuid from a val cube_idx for clickable CardDrawer rows.
  const testCubeUuids = [];
  const oracleFrequency = new Array(numOracles).fill(0);

  for (const cube of rawCubes) {
    cubeMap[cube.id] = cube.cards;
    allCubeCards.push(cube.cards);
    for (let j = 0; j < cube.cards.length; j++) oracleFrequency[cube.cards[j]]++;
    if (valCubeIds.has(cube.id)) {
      testCubeCards.push(cube.cards);
      testCubeUuids.push(cube.id);
    }
  }

  console.log(`\tLoaded ${rawCubes.length} cubes (${testCubeCards.length} referenced by val drafts).`);

  resetDir(`${trainDir}/cubes`);
  resetDir(`${testDir}/cubes`);
  const writeBatched = (dir, arr) => {
    for (let i = 0; i < arr.length / WRITE_BATCH_SIZE; i++) {
      writeFile(`${dir}/cubes/${padLeft(i, 4)}.json`,
        arr.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
    }
  };
  writeBatched(trainDir, allCubeCards);
  writeBatched(testDir, testCubeCards);

  fs.writeFileSync(`${trainDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  fs.writeFileSync(`${testDir}/oracleFrequency.json`, JSON.stringify(oracleFrequency));
  fs.writeFileSync(`${testDir}/cube_uuids.json`, JSON.stringify(testCubeUuids));
  console.log(`\tCube uuids sidecar: ${testCubeUuids.length} entries -> ${testDir}/cube_uuids.json.`);

  writeCubeDirectory(rawCubes, valCubeIds);

  console.log('\tDone processing cubes.');
  return { cubeMap, numTrainCubes: rawCubes.length, numTestCubes: testCubeCards.length };
};

// Phase S cube_uuids.json sidecar: positional cube_uuid list aligned with
// val_data.cubes (test/cubes/* iteration order). Derived from rawCubes in
// the same iteration order processCubes uses, so any caller that has the raw
// cubes list and a valCubeIds Set can emit/refresh this file consistently.
function writeCubeUuidsSidecar(rawCubes, valCubeIds) {
  const uuids = [];
  for (const c of rawCubes) {
    if (valCubeIds.has(c.id)) uuids.push(c.id);
  }
  fs.writeFileSync(`${testDir}/cube_uuids.json`, JSON.stringify(uuids));
  console.log(`\tCube uuids sidecar: ${uuids.length} entries -> ${testDir}/cube_uuids.json.`);
}

// Cube directory sidecar (Phase I): UUID + name + owner + image, plus the
// positional index back into train/cubes/ so the dashboard can lazily resolve
// the card list for any cube. Strictly additive — does not change any other
// output file. Extracted as a helper so the `--cubes-only` bootstrap can
// rebuild just this sidecar without re-running the full pipeline.
function writeCubeDirectory(rawCubes, valCubeIds) {
  const directory = new Array(rawCubes.length);
  for (let i = 0; i < rawCubes.length; i++) {
    const c = rawCubes[i];
    directory[i] = {
      cube_uuid: c.id,
      name: c.name || '',
      owner: c.owner || '',
      owner_id: c.owner_id || '',
      image_uri: c.image_uri || '',
      card_count: c.cards.length,
      has_val_drafts: valCubeIds.has(c.id),
      train_idx: i,
    };
  }
  fs.writeFileSync(`${dataDir}/cube_directory.json`, JSON.stringify(directory));
  console.log(`\tCube directory: ${directory.length} entries written to ${dataDir}/cube_directory.json.`);
}

// Bootstrap entry point: rebuild ONLY the cube directory sidecar from
// raw_data/cubes.json + the existing data/test/cubes/* (used to identify which
// cubes were sampled into val). Skips the heavy pick/deck passes — runs in
// seconds rather than hours. Used when adopting Phase I against an existing
// processed dataset.
function runCubesOnly() {
  console.log('Cubes-only bootstrap: rebuilding cube_directory.json...');
  const rawCubes = JSON.parse(fs.readFileSync(`${sourceDir}/cubes.json`, 'utf8'))
    .filter((cube) => cube.cards.length > 0);
  console.log(`\tLoaded ${rawCubes.length} non-empty cubes from raw_data.`);

  // Build fingerprints of val cubes from data/test/cubes/* (each file is a
  // batched array of card-index lists). Match raw cubes against these by
  // sorted-card-list content — exact since processCubes writes cube.cards
  // unchanged. Tag any matching raw cube as has_val_drafts.
  const valFingerprints = new Set();
  const testCubesDir = `${testDir}/cubes`;
  if (fs.existsSync(testCubesDir)) {
    for (const fname of fs.readdirSync(testCubesDir).sort()) {
      const batch = JSON.parse(fs.readFileSync(`${testCubesDir}/${fname}`, 'utf8'));
      for (const cards of batch) {
        valFingerprints.add(JSON.stringify(cards.slice().sort((a, b) => a - b)));
      }
    }
  }
  console.log(`\tBuilt ${valFingerprints.size} val cube fingerprints.`);

  const valCubeIds = new Set();
  for (const c of rawCubes) {
    const fp = JSON.stringify(c.cards.slice().sort((a, b) => a - b));
    if (valFingerprints.has(fp)) valCubeIds.add(c.id);
  }
  console.log(`\tMatched ${valCubeIds.size} raw cubes as val.`);

  writeCubeDirectory(rawCubes, valCubeIds);
  writeCubeUuidsSidecar(rawCubes, valCubeIds);
}

// =============================================================================
// Decks (per-draft split: a deck goes to val if its (owner, cube) is a val draft)
// =============================================================================

/**
 * Process decks from raw_data/decks/. Routes each deck to train/test by
 * CONTENT MATCHING against draft pick sequences in the SAME file.
 *
 * Each deck has owner, cube, mainboard, sideboard. Each draft (in picks/{i}.json)
 * is identified by (owner, cube, cubeCards). A deck and its source draft satisfy:
 *
 *   draft.picked_set ⊆ (deck.mainboard ∪ deck.sideboard)
 *
 * (deck = drafted pool + basics; basics are filtered out of mainboard/sideboard
 * during preprocessing.) For each (owner, cube) pair, multiple drafts may exist
 * — distinguishable by cubeCards. We select the draft whose pickedSet has the
 * largest overlap with the deck (typically full containment) as the deck's
 * source draft. If a deck shares (owner, cube) with NO drafts in the file, it
 * defaults to train and we log it as unmatched.
 *
 * This gives val_decks ≈ val_drafts (with multiplier ~1-2x for users who saved
 * multiple deck variants from the same draft).
 *
 * valDraftSet is the set of `${fileIdx}::${owner}::${cube}::${cubeCards}` keys
 * that the sampler selected — same format produced by pickPass1.
 *
 * Correlations matrix is computed from TRAIN decks only.
 */
const processDecks = (oracleCount, cubeMap, valDraftSet, validCubeIds) => {
  console.log('\tLoading decks (content-match deck → draft routing)...');
  const correlations = new Int32Array(oracleCount * oracleCount).fill(0);

  const deckFiles = fs.readdirSync(`${sourceDir}/decks`).sort();
  const pickFiles = fs.readdirSync(`${sourceDir}/picks`).sort();
  if (deckFiles.length !== pickFiles.length) {
    throw new Error(`deck/pick file count mismatch: ${deckFiles.length} vs ${pickFiles.length}`);
  }
  console.log(`\tLoaded ${deckFiles.length} deck files.`);

  const trainDecks = [];
  const testDecks = [];

  // Minimum mainboard size to accept a deck. Smaller than this and it's
  // garbage (token decks, art-card-only entries, etc.) that pollutes the
  // adjacency matrix.
  const MIN_MAINBOARD = 20;

  let unmatchedDecks = 0;
  let multiCandidateDecks = 0;
  // Phase J/M: index val deck global indices by cube UUID so the dashboard
  // can show "decks from this cube". Strictly additive sidecar.
  const decksByCube = {};
  // Phase O: same index for train decks.
  const trainDecksByCube = {};

  for (let i = 0; i < deckFiles.length; i++) {
    // Build draftGroups for this file: (owner, cube) → Map<cubeCards, Set(picked)>.
    // We use only valid picks (validatePick) so the sets match Pass-1 keys.
    const rawPicks = JSON.parse(fs.readFileSync(`${sourceDir}/picks/${pickFiles[i]}`, 'utf8'));
    const draftGroups = new Map();
    for (const pk of rawPicks) {
      const picked = validatePick(pk);
      if (picked === null) continue;
      const owner = pk.owner || '_anon';
      const cube = pk.cube || '_nocube';
      const oc = `${owner}::${cube}`;
      let inner = draftGroups.get(oc);
      if (inner === undefined) { inner = new Map(); draftGroups.set(oc, inner); }
      let pickedSet = inner.get(pk.cubeCards);
      if (pickedSet === undefined) { pickedSet = new Set(); inner.set(pk.cubeCards, pickedSet); }
      pickedSet.add(picked);
    }

    const rawDecks = JSON.parse(fs.readFileSync(`${sourceDir}/decks/${deckFiles[i]}`, 'utf8'));
    for (const deck of rawDecks) {
      // Cube filter: drop decks tied to invalid cubes. cube_cards would be
      // empty for these anyway (cubeMap was filtered), but dropping here
      // also avoids spurious correlation-matrix updates from garbage decks.
      const deckCube = deck.cube || '_nocube';
      if (!validCubeIds.has(deckCube)) continue;
      const basics = deck.basics ? new Set(deck.basics) : null;
      const keep = (card) => card !== -1 && !(basics && basics.has(card));
      const processed = {
        mainboard: deck.mainboard.filter(keep),
        sideboard: deck.sideboard.filter(keep),
        cube_cards: cubeMap[deck.cube] || [],
      };
      if (processed.mainboard.length < MIN_MAINBOARD) continue;

      // Content-match: which draft in this (owner, cube) group produced this deck?
      const owner = deck.owner || '_anon';
      const cube = deck.cube || '_nocube';
      const oc = `${owner}::${cube}`;
      const inner = draftGroups.get(oc);

      let isVal = false;
      if (inner === undefined) {
        // No drafts for this (owner, cube) — likely a non-draft deck.
        unmatchedDecks++;
      } else {
        // Build deck card set for subset checks.
        const deckCards = new Set();
        for (let k = 0; k < processed.mainboard.length; k++) deckCards.add(processed.mainboard[k]);
        for (let k = 0; k < processed.sideboard.length; k++) deckCards.add(processed.sideboard[k]);

        // Find the draft with most picked cards contained in the deck. Tie-break
        // by smaller pickedSet (more specific match).
        let bestCc = -1;
        let bestContained = -1;
        let bestSize = Infinity;
        let candidateCount = 0;
        for (const [cc, pickedSet] of inner) {
          let contained = 0;
          for (const p of pickedSet) if (deckCards.has(p)) contained++;
          if (contained > bestContained
              || (contained === bestContained && pickedSet.size < bestSize)) {
            bestContained = contained;
            bestSize = pickedSet.size;
            bestCc = cc;
          }
          if (contained === pickedSet.size && contained > 0) candidateCount++;
        }
        if (candidateCount > 1) multiCandidateDecks++;
        if (bestCc !== -1 && bestContained > 0) {
          const draftKey = `${i}::${owner}::${cube}::${bestCc}`;
          isVal = valDraftSet.has(draftKey);
        } else {
          unmatchedDecks++;
        }
      }

      if (isVal) {
        if (!decksByCube[cube]) decksByCube[cube] = [];
        decksByCube[cube].push(testDecks.length);
        testDecks.push(processed);
      } else {
        if (!trainDecksByCube[cube]) trainDecksByCube[cube] = [];
        trainDecksByCube[cube].push(trainDecks.length);
        trainDecks.push(processed);
        // Co-occurrences from TRAIN decks only.
        const mb = processed.mainboard;
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
    }
    if ((i + 1) % 100 === 0 || i + 1 === deckFiles.length) {
      console.log(`\t\t${i + 1}/${deckFiles.length} files | val=${testDecks.length} train=${trainDecks.length} unmatched=${unmatchedDecks}`);
    }
  }

  resetDir(`${trainDir}/decks`);
  resetDir(`${testDir}/decks`);
  const writeBatched = (dir, arr) => {
    for (let i = 0; i < arr.length / WRITE_BATCH_SIZE; i++) {
      writeFile(`${dir}/decks/${padLeft(i, 4)}.json`,
        arr.slice(i * WRITE_BATCH_SIZE, (i + 1) * WRITE_BATCH_SIZE));
    }
  };
  writeBatched(trainDir, trainDecks);
  writeBatched(testDir, testDecks);

  writeLargeArray(`${trainDir}/correlations.json`, correlations);

  fs.writeFileSync(`${testDir}/decks_by_cube_uuid.json`, JSON.stringify(decksByCube));
  fs.writeFileSync(`${trainDir}/decks_by_cube_uuid.json`, JSON.stringify(trainDecksByCube));

  console.log(`\tDone processing ${trainDecks.length} train + ${testDecks.length} test decks.`);
  console.log(`\t\tUnmatched (defaulted to train): ${unmatchedDecks}.`);
  console.log(`\t\tDecks with multiple full-containment candidates: ${multiCandidateDecks}.`);
  console.log(`\t\tdecks_by_cube_uuid: ${Object.keys(decksByCube).length} val cubes / `
    + `${Object.keys(trainDecksByCube).length} train cubes indexed.`);
  return {
    numTrainDecks: trainDecks.length,
    numTestDecks: testDecks.length,
    unmatchedDecks,
    multiCandidateDecks,
  };
};

// =============================================================================
// Oracle dict, elos (no land mask anymore — land penalty was removed)
// =============================================================================

const processOracleDict = () => {
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  fs.writeFileSync(`${trainDir}/oracleDict.json`, JSON.stringify(indexToOracle));
  fs.writeFileSync(`${testDir}/oracleDict.json`, JSON.stringify(indexToOracle));

  const simpleCardDict = JSON.parse(fs.readFileSync(`${sourceDir}/simpleCardDict.json`, 'utf8'));

  const elos = [];
  for (let i = 0; i < indexToOracle.length; i++) {
    const card = simpleCardDict[indexToOracle[i]] || {};
    const elo = card.elo || 1200;
    elos.push(Math.log(elo / 600));
  }
  let maxElo = -Infinity;
  for (let i = 0; i < elos.length; i++) {
    if (elos[i] > maxElo) maxElo = elos[i];
  }
  for (let i = 0; i < elos.length; i++) elos[i] = elos[i] / maxElo;

  fs.writeFileSync(`${testDir}/elos.json`, JSON.stringify(elos));
  fs.writeFileSync(`${trainDir}/elos.json`, JSON.stringify(elos));

  writeCardDictEnriched(indexToOracle, simpleCardDict);

  return indexToOracle.length;
};

// Phase L: enrich card metadata with set / colors / color_identity / rarity
// from the local CubeCobra carddict.json (env CUBECOBRA_CARD_DICT, or the
// default checkout path). Strictly additive — emits a sidecar
// data/train/cardDictEnriched.json keyed by oracle_id with the new fields
// added; simpleCardDict.json is untouched. Silently skipped if the source
// dict isn't available.
function writeCardDictEnriched(indexToOracle, simpleCardDict) {
  // Optional card-enrichment source. Point CUBECOBRA_CARD_DICT at a CubeCobra
  // checkout's private/carddict.json; defaults to a sibling-repo layout and is
  // silently skipped when absent.
  const dictPath = process.env.CUBECOBRA_CARD_DICT
    || path.join(__dirname, '..', '..', 'CubeCobra', 'packages', 'server', 'private', 'carddict.json');
  if (!fs.existsSync(dictPath)) {
    console.log(`\tSkipping card enrichment — ${dictPath} not found.`);
    return;
  }
  console.log(`\tLoading CubeCobra carddict for enrichment (${dictPath})...`);
  const carddict = JSON.parse(fs.readFileSync(dictPath, 'utf8'));

  // Build oracle_id → first matching entry. carddict is keyed by scryfall
  // print id; each entry has oracle_id. Multiple prints share an oracle_id —
  // any one's set/colors/rarity is fine for the dashboard (colors/identity
  // are oracle-stable; we accept whatever print this dict resolved first).
  const byOracle = new Map();
  for (const v of Object.values(carddict)) {
    const oid = v && v.oracle_id;
    if (!oid || byOracle.has(oid)) continue;
    byOracle.set(oid, v);
  }

  const enriched = {};
  let hits = 0;
  for (let i = 0; i < indexToOracle.length; i++) {
    const oid = indexToOracle[i];
    const base = simpleCardDict[oid] || {};
    const extra = byOracle.get(oid);
    if (extra) hits++;
    enriched[oid] = {
      name: base.name || '',
      image: base.image || '',
      type: base.type || '',
      cmc: base.cmc || 0,
      elo: base.elo || 1200,
      set: extra ? (extra.set || '') : '',
      set_name: extra ? (extra.set_name || '') : '',
      colors: extra ? (extra.colors || []) : [],
      color_identity: extra ? (extra.color_identity || []) : [],
      rarity: extra ? (extra.rarity || '') : '',
    };
  }
  fs.writeFileSync(`${trainDir}/cardDictEnriched.json`, JSON.stringify(enriched));
  console.log(`\tCard enrichment: ${hits}/${indexToOracle.length} oracle_ids matched, sidecar written to ${trainDir}/cardDictEnriched.json.`);
}

// Bootstrap entry point: rebuild ONLY data/train/cardDictEnriched.json.
function runEnrichCardsOnly() {
  console.log('Enrich-cards bootstrap: rebuilding cardDictEnriched.json...');
  const indexToOracle = Object.values(JSON.parse(fs.readFileSync(`${sourceDir}/indexToOracleMap.json`, 'utf8')));
  const simpleCardDict = JSON.parse(fs.readFileSync(`${sourceDir}/simpleCardDict.json`, 'utf8'));
  writeCardDictEnriched(indexToOracle, simpleCardDict);
}

// =============================================================================
// bomb_card_indices.json (top N cards by pickCount; used by val_bomb_agreement)
// =============================================================================

const writeBombCardIndices = (pickCount) => {
  const indexed = pickCount
    .map((c, i) => [c, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, BOMB_COUNT)
    .map(([_, i]) => i);
  fs.writeFileSync(`${trainDir}/bomb_card_indices.json`, JSON.stringify(indexed));
  fs.writeFileSync(`${testDir}/bomb_card_indices.json`, JSON.stringify(indexed));
  console.log(`\tWrote ${BOMB_COUNT} bomb card indices.`);
};

// =============================================================================
// Main
// =============================================================================

const run = () => {
  if (process.argv.includes('--cubes-only')) {
    runCubesOnly();
    process.exit(0);
  }
  if (process.argv.includes('--drafts-only')) {
    runDraftsOnly();
    process.exit(0);
  }
  if (process.argv.includes('--rebuild-drafts')) {
    runRebuildDrafts();
    process.exit(0);
  }
  if (process.argv.includes('--enrich-cards')) {
    runEnrichCardsOnly();
    process.exit(0);
  }

  ensureDir(trainDir);
  ensureDir(testDir);

  const metadata = { random_seed: RANDOM_SEED };

  console.log('Processing oracle dict / elos...');
  metadata.numOracles = processOracleDict();
  console.log(`\tWe have ${metadata.numOracles} oracles.`);

  // ---- Cube-size filter (computed once, used by every downstream stage)
  console.log(`\nFiltering cubes to size range [${CUBE_SIZE_MIN}, ${CUBE_SIZE_MAX}]...`);
  const { validCubeIds, stats: cubeFilterStats } = loadValidCubeIds();
  console.log(`\tCubes kept: ${cubeFilterStats.kept}/${cubeFilterStats.total} ` +
    `(${cubeFilterStats.droppedSmall} too small, ${cubeFilterStats.droppedLarge} too large).`);
  metadata.cubeFilter = {
    min: CUBE_SIZE_MIN, max: CUBE_SIZE_MAX,
    total: cubeFilterStats.total, kept: cubeFilterStats.kept,
    droppedSmall: cubeFilterStats.droppedSmall, droppedLarge: cubeFilterStats.droppedLarge,
  };

  // ---- Pick pass 1: gather pickCount + draft summaries + inverted index
  console.log('\nPick pass 1...');
  const { pickCount, draftSummaries, cardToDrafts } = pickPass1(metadata.numOracles, validCubeIds);

  // pickCount is written here (train) and after pass 2 we'll write to test too.
  // The recsys task and bomb metric use the TRAIN pickCount for both splits.
  fs.writeFileSync(`${trainDir}/pickCount.json`, JSON.stringify(pickCount));
  fs.writeFileSync(`${testDir}/pickCount.json`, JSON.stringify(pickCount));

  // Bomb indices for val_bomb_agreement metric.
  writeBombCardIndices(pickCount);

  // ---- Sampler: decide val drafts
  console.log('\nSampling val drafts...');
  let { valDraftKeys, manifest } = sampleValDrafts(draftSummaries, cardToDrafts, pickCount);
  // Free draft summaries + inverted index before pass 2 — pass 2 only needs
  // valDraftKeys. Footprint is ~1-2 GB combined and unnecessary downstream.
  draftSummaries.length = 0;
  for (let c = 0; c < cardToDrafts.length; c++) cardToDrafts[c] = undefined;
  if (global.gc) global.gc();

  // ---- Pick pass 2: route + write
  console.log('\nPick pass 2...');
  const { numTrainPicks, numTestPicks, valCubeIds } = pickPass2(metadata.numOracles, valDraftKeys, validCubeIds);
  metadata.numPicks = numTrainPicks;
  metadata.numTestPicks = numTestPicks;

  // ---- Cubes (asymmetric: train=all, test=val-referenced subset)
  console.log('\nProcessing cubes...');
  const { cubeMap, numTrainCubes, numTestCubes } = processCubes(metadata.numOracles, valCubeIds, validCubeIds);
  metadata.numCubes = numTrainCubes;
  metadata.numTestCubes = numTestCubes;

  // ---- Decks: content-match each deck to a specific draft (by file, owner,
  // cube, cubeCards), then route by whether that draft is val. Pass valDraftKeys
  // through unchanged — processDecks reconstructs the per-file (owner, cube, cc)
  // membership from those keys.
  console.log('\nProcessing decks...');
  const { numTrainDecks, numTestDecks, unmatchedDecks, multiCandidateDecks } =
    processDecks(metadata.numOracles, cubeMap, valDraftKeys, validCubeIds);
  metadata.numDecks = numTrainDecks;
  metadata.numTestDecks = numTestDecks;

  // ---- Final manifest
  manifest.val_picks_actual = numTestPicks;
  manifest.val_decks = numTestDecks;
  manifest.val_cubes = numTestCubes;
  manifest.unmatched_decks = unmatchedDecks;
  manifest.multi_candidate_decks = multiCandidateDecks;
  fs.writeFileSync(`${testDir}/val_manifest.json`, JSON.stringify(manifest, null, 2));

  fs.writeFileSync(`${trainDir}/metadata.json`, JSON.stringify(metadata));
  fs.writeFileSync(`${testDir}/metadata.json`, JSON.stringify(metadata));

  console.log('\nDone!');
  console.log(`  train: ${numTrainCubes} cubes, ${numTrainDecks} decks, ${numTrainPicks} picks`);
  console.log(`  test:  ${numTestCubes} cubes, ${numTestDecks} decks, ${numTestPicks} picks`);
  console.log(`  val_manifest.json written to ${testDir}/val_manifest.json`);
  process.exit(0);
};

run();
