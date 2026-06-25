// Validation-split sampler: random baseline + coverage top-up.
//
// Phase A draws a deterministic random sample until total picks reach
// `randomTargetPicks`, so per-card val pick rate matches the full dataset.
// Phase B then walks still-uncovered eligible cards (rarest first) and adds
// the draft that newly covers the most, guaranteeing every card with
// totalPicks >= `coverageThreshold` appears at least once. Hard-capped at
// `hardCapPicks`. Deterministic given `randomSeed`.

const { mulberry32 } = require('./random');

const sampleValDrafts = (draftSummaries, cardToDrafts, pickCount, config) => {
  const { coverageThreshold, randomTargetPicks, hardCapPicks, randomSeed, bombCount } = config;
  console.log('\tSampling val drafts (random baseline + coverage top-up)...');
  const numOracles = pickCount.length;

  // Eligible cards: those with totalPicks >= coverageThreshold.
  // needed[c] = 1 for eligible (presence is enough — distribution comes from
  // Phase A, presence comes from Phase B).
  const eligible = new Uint8Array(numOracles);
  let eligibleCards = 0;
  for (let c = 0; c < numOracles; c++) {
    if (pickCount[c] >= coverageThreshold) {
      eligible[c] = 1;
      eligibleCards++;
    }
  }
  console.log(`\t\t${eligibleCards} eligible cards (totalPicks >= ${coverageThreshold}).`);

  const covered = new Int32Array(numOracles); // covered[c] = val picks of card c so far
  const selected = new Uint8Array(draftSummaries.length);
  const valDraftKeys = new Set();
  let totalValPicks = 0;
  let uncoveredCount = eligibleCards;

  const acceptDraft = (id) => {
    if (selected[id]) return;
    selected[id] = 1;
    const d = draftSummaries[id];
    totalValPicks += d.nPicks;
    valDraftKeys.add(d.key);
    for (const [card, cnt] of d.cardCounts) {
      if (eligible[card] && covered[card] === 0) uncoveredCount--;
      covered[card] += cnt;
    }
  };

  // ---- Phase A: deterministic random sample to randomTargetPicks.
  console.log(`\t\tPhase A: random sample to ${randomTargetPicks} picks...`);
  const shuffledIds = new Int32Array(draftSummaries.length);
  for (let i = 0; i < draftSummaries.length; i++) shuffledIds[i] = i;
  // Fisher–Yates on the typed array.
  {
    const rand = mulberry32(randomSeed);
    for (let i = shuffledIds.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = shuffledIds[i]; shuffledIds[i] = shuffledIds[j]; shuffledIds[j] = tmp;
    }
  }
  let phaseACursor = 0;
  while (phaseACursor < shuffledIds.length && totalValPicks < randomTargetPicks) {
    acceptDraft(shuffledIds[phaseACursor++]);
  }
  const phaseACount = valDraftKeys.size;
  const phaseAPicks = totalValPicks;
  console.log(`\t\tPhase A done: ${phaseACount} drafts, ${phaseAPicks} picks. ` +
    `Uncovered eligible: ${uncoveredCount}/${eligibleCards}.`);

  // ---- Phase B: cover remaining cards, rarest first.
  console.log(`\t\tPhase B: coverage top-up for ${uncoveredCount} uncovered cards...`);
  const cardOrder = [];
  for (let c = 0; c < numOracles; c++) {
    if (eligible[c] && covered[c] === 0) cardOrder.push(c);
  }
  cardOrder.sort((a, b) => pickCount[a] - pickCount[b]);

  // marginalScore: number of still-uncovered eligible cards this draft would
  // newly cover. We don't bias by pick count of those cards — every uncovered
  // card is worth +1 (we just need presence).
  const marginalCoverage = (d) => {
    let s = 0;
    for (const card of d.cardCounts.keys()) {
      if (eligible[card] && covered[card] === 0) s++;
    }
    return s;
  };

  let uncoverableCards = 0;
  let cardsProcessed = 0;
  for (const c of cardOrder) {
    cardsProcessed++;
    if (covered[c] > 0) continue; // covered by an earlier Phase B selection
    if (totalValPicks >= hardCapPicks) {
      console.log(`\t\t\tHARD_CAP reached at card ${cardsProcessed}/${cardOrder.length}.`);
      break;
    }
    const candidates = cardToDrafts[c];
    if (candidates === undefined || candidates.length === 0) {
      uncoverableCards++;
      continue;
    }
    let bestId = -1;
    let bestMarginal = -1;
    let bestSize = Infinity;
    for (let k = 0; k < candidates.length; k++) {
      const id = candidates[k];
      if (selected[id]) continue;
      const d = draftSummaries[id];
      const m = marginalCoverage(d);
      if (m > bestMarginal || (m === bestMarginal && d.nPicks < bestSize)) {
        bestMarginal = m;
        bestSize = d.nPicks;
        bestId = id;
      }
    }
    if (bestId === -1) {
      uncoverableCards++; // every draft containing c already selected (rare)
      continue;
    }
    acceptDraft(bestId);
  }
  const phaseBCount = valDraftKeys.size - phaseACount;
  const phaseBPicks = totalValPicks - phaseAPicks;
  console.log(`\t\tPhase B done: ${phaseBCount} drafts, ${phaseBPicks} picks. ` +
    `Uncoverable cards: ${uncoverableCards}.`);

  // Coverage report.
  let cardsCovered = 0;
  let cardsUncovered = 0;
  const uncoveredCards = [];
  for (let c = 0; c < numOracles; c++) {
    if (!eligible[c]) continue;
    if (covered[c] > 0) cardsCovered++;
    else {
      cardsUncovered++;
      if (uncoveredCards.length < 50) {
        uncoveredCards.push({ card: c, totalPicks: pickCount[c] });
      }
    }
  }

  const manifest = {
    config: {
      RANDOM_TARGET_PICKS: randomTargetPicks,
      HARD_CAP_PICKS: hardCapPicks,
      COVERAGE_THRESHOLD: coverageThreshold,
      RANDOM_SEED: randomSeed,
      BOMB_COUNT: bombCount,
    },
    val_drafts: valDraftKeys.size,
    val_picks_estimated: totalValPicks,
    phaseA_drafts: phaseACount,
    phaseA_picks: phaseAPicks,
    phaseB_drafts: phaseBCount,
    phaseB_picks: phaseBPicks,
    eligible_cards_for_coverage: eligibleCards,
    cards_covered: cardsCovered,
    cards_uncovered: cardsUncovered,
    uncoverable_cards: uncoverableCards,
    uncovered_card_sample: uncoveredCards,
  };
  console.log(`\tSampler done: ${valDraftKeys.size} drafts (${totalValPicks} picks).`);
  console.log(`\t\tCoverage: ${cardsCovered}/${eligibleCards} eligible covered, ${cardsUncovered} uncovered.`);
  return { valDraftKeys, manifest };
};

module.exports = { sampleValDrafts };
