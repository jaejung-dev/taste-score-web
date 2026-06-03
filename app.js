const DATA_URL = "data.json?v=20260603-ood-1";

const fmt = (value, digits = 5) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "pending"
    : Number(value).toFixed(digits);

const byId = (id) => document.getElementById(id);

const pct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "pending"
    : `${(Number(value) * 100).toFixed(digits)}%`;

function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function renderSummary(data) {
  const el = byId("summary");
  el.innerHTML = "";
  const cards = [
    ["Test rows", data.summary.test_rows ?? data.summary.prompts],
    ["Test pairs", data.summary.test_unique_pairs ?? data.summary.pairs],
    ["Selected prompts", data.summary.selected_prompts ?? data.summary.prompts],
    ["Selected images", data.summary.selected_candidates ?? data.summary.candidates],
  ];
  if (data.summary.ood_prompts) {
    cards.push(["OOD probes", data.summary.ood_prompts]);
  }
  cards.forEach(([label, value]) => {
    const card = createEl("div", "summary-card");
    card.append(createEl("span", "summary-value", value));
    card.append(createEl("span", "summary-label", label));
    el.append(card);
  });
}

function renderEvaluation(data) {
  const el = byId("evaluation");
  if (!el || !data.evaluation?.overview) return;
  const overview = data.evaluation.overview;
  el.innerHTML = "";
  el.append(createEl("h2", "", "Human Ranking Alignment"));
  el.append(
    createEl(
      "p",
      "score-note",
      "Higher Spearman means TASTE ranks candidate images more similarly to human judgments.",
    ),
  );

  const cards = createEl("div", "metric-grid");
  [
    ["Human-rank Spearman", fmt(overview.image_rank_spearman), "model image score vs human ranking across all test images", "primary"],
    ["Pairwise accuracy", pct(overview.pairwise_accuracy), `${overview.n_pairs} pair comparisons vs human majority`, ""],
  ].forEach(([label, value, detail, kind]) => {
    const card = createEl("div", `metric-card ${kind}`.trim());
    card.append(createEl("span", "metric-value", value));
    card.append(createEl("span", "metric-label", label));
    card.append(createEl("span", "metric-detail", detail));
    cards.append(card);
  });
  el.append(cards);

  const tableWrap = createEl("div", "table-wrap compact-table");
  const table = createEl("table", "score-table eval-table");
  const thead = document.createElement("thead");
  const head = document.createElement("tr");
  ["Dimension", "Images", "Human-rank Spearman", "Pairwise Acc.", "Pairs"].forEach((label) =>
    head.append(createEl("th", "", label)),
  );
  thead.append(head);
  const tbody = document.createElement("tbody");
  data.evaluation.by_dimension.forEach((row) => {
    const tr = document.createElement("tr");
    [
      row.label,
      Math.round((row.n_pairs / 6) * 4),
      fmt(row.image_rank_spearman),
      pct(row.pairwise_accuracy),
      row.n_pairs,
    ].forEach((value, index) => tr.append(createEl(index === 0 ? "th" : "td", "", value)));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  tableWrap.append(table);
  el.append(tableWrap);
}

function focusScore(candidate, prompt) {
  return candidate.taste_scores?.[prompt.focus_dimension];
}

const IMSCORE_LABELS = {
  hpsv21: "HPS v2.1",
  pickscore: "PickScore",
  clipscore: "CLIPScore",
  imagereward: "ImageReward",
  laion_aesthetic: "LAION aesthetic",
};

const IMSCORE_ORDER = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"];

function bestValues(prompt) {
  const candidates = prompt.candidates || [];
  const best = {
    taste: Math.max(
      ...candidates
        .map((candidate) => focusScore(candidate, prompt))
        .filter((value) => value !== undefined && value !== null),
    ),
    humanRank: Math.min(
      ...candidates
        .map((candidate) => candidate.human_mean_rank)
        .filter((value) => value !== undefined && value !== null),
    ),
    imscore: {},
  };

  IMSCORE_ORDER.forEach((metric) => {
    best.imscore[metric] = Math.max(
      ...candidates
        .map((candidate) => candidate.imscore_scores?.[metric])
        .filter((value) => value !== undefined && value !== null),
    );
  });
  return best;
}

function isBest(value, best, lowerIsBetter = false) {
  if (value === undefined || value === null || !Number.isFinite(best)) return false;
  const delta = Math.abs(Number(value) - Number(best));
  return lowerIsBetter ? delta < 1e-9 : delta < 1e-9;
}

function bestBadge() {
  return createEl("span", "best-badge", "Best");
}

function candidateImage(candidate, prompt) {
  const imgWrap = createEl("div", "image-wrap");
  const img = document.createElement("img");
  img.src = candidate.asset;
  img.alt = `${candidate.label} for ${prompt.title}`;
  img.loading = "lazy";
  imgWrap.append(img);
  return imgWrap;
}

function renderTasteScoreBox(candidate, prompt, best) {
  const score = focusScore(candidate, prompt);
  const tasteBox = createEl(
    "div",
    `candidate-score-box taste-box ${isBest(score, best.taste) ? "best-score" : ""}`.trim(),
  );
  tasteBox.append(createEl("span", "score-label", `TASTE ${prompt.dimension_label}`));
  const tasteValue = createEl("div", "score-value-line");
  tasteValue.append(createEl("strong", "score-number", fmt(score)));
  if (isBest(score, best.taste)) tasteValue.append(bestBadge());
  tasteBox.append(tasteValue);
  return tasteBox;
}

function renderImscoreList(candidate, best) {
  if (!candidate.imscore_scores) return null;
  const imscore = createEl("div", "imscore-list");
  imscore.append(createEl("div", "score-section-label", "Image/Text Scores"));
  IMSCORE_ORDER.forEach((metric) => {
    const value = candidate.imscore_scores[metric];
    if (value === undefined || value === null) return;
    const isWinner = isBest(value, best.imscore[metric]);
    const row = createEl("div", `imscore-row ${isWinner ? "best-score" : ""}`.trim());
    row.append(createEl("span", "", IMSCORE_LABELS[metric] || metric));
    const valueWrap = createEl("div", "imscore-value");
    valueWrap.append(createEl("strong", "", fmt(value)));
    if (isWinner) valueWrap.append(bestBadge());
    row.append(valueWrap);
    imscore.append(row);
  });
  return imscore.childNodes.length > 1 ? imscore : null;
}

function renderCandidate(candidate, prompt, best) {
  const card = createEl("article", "candidate");
  const meta = createEl("div", "candidate-meta");
  const title = createEl("div", "candidate-title", candidate.label);
  meta.append(title);

  meta.append(createEl("div", "score-section-label", "TASTE result"));
  const scoreGrid = createEl("div", "candidate-score-grid");
  const tasteBox = renderTasteScoreBox(candidate, prompt, best);
  const humanBox = createEl(
    "div",
    `candidate-score-box human-box ${
      isBest(candidate.human_mean_rank, best.humanRank, true) ? "best-score" : ""
    }`.trim(),
  );
  humanBox.append(createEl("span", "score-label", "Human mean rank"));
  const humanValue = createEl("div", "score-value-line");
  humanValue.append(createEl("strong", "score-number", fmt(candidate.human_mean_rank, 2)));
  if (isBest(candidate.human_mean_rank, best.humanRank, true)) humanValue.append(bestBadge());
  humanBox.append(humanValue);
  scoreGrid.append(tasteBox, humanBox);
  meta.append(scoreGrid);

  const human = createEl(
    "div",
    "human",
    `First-place votes ${candidate.human_first_place_votes}`,
  );
  meta.append(human);

  const imscore = renderImscoreList(candidate, best);
  if (imscore) meta.append(imscore);

  card.append(candidateImage(candidate, prompt), meta);
  return card;
}

function renderOodCandidate(candidate, prompt, best) {
  const card = createEl("article", "candidate ood-candidate");
  const meta = createEl("div", "candidate-meta");
  meta.append(createEl("div", "candidate-title", candidate.label));
  meta.append(createEl("div", "score-section-label", "TASTE result"));
  const scoreGrid = createEl("div", "candidate-score-grid single-score");
  scoreGrid.append(renderTasteScoreBox(candidate, prompt, best));
  meta.append(scoreGrid);
  const imscore = renderImscoreList(candidate, best);
  if (imscore) meta.append(imscore);
  card.append(candidateImage(candidate, prompt), meta);
  return card;
}

function orderedCandidates(prompt) {
  return [...prompt.candidates].sort((a, b) => {
    const rankA = Number.isFinite(a.human_mean_rank) ? a.human_mean_rank : Number.POSITIVE_INFINITY;
    const rankB = Number.isFinite(b.human_mean_rank) ? b.human_mean_rank : Number.POSITIVE_INFINITY;
    return rankA - rankB;
  });
}

function renderPairMatrix(prompt, ordered) {
  const pairRows = prompt.taste_ordered_pair_scores || prompt.taste_pair_scores;
  if (!pairRows?.length) return null;
  const candidates = ordered.map((candidate) => candidate.id);
  const labels = Object.fromEntries(prompt.candidates.map((c) => [c.id, c.label]));
  const probs = new Map();
  pairRows.forEach((pair) => {
    const value = pair.probabilities[prompt.focus_dimension];
    probs.set(`${pair.candidate_a}:${pair.candidate_b}`, value);
  });

  const section = createEl("div", "matrix-section");
  section.append(createEl("h3", "", `Pairwise model output: ${prompt.dimension_label}`));
  const grid = createEl("div", "matrix");
  grid.style.setProperty("--matrix-size", candidates.length + 1);
  grid.append(createEl("div", "matrix-cell matrix-head", "A beats B"));
  candidates.forEach((id) => grid.append(createEl("div", "matrix-cell matrix-head", labels[id])));

  candidates.forEach((rowId) => {
    grid.append(createEl("div", "matrix-cell matrix-head", labels[rowId]));
    candidates.forEach((colId) => {
      const cell = createEl("div", "matrix-cell");
      if (rowId === colId) {
        cell.textContent = "—";
        cell.classList.add("muted-cell");
      } else {
        const value = probs.get(`${rowId}:${colId}`);
        cell.textContent = value === undefined ? "pending" : fmt(value);
        if (value !== undefined && value >= 0.5) cell.classList.add("win-cell");
      }
      grid.append(cell);
    });
  });
  section.append(grid);
  return section;
}

function renderPrompt(prompt) {
  const section = createEl("section", "prompt-card");
  const head = createEl("div", "prompt-head");
  const title = createEl("div");
  title.append(
    createEl(
      "p",
      "eyebrow",
      `test · ${prompt.dimension_label} · prompt ${prompt.prompt_id} · ${prompt.scene_id}`,
    ),
  );
  title.append(createEl("h2", "", prompt.title));
  const promptText = createEl("p", "prompt-text", prompt.prompt);
  const stats = createEl(
    "p",
    "prompt-stats",
    `Human agreement ${fmt(prompt.mean_human_agreement, 2)} · unanimous pair rate ${fmt(prompt.unanimous_rate, 2)}`,
  );
  head.append(title, promptText, stats);

  const sortedCandidates = orderedCandidates(prompt);
  const best = bestValues(prompt);
  const candidates = createEl("div", "candidate-grid");
  sortedCandidates.forEach((candidate) => candidates.append(renderCandidate(candidate, prompt, best)));

  section.append(head, candidates);
  const matrix = renderPairMatrix(prompt, sortedCandidates);
  if (matrix) section.append(matrix);
  return section;
}

function renderOodPrompt(prompt) {
  const section = createEl("section", "prompt-card ood-card");
  const head = createEl("div", "prompt-head");
  const title = createEl("div");
  title.append(createEl("p", "eyebrow", `OOD probe · ${prompt.ood_type}`));
  title.append(createEl("h2", "", prompt.title));
  head.append(title, createEl("p", "prompt-text", prompt.prompt));

  const best = bestValues(prompt);
  const candidates = createEl("div", "candidate-grid");
  prompt.candidates.forEach((candidate) => candidates.append(renderOodCandidate(candidate, prompt, best)));

  section.append(head, candidates);
  const matrix = renderPairMatrix(prompt, prompt.candidates);
  if (matrix) section.append(matrix);
  return section;
}

function renderOodSection(data) {
  const el = byId("ood-prompts");
  if (!el || !data.ood_prompts?.length) return;
  el.innerHTML = "";
  const head = createEl("div", "section-head");
  head.append(createEl("p", "eyebrow", "Out-of-distribution probes"));
  head.append(createEl("h2", "", "TASTE Preference outside the main benchmark"));
  head.append(
    createEl(
      "p",
      "prompt-text",
      "These probes use the same prompt format as the benchmark, but have no human ranking labels. They show only TASTE Preference and image/text metrics.",
    ),
  );
  el.append(head);
  data.ood_prompts.forEach((prompt) => el.append(renderOodPrompt(prompt)));
}

async function main() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  renderSummary(data);
  renderEvaluation(data);
  const prompts = byId("prompts");
  prompts.innerHTML = "";
  data.prompts.forEach((prompt) => prompts.append(renderPrompt(prompt)));
  renderOodSection(data);
}

main().catch((error) => {
  console.error(error);
  byId("prompts").textContent = "Failed to load TASTE sample data.";
});
