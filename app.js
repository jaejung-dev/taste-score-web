const DATA_URL = "data.json?v=20260603-pairwise-1";

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
  [
    ["Test rows", data.summary.test_rows ?? data.summary.prompts],
    ["Test pairs", data.summary.test_unique_pairs ?? data.summary.pairs],
    ["Selected prompts", data.summary.selected_prompts ?? data.summary.prompts],
    ["Selected images", data.summary.selected_candidates ?? data.summary.candidates],
    ["TASTE scored", data.summary.taste_scored ? "yes" : "pending"],
    ["ImScore scored", data.summary.imscore_scored ? "yes" : "pending"],
  ].forEach(([label, value]) => {
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

function renderCandidate(candidate, prompt) {
  const card = createEl("article", "candidate");
  const imgWrap = createEl("div", "image-wrap");
  const img = document.createElement("img");
  img.src = candidate.asset;
  img.alt = `${candidate.label} for ${prompt.title}`;
  img.loading = "lazy";
  imgWrap.append(img);

  const meta = createEl("div", "candidate-meta");
  const title = createEl("div", "candidate-title", candidate.label);
  meta.append(title);

  const score = focusScore(candidate, prompt);
  const scoreGrid = createEl("div", "candidate-score-grid");
  const tasteBox = createEl("div", "candidate-score-box taste-box");
  tasteBox.append(createEl("span", "score-label", `TASTE ${prompt.dimension_label}`));
  tasteBox.append(createEl("strong", "score-number", fmt(score)));
  const humanBox = createEl("div", "candidate-score-box human-box");
  humanBox.append(createEl("span", "score-label", "Human mean rank"));
  humanBox.append(createEl("strong", "score-number", fmt(candidate.human_mean_rank, 2)));
  scoreGrid.append(tasteBox, humanBox);
  meta.append(scoreGrid);

  const human = createEl(
    "div",
    "human",
    `First-place votes ${candidate.human_first_place_votes}`,
  );
  meta.append(human);

  if (candidate.imscore_scores) {
    const imscore = createEl("div", "imscore-list");
    IMSCORE_ORDER.forEach((metric) => {
      const value = candidate.imscore_scores[metric];
      if (value === undefined || value === null) return;
      const row = createEl("div", "imscore-row");
      row.append(createEl("span", "", IMSCORE_LABELS[metric] || metric));
      row.append(createEl("strong", "", fmt(value)));
      imscore.append(row);
    });
    if (imscore.childNodes.length) {
      meta.append(imscore);
    }
  }

  card.append(imgWrap, meta);
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
  const candidates = createEl("div", "candidate-grid");
  sortedCandidates.forEach((candidate) => candidates.append(renderCandidate(candidate, prompt)));

  section.append(head, candidates);
  const matrix = renderPairMatrix(prompt, sortedCandidates);
  if (matrix) section.append(matrix);
  return section;
}

async function main() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  renderSummary(data);
  renderEvaluation(data);
  const prompts = byId("prompts");
  prompts.innerHTML = "";
  data.prompts.forEach((prompt) => prompts.append(renderPrompt(prompt)));
}

main().catch((error) => {
  console.error(error);
  byId("prompts").textContent = "Failed to load TASTE sample data.";
});
