const DATA_URL = "data.json?v=20260603-norankboxes-1";

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
      "Primary metric: Spearman correlation between the model's per-image aggregate score and human mean rank across battles_test.csv. Human rank is inverted because rank 1 is best, so higher means the model orders images more like the human panel.",
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
  const score = focusScore(candidate, prompt);
  const scoreText = score === undefined ? "pending" : `model ${fmt(score)}`;
  const scoreClass = score === undefined ? "score muted" : "score";
  title.append(createEl("span", scoreClass, scoreText));

  const human = createEl(
    "div",
    "human",
    `Human mean rank ${fmt(candidate.human_mean_rank, 2)} · first-place votes ${candidate.human_first_place_votes}`,
  );
  meta.append(title, human);

  card.append(imgWrap, meta);
  return card;
}

function renderImageScores(prompt) {
  const section = createEl("div", "score-table-section");
  section.append(createEl("h3", "", "TASTE score vs human rank"));

  const tableWrap = createEl("div", "table-wrap");
  const table = createEl("table", "score-table");
  const columns = [
    { key: "image", label: "Candidate" },
    { key: "human", label: "Human rank" },
    { key: prompt.focus_dimension, label: `TASTE ${prompt.dimension_label} score`, focus: true },
  ];
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((column) => headRow.append(createEl("th", "", column.label)));
  thead.append(headRow);

  const tbody = document.createElement("tbody");
  prompt.candidates.forEach((candidate) => {
    const row = document.createElement("tr");
    columns.forEach((column, index) => {
      let value;
      if (column.key === "image") value = candidate.label;
      else if (column.key === "human") value = fmt(candidate.human_mean_rank, 2);
      else value = fmt(candidate.model_output_scores?.[column.key]);
      const cell = createEl(index === 0 ? "th" : "td", "", value);
      if (column.focus) cell.classList.add("focus-score");
      row.append(cell);
    });
    tbody.append(row);
  });
  table.append(thead, tbody);
  tableWrap.append(table);
  section.append(tableWrap);
  return section;
}

function renderPairMatrix(prompt) {
  if (!prompt.taste_pair_scores?.length) return null;
  const candidates = prompt.candidates.map((candidate) => candidate.id);
  const labels = Object.fromEntries(prompt.candidates.map((c) => [c.id, c.label]));
  const probs = new Map();
  prompt.taste_pair_scores.forEach((pair) => {
    const value = pair.probabilities[prompt.focus_dimension];
    probs.set(`${pair.candidate_a}:${pair.candidate_b}`, value);
    probs.set(`${pair.candidate_b}:${pair.candidate_a}`, 1 - value);
  });

  const section = createEl("div", "matrix-section");
  section.append(createEl("h3", "", `Raw pairwise model output: ${prompt.dimension_label}`));
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
        cell.textContent = fmt(value);
        if (value >= 0.5) cell.classList.add("win-cell");
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

  const candidates = createEl("div", "candidate-grid");
  prompt.candidates.forEach((candidate) => candidates.append(renderCandidate(candidate, prompt)));

  section.append(head, candidates);
  section.append(
    renderImageScores(prompt),
  );
  const matrix = renderPairMatrix(prompt);
  if (matrix) section.append(matrix);
  return section;
}

async function main() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  renderSummary(data);
  renderEvaluation(data);
  const lede = document.querySelector(".lede");
  if (lede && data.score_explanation) {
    lede.textContent = data.score_explanation;
  }
  const prompts = byId("prompts");
  prompts.innerHTML = "";
  data.prompts.forEach((prompt) => prompts.append(renderPrompt(prompt)));
}

main().catch((error) => {
  console.error(error);
  byId("prompts").textContent = "Failed to load TASTE sample data.";
});
