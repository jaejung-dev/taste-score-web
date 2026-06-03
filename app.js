const DATA_URL = "data.json?v=20260603-eval-1";

const fmt = (value, digits = 3) =>
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
  el.append(createEl("h2", "", "Human Agreement Check"));
  el.append(
    createEl(
      "p",
      "score-note",
      "Full battles_test.csv evaluation. These numbers compare model pairwise outputs against the 5-rater human majority on the manifest, and should be read as manifest-level evidence rather than a new official validation split claim.",
    ),
  );

  const cards = createEl("div", "metric-grid");
  [
    ["Pairwise accuracy", pct(overview.pairwise_accuracy), `${overview.n_pairs} unique pairs`],
    ["Vote-share Spearman", fmt(overview.vote_share_spearman), "model probability vs human vote share"],
    ["Top-1 image match", pct(overview.prompt_top1_accuracy), `${overview.n_prompt_groups} prompt groups`],
    ["Image-rank Spearman", fmt(overview.image_rank_spearman), "aggregate score vs inverse human rank"],
  ].forEach(([label, value, detail]) => {
    const card = createEl("div", "metric-card");
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
  ["Dimension", "Pairs", "Pairwise Acc.", "Vote Spearman", "Top-1 Match", "Rank Spearman"].forEach((label) =>
    head.append(createEl("th", "", label)),
  );
  thead.append(head);
  const tbody = document.createElement("tbody");
  data.evaluation.by_dimension.forEach((row) => {
    const tr = document.createElement("tr");
    [
      row.label,
      row.n_pairs,
      pct(row.pairwise_accuracy),
      fmt(row.vote_share_spearman),
      pct(row.prompt_top1_accuracy),
      fmt(row.image_rank_spearman),
    ].forEach((value, index) => tr.append(createEl(index === 0 ? "th" : "td", "", value)));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  tableWrap.append(table);
  el.append(tableWrap);
}

function renderSelection(data) {
  const el = byId("selection");
  if (!el || !data.sample_selection) return;
  el.innerHTML = "";
  el.append(createEl("h3", "", "Sample selection"));
  el.append(createEl("p", "", data.sample_selection.method));
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

  if (candidate.taste_overall !== undefined) {
    meta.append(createEl("div", "overall", `Model output overall ${fmt(candidate.taste_overall)}`));
  }

  card.append(imgWrap, meta);
  return card;
}

function renderImageScores(prompt, dimensionLabels, dimensionOrder) {
  const section = createEl("div", "score-table-section");
  section.append(createEl("h3", "", "Per-image model output scores"));
  section.append(
    createEl(
      "p",
      "score-note",
      prompt.model_output_note ||
        "Per-image scores are derived by averaging pairwise win probabilities.",
    ),
  );

  const tableWrap = createEl("div", "table-wrap");
  const table = createEl("table", "score-table");
  const columns = [
    { key: "image", label: "Image" },
    { key: "human", label: "Human rank" },
    { key: prompt.focus_dimension, label: `${prompt.dimension_label} score`, focus: true },
    { key: "overall", label: "Overall" },
    ...dimensionOrder
      .filter((dimension) => dimension !== prompt.focus_dimension)
      .map((dimension) => ({
        key: dimension,
        label: dimensionLabels[dimension] || dimension,
      })),
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
      else if (column.key === "overall") value = fmt(candidate.model_output_overall);
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

function renderRankList(title, items, scoreFormatter) {
  const box = createEl("div", "rank-box");
  box.append(createEl("h3", "", title));
  const list = createEl("ol", "rank-list");
  items.forEach((item) => {
    const li = createEl("li", "");
    li.append(createEl("span", "", item.label));
    li.append(createEl("strong", "", scoreFormatter(item)));
    list.append(li);
  });
  box.append(list);
  return box;
}

function renderRanks(prompt) {
  const wrap = createEl("div", "rank-grid");
  const human = [...prompt.candidates]
    .sort((a, b) => a.human_mean_rank - b.human_mean_rank)
    .map((candidate) => ({
      label: candidate.label,
      score: candidate.human_mean_rank,
    }));
  wrap.append(
    renderRankList("Human ranking", human, (item) => `rank ${fmt(item.score, 2)}`),
  );

  const taste = prompt.taste_rankings?.[prompt.focus_dimension];
  if (taste?.length) {
    wrap.append(
      renderRankList(
        `TASTE ${prompt.dimension_label}`,
        taste,
        (item) => fmt(item.score),
      ),
    );
  } else {
    const pending = createEl("div", "rank-box pending-box");
    pending.append(createEl("h3", "", `TASTE ${prompt.dimension_label}`));
    pending.append(createEl("p", "", "Pairwise scores are still pending for this snapshot."));
    wrap.append(pending);
  }
  return wrap;
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

  section.append(head, candidates, renderRanks(prompt));
  section.append(
    renderImageScores(
      prompt,
      window.__dimensionLabels || {},
      window.__dimensionOrder || prompt.taste_dimensions || [],
    ),
  );
  const matrix = renderPairMatrix(prompt);
  if (matrix) section.append(matrix);
  return section;
}

async function main() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  window.__dimensionLabels = data.dimension_labels || {};
  window.__dimensionOrder = data.dimension_order || data.taste_dimensions || [];
  renderSummary(data);
  renderEvaluation(data);
  renderSelection(data);
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
