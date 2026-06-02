const DATA_URL = "data.json?v=20260603-draft-1";

const fmt = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "pending"
    : Number(value).toFixed(digits);

const byId = (id) => document.getElementById(id);

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
    ["Prompts", data.summary.prompts],
    ["Candidates", data.summary.candidates],
    ["Pairwise battles", data.summary.pairs],
    ["TASTE scored", data.summary.taste_scored ? "yes" : "pending"],
  ].forEach(([label, value]) => {
    const card = createEl("div", "summary-card");
    card.append(createEl("span", "summary-value", value));
    card.append(createEl("span", "summary-label", label));
    el.append(card);
  });
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
  const scoreText = score === undefined ? "TASTE pending" : `TASTE ${fmt(score)}`;
  const scoreClass = score === undefined ? "score muted" : "score";
  title.append(createEl("span", scoreClass, scoreText));

  const human = createEl(
    "div",
    "human",
    `Human mean rank ${fmt(candidate.human_mean_rank, 2)} · first-place votes ${candidate.human_first_place_votes}`,
  );
  meta.append(title, human);

  if (candidate.taste_overall !== undefined) {
    meta.append(createEl("div", "overall", `TASTE overall ${fmt(candidate.taste_overall)}`));
  }

  card.append(imgWrap, meta);
  return card;
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
  section.append(createEl("h3", "", `Pairwise TASTE probabilities: ${prompt.dimension_label}`));
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
  title.append(createEl("p", "eyebrow", `${prompt.track} · ${prompt.dimension_label}`));
  title.append(createEl("h2", "", prompt.title));
  const promptText = createEl("p", "prompt-text", prompt.prompt);
  head.append(title, promptText);

  const candidates = createEl("div", "candidate-grid");
  prompt.candidates.forEach((candidate) => candidates.append(renderCandidate(candidate, prompt)));

  section.append(head, candidates, renderRanks(prompt));
  const matrix = renderPairMatrix(prompt);
  if (matrix) section.append(matrix);
  return section;
}

async function main() {
  const response = await fetch(DATA_URL);
  const data = await response.json();
  renderSummary(data);
  const prompts = byId("prompts");
  prompts.innerHTML = "";
  data.prompts.forEach((prompt) => prompts.append(renderPrompt(prompt)));
}

main().catch((error) => {
  console.error(error);
  byId("prompts").textContent = "Failed to load TASTE sample data.";
});
