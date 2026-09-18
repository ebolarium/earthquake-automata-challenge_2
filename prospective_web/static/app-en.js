"use strict";

const state = {
  dashboard: null, comparison: "gated_vs_etas", points: [], mapRegion: "california-relm",
  mapLayer: "gated_etas_log_ratio", mapData: null, mapCache: new Map(), mapPoints: [],
};
const canvas = document.getElementById("score-chart");
const context = canvas.getContext("2d");
const stage = document.getElementById("chart-stage");
const tooltip = document.getElementById("chart-tooltip");
const mapCanvas = document.getElementById("forecast-grid-map");
const mapContext = mapCanvas.getContext("2d");
const mapStage = document.getElementById("forecast-map-stage");
const mapTooltip = document.getElementById("forecast-map-tooltip");

async function loadDashboard() {
  const refresh = document.getElementById("refresh-button");
  refresh.classList.add("loading");
  hideError();
  try {
    const response = await fetch("/api/dashboard", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.dashboard = await response.json();
    renderAll();
  } catch (error) {
    showError(`Dashboard data could not be loaded: ${error.message}`);
    setText("status-label", "Data unavailable");
    document.getElementById("status-dot").className = "attention";
  } finally {
    refresh.classList.remove("loading");
  }
}

function renderAll() {
  renderStatus();
  renderDryRun();
  renderFilters();
  renderMetrics();
  renderChart();
  renderRegions();
  renderScores();
  renderProtocol();
  renderMapRegionControl();
}

async function loadForecastMap(force = false) {
  const empty = document.getElementById("forecast-map-empty");
  empty.textContent = "Loading map";
  empty.classList.remove("hidden");
  mapTooltip.classList.remove("visible");
  try {
    if (!force && state.mapCache.has(state.mapRegion)) {
      state.mapData = state.mapCache.get(state.mapRegion);
    } else {
      const response = await fetch(`/api/forecast-map?region=${encodeURIComponent(state.mapRegion)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.mapData = await response.json();
      state.mapCache.set(state.mapRegion, state.mapData);
    }
    renderForecastMap();
    empty.classList.add("hidden");
  } catch (error) {
    state.mapData = null;
    empty.textContent = "Forecast map unavailable";
    showError(`Forecast map could not be loaded: ${error.message}`);
  }
}

async function loadNewsletterStatus() {
  try {
    const response = await fetch("/api/newsletter", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    setText("newsletter-count", formatInteger(result.subscribers));
  } catch (_) {
    setText("newsletter-count", "—");
  }
}

async function submitNewsletter(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  const message = document.getElementById("newsletter-message");
  const data = new FormData(form);
  button.disabled = true;
  message.className = "";
  message.textContent = "Processing subscription…";
  try {
    const response = await fetch("/api/newsletter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: data.get("email"), company: data.get("company"), locale: "en" }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    form.reset();
    message.className = "success";
    message.textContent = "A verification link was sent. Check your inbox.";
  } catch (_) {
    message.className = "error";
    message.textContent = "Subscription failed. Please try again later.";
  } finally {
    button.disabled = false;
  }
}

function renderMapRegionControl() {
  const control = document.getElementById("map-region-control");
  control.replaceChildren();
  state.dashboard.regions.forEach((region) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `segment${state.mapRegion === region.region_id ? " active" : ""}`;
    button.textContent = shortRegion(region.name);
    button.addEventListener("click", () => {
      state.mapRegion = region.region_id;
      renderMapRegionControl();
      loadForecastMap();
    });
    control.appendChild(button);
  });
}

function renderForecastMap() {
  const data = state.mapData;
  if (!data) return;
  const layerNames = { etas: "ETAS", safe: "Safe", fixed: "Fixed expert", gated: "Gated", gated_etas_log_ratio: "Gated − ETAS" };
  setText("forecast-map-title", data.region_name);
  setText("forecast-map-window", `${formatDateTime(data.target_start)} → ${formatDateTime(data.target_end)} UTC`);
  setText("forecast-map-layer", layerNames[state.mapLayer]);
  setText("map-fact-target", formatDate(data.target_start));
  setText("map-fact-published", `${formatDateTime(data.published_at)} UTC`);
  setText("map-fact-cells", formatInteger(data.summary.cells));
  setText("map-fact-etas", formatMapTotal(data.summary.etas_total));
  setText("map-fact-safe", formatMapTotal(data.summary.safe_total));
  setText("map-fact-ch008", formatMapTotal(data.summary.gated_total ?? data.summary.ch008_total));
  setText("map-semantics-note", data.semantics === "one_day_expected_count_per_cell"
    ? "Layers show one-day expected event counts per cell."
    : "These layers show the pre-target direct-background mass used in sequential ETAS evaluation.");
  const difference = state.mapLayer.includes("log_ratio");
  const scale = document.querySelector(".map-scale");
  scale.classList.toggle("sequential", !difference);
  setText("map-scale-low", difference ? "ETAS higher" : "Low");
  setText("map-scale-high", difference ? "Gated higher" : "High");
  resizeForecastMap();
}

function resizeForecastMap() {
  if (!state.mapData) return;
  const rect = mapStage.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  mapCanvas.width = Math.max(1, Math.round(rect.width * ratio));
  mapCanvas.height = Math.max(1, Math.round(rect.height * ratio));
  mapCanvas.style.width = `${rect.width}px`;
  mapCanvas.style.height = `${rect.height}px`;
  mapContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawForecastMap(rect.width, rect.height);
}

function drawForecastMap(width, height) {
  const data = state.mapData;
  mapContext.clearRect(0, 0, width, height);
  mapContext.fillStyle = "#edf1ef";
  mapContext.fillRect(0, 0, width, height);
  const origins = data.origins;
  const values = data.layers[state.mapLayer];
  const spacing = data.spacing_degrees;
  const longitudes = origins.map((point) => point[0]);
  const latitudes = origins.map((point) => point[1]);
  const bounds = {
    minLon: Math.min(...longitudes), maxLon: Math.max(...longitudes) + spacing,
    minLat: Math.min(...latitudes), maxLat: Math.max(...latitudes) + spacing,
  };
  const cosine = Math.max(0.35, Math.cos((bounds.minLat + bounds.maxLat) / 2 * Math.PI / 180));
  const padding = 22;
  const scale = Math.min((width - padding * 2) / ((bounds.maxLon - bounds.minLon) * cosine), (height - padding * 2) / (bounds.maxLat - bounds.minLat));
  const usedWidth = (bounds.maxLon - bounds.minLon) * cosine * scale;
  const usedHeight = (bounds.maxLat - bounds.minLat) * scale;
  const left = (width - usedWidth) / 2;
  const top = (height - usedHeight) / 2;
  const positives = values.filter((value) => value > 0);
  const logs = positives.map((value) => Math.log10(value));
  const low = quantile(logs, 0.02);
  const high = quantile(logs, 0.98);
  const differenceExtent = Math.max(0.000001, quantile(values.map(Math.abs), 0.98));
  state.mapPoints = [];
  origins.forEach(([longitude, latitude], index) => {
    const x = left + (longitude - bounds.minLon) * cosine * scale;
    const y = top + (bounds.maxLat - latitude - spacing) * scale;
    const cellWidth = spacing * cosine * scale + 0.45;
    const cellHeight = spacing * scale + 0.45;
    const value = values[index];
    const normalized = state.mapLayer.includes("log_ratio")
      ? Math.max(-1, Math.min(1, value / differenceExtent))
      : high === low ? 0.5 : Math.max(0, Math.min(1, (Math.log10(Math.max(value, Number.MIN_VALUE)) - low) / (high - low)));
    mapContext.fillStyle = state.mapLayer.includes("log_ratio") ? differenceColor(normalized) : rateColor(normalized);
    mapContext.fillRect(x, y, cellWidth, cellHeight);
    state.mapPoints.push({ x, y, width: cellWidth, height: cellHeight, longitude, latitude, value });
  });
}

function handleMapPointer(event) {
  const rect = mapCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const point = state.mapPoints.find((item) => x >= item.x && x <= item.x + item.width && y >= item.y && y <= item.y + item.height);
  if (!point) { mapTooltip.classList.remove("visible"); return; }
  const value = state.mapLayer.includes("log_ratio") ? formatSigned(point.value, 4) : formatMapTotal(point.value);
  mapTooltip.innerHTML = `<strong>${point.latitude.toFixed(2)}°, ${point.longitude.toFixed(2)}°</strong><br>${value}`;
  mapTooltip.style.left = `${Math.min(x + 10, rect.width - 125)}px`;
  mapTooltip.style.top = `${Math.max(y - 42, 8)}px`;
  mapTooltip.classList.add("visible");
}

function quantile(values, probability) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * probability))];
}
function mixColor(from, to, amount) { return `rgb(${from.map((value, index) => Math.round(value + (to[index] - value) * amount)).join(",")})`; }
function rateColor(value) { return value < 0.55 ? mixColor([238, 241, 213], [216, 162, 55], value / 0.55) : mixColor([216, 162, 55], [8, 127, 122], (value - 0.55) / 0.45); }
function differenceColor(value) { return value < 0 ? mixColor([244, 246, 245], [207, 91, 76], -value) : mixColor([244, 246, 245], [8, 127, 122], value); }
function formatMapTotal(value) { return Number(value).toLocaleString("en-US", { maximumSignificantDigits: 5 }); }

function renderDryRun() {
  const progress = state.dashboard.test_progress || state.dashboard.dry_run;
  const formal = state.dashboard.protocol.mode === "prospective";
  const notice = document.getElementById("run-notice");
  const copy = (formal ? {
    awaiting_scores: ["Prospective test started", "Waiting for the first completed target-day score."],
    running: ["Prospective test running", `Calendar ${progress.calendar_days_elapsed}/${progress.planned_days} · ${progress.successful_scored_days} successfully scored days.`],
    settling: ["The fixed 365-day window is complete", `Awaiting final scores for ${progress.successful_scored_days} successful days: ${progress.final_days}/${progress.successful_scored_days}.`],
    complete: ["Prospective test complete", `${progress.final_days} successful days finalized · ${progress.missed_calendar_days} days missed.`],
  } : {
    awaiting_scores: ["Dry run started", "Waiting for the first completed target-day score."],
    running: ["Dry run running", `Calendar ${progress.calendar_days_elapsed}/${progress.planned_days} · ${progress.successful_scored_days} successfully scored days.`],
    settling: ["The fixed 14-day window is complete", `Awaiting final scores for ${progress.successful_scored_days} successful days: ${progress.final_days}/${progress.successful_scored_days}.`],
    complete: ["Dry run complete", `${progress.final_days} successful days finalized · ${progress.missed_calendar_days} days missed.`],
  })[progress.phase];
  const shownDays = progress.calendar_days_elapsed;
  notice.className = `run-notice ${progress.phase.replace("_", "-")}`;
  setText("run-eyebrow", progress.phase === "complete" ? "COMPLETE" : progress.phase === "settling" ? "SETTLING" : formal ? "PROSPECTIVE TEST" : "DRY RUN");
  setText("run-title", copy[0]);
  setText("run-detail", copy[1]);
  setText("run-progress-label", `${shownDays}/${progress.planned_days} days`);
  document.getElementById("run-progress-bar").style.width = `${Math.min(100, shownDays / progress.planned_days * 100)}%`;
}

function renderStatus() {
  const ok = state.dashboard.pipeline_status === "ok";
  const invalid = state.dashboard.pooled_primary_claim_status === "inconclusive";
  setText("status-label", invalid ? "Primary claim inconclusive" : ok ? "Pipeline healthy" : "Attention required");
  setText("updated-label", `${formatDateTime(state.dashboard.generated_at)} UTC`);
  document.getElementById("status-dot").className = ok ? "ok" : "attention";
}

function selectedScores(revision = "provisional") {
  return (state.dashboard?.daily_scores || []).filter((score) => score.revision === revision);
}

function aggregate(scores) {
  const comparisons = scores.map((score) => score.multi_model?.comparisons?.[state.comparison]).filter(Boolean);
  const events = comparisons.reduce((sum, score) => sum + score.events, 0);
  const gain = comparisons.reduce((sum, score) => sum + score.total_gain, 0);
  const mean = events ? gain / events : null;
  return { days: scores.length, events, gain, mean };
}

function renderFilters() {
  const labels = [["gated_vs_etas", "Gated / ETAS"], ["safe_vs_etas", "Safe / ETAS"], ["fixed_vs_etas", "Fixed / ETAS"], ["gated_vs_safe", "Gated / Safe"]];
  const control = document.getElementById("comparison-filter");
  control.replaceChildren();
  labels.forEach(([id, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `segment${state.comparison === id ? " active" : ""}`;
    button.textContent = label;
    button.addEventListener("click", () => { state.comparison = id; renderAll(); });
    control.appendChild(button);
  });
}

function renderMetrics() {
  const provisional = aggregate(selectedScores());
  const models = state.dashboard.model_comparisons?.provisional;
  const safe = models?.comparisons?.safe_vs_etas;
  const gate = models?.latest_gate;
  setText("metric-target", state.dashboard.latest_target_start ? formatDate(state.dashboard.latest_target_start) : "Waiting");
  setText("metric-igpe", formatGain(provisional.mean));
  setText("metric-factor", comparisonLabel(state.comparison));
  setText("metric-safe-igpe", formatGain(safe?.mean_igpe));
  setText("metric-gate-weight", gate ? `${(gate.weight * 100).toFixed(1)}%` : "—");
  setText("metric-gate-evidence", gate ? `log BF ${formatSigned(gate.log_bayes_factor, 3)}` : "BF20 hurdle");
  setText("metric-events", formatInteger(provisional.events));
  const progress = state.dashboard.test_progress || state.dashboard.dry_run;
  setText("metric-days", `${progress.successful_scored_days} successful · ${progress.calendar_days_elapsed}/${progress.planned_days} calendar`);
  renderSummary("provisional", aggregate(selectedScores("provisional")));
  renderSummary("final", aggregate(selectedScores("final")));
  setText("incident-count", formatInteger(state.dashboard.open_incidents));
  setText("incident-detail", state.dashboard.open_incidents ? "Open incident" : "No open warnings");
}

function renderSummary(id, value) {
  setText(`${id}-score`, formatGain(value.mean));
  setText(`${id}-detail`, value.days ? `${value.events} events · ${formatSigned(value.gain)} total` : (id === "final" ? "7-day settlement window" : "No score yet"));
}

function renderChart() {
  const scores = selectedScores().sort((a, b) => a.target_date.localeCompare(b.target_date));
  document.getElementById("chart-empty").classList.toggle("hidden", scores.length > 0);
  const total = aggregate(scores);
  setText("chart-total", total.mean === null ? "—" : `${formatSigned(total.mean)} IGPE`);
  setText("chart-subtitle", `${comparisonLabel(state.comparison)} · provisional`);
  resizeCanvas();
}

function resizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawChart(rect.width, rect.height);
}

function drawChart(width, height) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcfb";
  context.fillRect(0, 0, width, height);
  const scores = selectedScores().sort((a, b) => a.target_date.localeCompare(b.target_date));
  state.points = [];
  if (!scores.length) return;
  const padding = { left: 48, right: 18, top: 22, bottom: 35 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const values = scores.map((score) => score.multi_model?.comparisons?.[state.comparison]?.mean_igpe || 0);
  const extent = Math.max(0.002, ...values.map(Math.abs)) * 1.2;
  const y = (value) => padding.top + (extent - value) / (extent * 2) * innerHeight;
  const zero = y(0);
  context.strokeStyle = "#d6dedb";
  context.lineWidth = 1;
  [-extent, 0, extent].forEach((value) => {
    context.beginPath(); context.moveTo(padding.left, y(value)); context.lineTo(width - padding.right, y(value)); context.stroke();
    context.fillStyle = "#68736f"; context.font = "9px system-ui"; context.textAlign = "right"; context.fillText(value.toFixed(3), padding.left - 7, y(value) + 3);
  });
  const slot = innerWidth / scores.length;
  const barWidth = Math.max(4, Math.min(28, slot * 0.58));
  scores.forEach((score, index) => {
    const value = score.multi_model?.comparisons?.[state.comparison]?.mean_igpe || 0;
    const x = padding.left + slot * index + slot / 2;
    const top = Math.min(zero, y(value));
    const barHeight = Math.max(1, Math.abs(y(value) - zero));
    context.fillStyle = value >= 0 ? "#087f7a" : "#cf5b4c";
    context.fillRect(x - barWidth / 2, top, barWidth, barHeight);
    state.points.push({ x, y: top, width: barWidth, height: barHeight, score });
    if (scores.length <= 14 || index % Math.ceil(scores.length / 10) === 0) {
      context.save(); context.translate(x, height - 9); context.rotate(-0.45); context.fillStyle = "#68736f"; context.font = "9px system-ui"; context.textAlign = "right"; context.fillText(formatShortDate(score.target_date), 0, 0); context.restore();
    }
  });
}

function renderRegions() {
  const body = document.getElementById("regions-body");
  body.replaceChildren();
  state.dashboard.regions.forEach((region) => {
    const row = document.createElement("tr");
    const forecast = region.latest_forecast;
    row.append(
      cell(primary(region.name, `M≥${region.minimum_magnitude.toFixed(1)} · ${region.catalog_source}`)),
      cell(status(forecast, region.operations)),
      cell(forecast ? formatDate(forecast.target_start) : "—"),
      cell(primary(region.latest_catalog ? formatDateTime(region.latest_catalog.cutoff) : "—", region.latest_catalog ? `${region.latest_catalog.window_events} events / 30 days` : "No snapshot")),
      cell(comparisonValue(region, "gated_vs_etas")),
      cell(formatP(region.csep?.provisional?.challenger?.n_test_two_sided_p)),
      cell(formatP(region.csep?.provisional?.challenger?.l_test_lower_tail_p)),
      cell(formatP(region.csep?.provisional?.r_test?.one_sided_p)),
    );
    body.appendChild(row);
  });
}

function renderScores() {
  const body = document.getElementById("scores-body");
  const scores = [...state.dashboard.daily_scores].sort((a, b) => b.target_date.localeCompare(a.target_date) || a.region_id.localeCompare(b.region_id));
  body.replaceChildren();
  document.getElementById("scores-empty").classList.toggle("hidden", scores.length > 0);
  scores.forEach((score) => {
    const comparisons = score.multi_model?.comparisons || {};
    const row = document.createElement("tr");
    row.append(
      cell(formatDate(score.target_date)), cell(regionName(score.region_id)),
      cell(score.revision === "final" ? "Final" : "Provisional"),
      cell(formatInteger(score.event_count)), cell(formatGain(comparisons.gated_vs_etas?.mean_igpe)),
      cell(formatP(score.csep?.challenger?.n_test_two_sided_p)),
      cell(formatP(score.csep?.challenger?.l_test_lower_tail_p)),
      cell(formatP(score.csep?.r_test?.one_sided_p)),
    );
    body.appendChild(row);
  });
}

function renderProtocol() {
  const protocol = state.dashboard.protocol;
  const facts = document.getElementById("protocol-facts");
  facts.replaceChildren();
  const mode = protocol.mode === "prospective" ? "365-day prospective test" : "14-day dry run";
  setText("research-question", state.dashboard.research_question || "—");
  [["Identity", protocol.protocol_id], ["Mode", mode], ["Prospective claim", protocol.counts_toward_prospective_claim ? "Included" : "Not included"], ["Regions", String(state.dashboard.regions.length)], ["Final delay", `${protocol.settled_score_delay_days} days`]].forEach(([label, value]) => {
    const div = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd"); dt.textContent = label; dd.textContent = value; div.append(dt, dd); facts.appendChild(div);
  });
  const regions = document.getElementById("protocol-regions");
  regions.replaceChildren();
  state.dashboard.regions.forEach((region) => {
    const item = document.createElement("div"); item.className = "protocol-region";
    const copy = document.createElement("div"); const title = document.createElement("strong"); const detail = document.createElement("small");
    title.textContent = region.name; detail.textContent = depthLabel(region); copy.append(title, detail);
    const magnitude = document.createElement("span"); magnitude.textContent = `M≥${region.minimum_magnitude.toFixed(1)}`;
    item.append(copy, magnitude); regions.appendChild(item);
  });
  const ranges = document.getElementById("parameter-ranges");
  ranges.replaceChildren();
  Object.entries(state.dashboard.parameter_reporting?.etas_typical_ranges || {}).forEach(([name, value]) => {
    const div = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd");
    dt.textContent = name; dd.textContent = `${Number(value[0]).toPrecision(4)} – ${Number(value[1]).toPrecision(4)}`;
    div.append(dt, dd); ranges.appendChild(div);
  });
}

function handleChartPointer(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const point = state.points.find((item) => Math.abs(item.x - x) <= Math.max(9, item.width));
  if (!point) { tooltip.classList.remove("visible"); return; }
  const value = point.score.multi_model?.comparisons?.[state.comparison]?.mean_igpe;
  tooltip.innerHTML = `<strong>${formatDate(point.score.target_date)}</strong><br>${comparisonLabel(state.comparison)} · ${formatGain(value)} IGPE<br>${point.score.event_count} events`;
  tooltip.style.left = `${Math.min(point.x + 10, rect.width - 150)}px`;
  tooltip.style.top = `${Math.max(point.y - 50, 8)}px`;
  tooltip.classList.add("visible");
}

function primary(titleText, detailText) { const wrap = document.createElement("div"); wrap.className = "primary-cell"; const title = document.createElement("strong"); const detail = document.createElement("small"); title.textContent = titleText; detail.textContent = detailText; wrap.append(title, detail); return wrap; }
function status(forecast, operations) { const span = document.createElement("span"); const invalid = operations && !operations.primary_eligible; span.className = `status-pill${forecast && forecast.status === "published" && !invalid ? "" : " waiting"}`; span.textContent = invalid ? "Primary-ineligible" : forecast && forecast.status === "published" ? "Published" : "Waiting"; return span; }
function scoreValue(summary) { if (!summary.days) return "Waiting"; const span = document.createElement("span"); span.className = gainClass(summary.mean_igpe); span.textContent = `${formatGain(summary.mean_igpe)} · ${summary.events} events`; return span; }
function comparisonValue(region, name) { const value = region.model_comparisons?.provisional?.comparisons?.[name]; if (!value?.events) return "Waiting"; const span = document.createElement("span"); span.className = gainClass(value.mean_igpe); span.textContent = `${formatGain(value.mean_igpe)} · ${value.events} events`; return span; }
function comparisonLabel(name) { return ({ gated_vs_etas: "Gated model / ETAS", safe_vs_etas: "Safe model / ETAS", fixed_vs_etas: "Fixed expert / ETAS", gated_vs_safe: "Gated model / safe" })[name] || name; }
function cell(content) { const td = document.createElement("td"); if (content instanceof Node) td.appendChild(content); else td.textContent = content; return td; }
function gainClass(value) { return value > 0 ? "gain-positive" : value < 0 ? "gain-negative" : "gain-neutral"; }
function regionName(id) { return state.dashboard.regions.find((region) => region.region_id === id)?.name || id; }
function shortRegion(name) { return name.includes("California") ? "California" : name.includes("Zealand") ? "New Zealand" : name.includes("Japan") ? "Japan C" : "Chile"; }
function depthLabel(region) { const min = region.minimum_depth_km ?? 0; return region.maximum_depth_km === null ? `${min}+ km depth` : `${min}–${region.maximum_depth_km} km depth`; }
function formatGain(value) { return value === null || value === undefined ? "—" : formatSigned(value, 4); }
function formatP(value) { return value === null || value === undefined ? "—" : Number(value).toFixed(3); }
function formatSigned(value, digits = 3) { return `${value > 0 ? "+" : ""}${Number(value).toFixed(digits)}`; }
function formatFactor(value) { return `${Number(value).toFixed(4)}×`; }
function formatInteger(value) { return new Intl.NumberFormat("en-US").format(value); }
function formatDate(value) { return new Intl.DateTimeFormat("en-US", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(value)); }
function formatShortDate(value) { return new Intl.DateTimeFormat("en-US", { day: "2-digit", month: "short", timeZone: "UTC" }).format(new Date(value)); }
function formatDateTime(value) { return new Intl.DateTimeFormat("en-US", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC", hour12: false }).format(new Date(value)); }
function setText(id, value) { document.getElementById(id).textContent = value; }
function showError(message) { const banner = document.getElementById("error-banner"); banner.textContent = message; banner.classList.add("visible"); }
function hideError() { document.getElementById("error-banner").classList.remove("visible"); }

document.querySelectorAll(".tab[data-view]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => { const active = item === button; item.classList.toggle("active", active); item.setAttribute("aria-selected", String(active)); });
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.getElementById(`${button.dataset.view}-view`).classList.add("active");
  if (button.dataset.view === "overview") requestAnimationFrame(resizeCanvas);
  if (button.dataset.view === "maps") {
    if (state.mapData && state.mapData.region_id === state.mapRegion) requestAnimationFrame(resizeForecastMap);
    else loadForecastMap();
  }
}));
document.querySelectorAll("[data-map-layer]").forEach((button) => button.addEventListener("click", () => {
  state.mapLayer = button.dataset.mapLayer;
  document.querySelectorAll("[data-map-layer]").forEach((item) => item.classList.toggle("active", item === button));
  renderForecastMap();
}));
document.getElementById("refresh-button").addEventListener("click", async () => {
  await loadDashboard();
  loadNewsletterStatus();
  if (state.mapData) loadForecastMap(true);
});
document.getElementById("newsletter-form").addEventListener("submit", submitNewsletter);
canvas.addEventListener("pointermove", handleChartPointer);
canvas.addEventListener("pointerleave", () => tooltip.classList.remove("visible"));
mapCanvas.addEventListener("pointermove", handleMapPointer);
mapCanvas.addEventListener("pointerleave", () => mapTooltip.classList.remove("visible"));
new ResizeObserver(resizeCanvas).observe(stage);
new ResizeObserver(resizeForecastMap).observe(mapStage);
loadDashboard();
loadNewsletterStatus();
