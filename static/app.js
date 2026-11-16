/* FraudRadar — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const C = { acid: "#c6ff3d", coral: "#ff3d6e", amber: "#ffd23d", orange: "#ff8a3d" };
  const RISK_COLOR = { low: C.acid, elevated: C.amber, high: C.orange, critical: C.coral };
  const RISKS = ["low", "elevated", "high", "critical"];
  const money = (v, d = 2) => "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (p, d = 1) => (p * 100).toFixed(d) + "%";
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(url, opts) {
    const res = await fetch(url, opts);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
    return body;
  }

  function countUp(el, to, fmt = v => Math.round(v).toLocaleString(), ms = 700) {
    const from = Number(el.dataset.v || 0);
    el.dataset.v = to;
    const t0 = performance.now();
    const step = t => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ================= routing ================= */
  const VIEWS = { investigate: "Check a transaction", batch: "Batch scanner", model: "Model insights" };
  function route() {
    const v = (location.hash || "#investigate").slice(1);
    const view = VIEWS[v] ? v : "investigate";
    $$(".view").forEach(s => s.classList.toggle("active", s.id === "view-" + view));
    $$(".side-nav a").forEach(a => a.classList.toggle("active", a.dataset.view === view));
    $("#viewTitle").textContent = VIEWS[view];
    $("#sidebar").classList.remove("open");
    window.scrollTo({ top: 0 });
    if (view === "model") animateImportance();
  }
  window.addEventListener("hashchange", route);
  $("#menuBtn").addEventListener("click", e => { e.stopPropagation(); $("#sidebar").classList.toggle("open"); });
  document.addEventListener("click", e => { if (!e.target.closest("#sidebar")) $("#sidebar").classList.remove("open"); });

  /* ================= model info ================= */
  api("/api/meta").then(m => {
    $("#mcAlgo").textContent = m.algorithm.replace("Classifier", "");
    $("#mcTrees").textContent = m.n_estimators;
    const max = m.top_features[0]?.importance || 1;
    $("#importance").innerHTML = m.top_features.map(f =>
      `<div class="imp-row"><span>${esc(f.feature)}</span><div class="track"><i data-w="${(f.importance / max) * 100}"></i></div><span>${pct(f.importance)}</span></div>`).join("");
    if (location.hash === "#model") animateImportance();
  }).catch(() => {});
  function animateImportance() { $$("#importance i").forEach((i, k) => setTimeout(() => (i.style.width = i.dataset.w + "%"), 80 * k)); }

  api("/api/overview").then(o => {
    if (!o.rows) return;
    $("#dRows").textContent = o.rows.toLocaleString();
    $("#dFraud").textContent = o.fraud.toLocaleString();
    $("#dRate").textContent = pct(o.fraud_rate, 2);
    $("#dAvg").textContent = money(o.avg_fraud_amount, 0);
  }).catch(() => {});

  /* ================= check a transaction ================= */
  const fp = $("#fingerprint");
  fp.innerHTML = Array.from({ length: 28 }, (_, i) => `<div class="fp-bar" data-tip="V${i + 1}"><i></i></div>`).join("");

  const cc = $("#cc");
  function showCase(d) {
    const t = d.transaction, p = d.probability, flag = d.prediction === 1;
    cc.classList.remove("swipe", "flagged", "cleared"); void cc.offsetWidth; cc.classList.add("swipe");
    $("#ccNumber").textContent = `4${Math.floor(100 + Math.random() * 899)} •••• •••• ${1000 + Math.floor(Math.random() * 9000)}`;
    $("#ccAmount").textContent = money(t.Amount);
    $("#ccCase").textContent = "#" + Math.random().toString(16).slice(2, 7).toUpperCase();
    setTimeout(() => cc.classList.add(flag ? "flagged" : "cleared"), 700);

    const vals = Array.from({ length: 28 }, (_, i) => t["V" + (i + 1)]);
    const max = Math.max(4, ...vals.map(Math.abs));
    $$(".fp-bar", fp).forEach((bar, i) => {
      const v = vals[i], el = $("i", bar);
      bar.dataset.tip = `V${i + 1}: ${v.toFixed(3)}`;
      el.className = v >= 0 ? "pos" : "neg";
      el.style.height = "2px";
      setTimeout(() => (el.style.height = Math.max(2, (Math.abs(v) / max) * 50) + "%"), 120 + i * 22);
    });
    $("#sAmount").textContent = money(t.Amount);
    $("#sTime").textContent = Math.round(t.Time).toLocaleString();
    $("#sTruth").textContent = d.actual === null || d.actual === undefined ? "Unknown" : d.actual ? "Fraud" : "Genuine";
    $("#sTruth").style.color = d.actual ? C.coral : C.acid;
    $("#sModel").textContent = flag ? "Fraud" : "Genuine";
    $("#sModel").style.color = flag ? C.coral : C.acid;

    $("#gFill").style.strokeDashoffset = 283 * (1 - p);
    $("#gNeedle").style.transform = `rotate(${-90 + p * 180}deg)`;
    countUp($("#gPct"), p * 100, v => v.toFixed(1), 1000);
    const st = $("#stamp");
    st.className = "stamp"; void st.offsetWidth;
    st.textContent = flag ? "FRAUD" : "GENUINE";
    st.classList.add("show", flag ? "flagged" : "cleared");
    $$(".risk-scale span").forEach(s => s.classList.toggle("on", s.dataset.r === d.risk));
    let note = flag
      ? `${Math.round(p * 150)} of 150 trees voted fraud.`
      : `Only ${Math.round(p * 150)} of 150 trees voted fraud.`;
    if (d.actual !== null && d.actual !== undefined) {
      note += d.actual === d.prediction ? " ✓ The model got it right." : d.actual ? " ✗ Missed — this one was really fraud." : " ✗ False alarm — it was genuine.";
    }
    $("#verdictNote").textContent = note;
  }

  $$(".pull-row button").forEach(b => b.addEventListener("click", async () => {
    $$(".pull-row button").forEach(x => (x.disabled = true));
    try { showCase(await api(`/api/sample?kind=${b.dataset.kind}`)); }
    catch (err) { $("#verdictNote").textContent = err.message; }
    finally { setTimeout(() => $$(".pull-row button").forEach(x => (x.disabled = false)), 700); }
  }));

  // 3D tilt + holographic sheen
  const stage = $(".card-stage");
  stage.addEventListener("pointermove", e => {
    const r = stage.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
    cc.style.transform = `rotateY(${x * 24}deg) rotateX(${-y * 20}deg)`;
    $(".cc-holo", cc).style.setProperty("--holo", `${(x + y) * 220}deg`);
  });
  stage.addEventListener("pointerleave", () => (cc.style.transform = ""));

  /* ================= batch ================= */
  const drop = $("#dropzone"), input = $("#fileInput"), status = $("#scanStatus");
  let lastRows = [], batchFilter = "all";
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) scan(f); });
  input.addEventListener("change", () => { if (input.files[0]) scan(input.files[0]); input.value = ""; });
  $("#useSample").addEventListener("click", async () => {
    try { const blob = await (await fetch("/download/sample.csv")).blob(); scan(new File([blob], "sample_transactions.csv")); }
    catch { status.textContent = "Could not load the sample file."; status.className = "scan-status err"; }
  });

  async function scan(file) {
    status.className = "scan-status"; status.textContent = `Scanning ${file.name}…`;
    drop.classList.add("scanning");
    const fd = new FormData(); fd.append("file", file);
    const t0 = performance.now();
    try {
      const [d] = await Promise.all([api("/api/predict-csv", { method: "POST", body: fd }), new Promise(r => setTimeout(r, 1000))]);
      status.textContent = `✓ ${d.summary.total.toLocaleString()} transactions scored in ${((performance.now() - t0) / 1000).toFixed(1)}s`;
      renderBatch(d);
    } catch (err) {
      status.className = "scan-status err"; status.textContent = "✗ " + err.message;
    } finally { drop.classList.remove("scanning"); }
  }

  function scoreCell(p, risk) {
    return `<div class="score"><div class="bar"><i style="width:${Math.max(2, p * 100)}%;background:${RISK_COLOR[risk]}"></i></div><span style="color:${RISK_COLOR[risk]}">${pct(p)}</span></div>`;
  }

  function renderBatch(d) {
    const s = d.summary;
    $("#results").hidden = false;
    countUp($("#rTotal"), s.total); countUp($("#rGenuine"), s.genuine); countUp($("#rFraud"), s.fraud);
    countUp($("#rAmount"), s.fraud_amount, v => money(v));
    const lab = $("#labelled");
    if (s.labelled) {
      lab.hidden = false;
      lab.innerHTML = `<span>Labels found — checking the model against the truth:</span>
        <span>accuracy <b>${pct(s.labelled.accuracy)}</b></span>
        <span>fraud caught <b>${s.labelled.caught} / ${s.labelled.actual_fraud}</b></span>
        <span>false alarms <b>${s.labelled.false_alarms}</b></span>`;
    } else lab.hidden = true;
    const spec = $("#spectrum");
    spec.innerHTML = RISKS.map(k => `<div class="${k}" title="${k}: ${s.risk_buckets[k]}">${s.risk_buckets[k] ? k.toUpperCase() + " · " + s.risk_buckets[k] : ""}</div>`).join("");
    requestAnimationFrame(() => $$("div", spec).forEach((el, i) => {
      const v = s.risk_buckets[RISKS[i]]; el.style.width = v ? Math.max(8, (v / s.total) * 100) + "%" : "0";
    }));
    lastRows = d.rows;
    $("#feedNote").textContent = d.truncated ? `Showing the ${d.rows.length.toLocaleString()} riskiest of ${s.total.toLocaleString()}` : "Most suspicious first";
    renderBatchRows();
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderBatchRows() {
    const rows = batchFilter === "flagged" ? lastRows.filter(r => r.prediction === 1) : lastRows;
    $("#batchFeed").innerHTML = rows.length ? rows.slice(0, 400).map((r, i) => {
      let truth = "";
      if (r.actual !== null) truth = r.actual === r.prediction ? '<span class="truth">✓</span>' : `<span class="truth miss">✗ was ${r.actual ? "fraud" : "genuine"}</span>`;
      return `<div class="tr${r.prediction ? " flag" : ""}" style="animation-delay:${Math.min(i, 30) * 25}ms"><span class="muted-id">#${r.index}</span>
        <span>${Math.round(r.time).toLocaleString()}s</span><span class="amt">${money(r.amount)}</span>
        ${scoreCell(r.probability, r.risk)}<span>${r.prediction ? '<span class="tag fraud">FRAUD</span>' : '<span class="tag genuine">GENUINE</span>'}${truth}</span></div>`;
    }).join("") : '<div class="empty" style="margin:1rem">No fraud found in this file.</div>';
  }
  $$("#results .chip").forEach(c => c.addEventListener("click", () => {
    $$("#results .chip").forEach(x => x.classList.toggle("active", x === c));
    batchFilter = c.dataset.filter; renderBatchRows();
  }));
  $("#downloadResults").addEventListener("click", () => {
    const head = "Row,Time,Amount,Fraud_Probability,Risk,Prediction,Actual\n";
    const body = lastRows.map(r => [r.index, r.time, r.amount, r.probability.toFixed(4), r.risk, r.prediction ? "FRAUD" : "GENUINE", r.actual ?? ""].join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([head + body], { type: "text/csv" }));
    a.download = "fraudradar_results.csv"; a.click(); URL.revokeObjectURL(a.href);
  });

  route();
})();
