/* zero-touch-fl control panel.
 *
 * One page, four views (training / versions / benchmark / run history) over the
 * stdlib server in webui/server.py. Two independent long-poll loops -- one per
 * runner -- feed the training and benchmark panels, so a benchmark started while
 * training is finishing does not interleave into the wrong charts.
 *
 * The page never computes anything the CLIs already computed. Training metrics are
 * the round records main.py writes to logs/round_data/rounds.jsonl, republished
 * verbatim; benchmark metrics are the per-round cells benchmark/metrics.py
 * accumulates. This view renders them, it does not re-derive them.
 */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const C = window.Charts;

  const state = {
    boot: null,
    overrides: {},              // dotted config path -> raw string/bool from the form
    train: { since: 0, status: null, rounds: [], charts: {}, p1: [], started: 0, timer: null },
    bench: {
      since: 0, status: null, cfg: null, agg: {}, rounds: [], charts: {},
      focusAttack: null, focusDefense: null, lastRound: null, summary: null,
      started: 0, timer: null,
      // A sweep over several fine-tuned versions: `queue` is where we are in it,
      // `version` is the leg running now, `byVersion` is every finished leg.
      queue: null, version: null, byVersion: [],
    },
  };

  /* ------------------------------------------------------------------ */
  /* Small helpers                                                       */
  /* ------------------------------------------------------------------ */
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else node.setAttribute(k, v);
    }
    (children || []).forEach((c) => c && node.appendChild(
      typeof c === "string" ? document.createTextNode(c) : c));
    return node;
  }

  function toast(message, kind) {
    const node = el("div", { class: "toast " + (kind || ""), text: message });
    $("#toasts").appendChild(node);
    setTimeout(() => {
      node.style.opacity = "0";
      node.style.transition = "opacity .3s";
      setTimeout(() => node.remove(), 320);
    }, kind === "error" ? 9000 : 4200);
  }

  async function api(path, body, method) {
    const opts = { method: method || (body ? "POST" : "GET"),
                   headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    let payload = null;
    try { payload = await res.json(); } catch (e) { payload = null; }
    if (!res.ok) throw new Error((payload && payload.error) || res.statusText);
    return payload;
  }

  const pct = (v, d) => (v === null || v === undefined || Number.isNaN(v))
    ? "--" : (v * 100).toFixed(d === undefined ? 1 : d) + "%";
  const num = (v, d) => (v === null || v === undefined || Number.isNaN(v))
    ? "--" : (+v).toFixed(d === undefined ? 3 : d);
  const signed = (v, d) => (v === null || v === undefined || Number.isNaN(v))
    ? "--" : (v >= 0 ? "+" : "") + (+v).toFixed(d === undefined ? 4 : d);

  function bytes(n) {
    if (!n) return "0 B";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 1 : 0) + " " + u[i];
  }

  function since(ts) {
    if (!ts) return "";
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    const m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h) return h + "h " + (m % 60) + "m";
    if (m) return m + "m " + (s % 60) + "s";
    return s + "s";
  }

  function mean(xs) {
    const v = xs.filter((x) => typeof x === "number" && !Number.isNaN(x));
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }

  function kpi(container, items) {
    container.innerHTML = "";
    items.forEach((it) => {
      if (!it) return;
      container.appendChild(el("div", { class: "kpi " + (it.tone || "") }, [
        el("div", { class: "k", text: it.k, title: it.title || it.k }),
        el("div", { class: "v", text: it.v }),
        it.s ? el("div", { class: "s", text: it.s }) : null,
      ]));
    });
  }

  /* ------------------------------------------------------------------ */
  /* Navigation                                                          */
  /* ------------------------------------------------------------------ */
  function showView(name) {
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
    $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    location.hash = name;
    $(".main").scrollTop = 0;   // the scroll container is .main, not the window
    // Canvases sized while hidden come out 0x0; redraw whatever this view owns.
    setTimeout(() => {
      Object.values(state.train.charts).forEach((c) => c.draw && c.draw());
      Object.values(state.bench.charts).forEach((c) => c.draw && c.draw());
    }, 20);
    if (name === "runs") loadRuns();
  }

  /* ------------------------------------------------------------------ */
  /* Config editor -- generated from configs/base.yaml itself             */
  /* ------------------------------------------------------------------ */
  function renderConfig() {
    const wrap = $("#cfg-fields");
    const { groups, fields } = state.boot.config;
    const filter = ($("#cfg-search").value || "").trim().toLowerCase();
    const primaryOnly = $("#cfg-primary-only").checked;
    wrap.innerHTML = "";
    let shown = 0;

    groups.forEach((group) => {
      const paths = group.paths.filter((p) => {
        const f = fields[p];
        if (f.frozen) return false;
        if (primaryOnly && !f.primary && !(p in state.overrides)) return false;
        if (!filter) return true;
        return p.toLowerCase().includes(filter) ||
               (f.doc || "").toLowerCase().includes(filter);
      });
      if (!paths.length) return;
      shown += paths.length;
      const body = el("div", { class: "body" }, [
        el("div", { class: "fields" }, paths.map((p) => configField(fields[p]))),
      ]);
      const changed = paths.filter((p) => p in state.overrides).length;
      const details = el("details", { class: "group", open: filter || changed ? "" : null }, [
        el("summary", {}, [
          document.createTextNode(group.name),
          el("span", { class: "count", text: `${paths.length} setting${paths.length === 1 ? "" : "s"}` +
            (changed ? ` · ${changed} changed` : "") }),
        ]),
        body,
      ]);
      wrap.appendChild(details);
    });

    if (!shown) {
      wrap.appendChild(el("div", { class: "empty-state" }, [
        el("h3", { text: "Nothing matches" }),
        el("div", { text: "Clear the filter, or untick “Essentials only” to search every key in base.yaml." }),
      ]));
    }
    const n = Object.keys(state.overrides).length;
    $("#train-config-count").textContent = n ? `${n} override${n === 1 ? "" : "s"}` : "base.yaml as shipped";
  }

  function configField(f) {
    const current = f.path in state.overrides ? state.overrides[f.path] : f.value;
    const wrap = el("div", { class: "field" + (f.path in state.overrides ? " changed" : "") });
    wrap.appendChild(el("label", {}, [
      document.createTextNode(f.path.split(".").pop()),
      el("span", { class: "path", text: f.path }),
    ]));

    let input;
    const commit = (value) => {
      const same = String(value) === String(f.value) ||
        (Array.isArray(f.value) && String(value) === f.value.join(","));
      if (same) delete state.overrides[f.path];
      else state.overrides[f.path] = value;
      wrap.classList.toggle("changed", f.path in state.overrides);
      if (f.path === "defense.mode") renderLearnWarning();
      const n = Object.keys(state.overrides).length;
      $("#train-config-count").textContent = n ? `${n} override${n === 1 ? "" : "s"}` : "base.yaml as shipped";
    };

    if (f.type === "bool") {
      input = el("input", { type: "checkbox" });
      input.checked = current === true || current === "true";
      input.addEventListener("change", () => commit(input.checked));
      const sw = el("label", { class: "switch" }, [input,
        el("span", { class: "hint", text: input.checked ? "on" : "off" })]);
      input.addEventListener("change", () => {
        sw.lastChild.textContent = input.checked ? "on" : "off";
      });
      wrap.appendChild(sw);
    } else if (f.enum) {
      input = el("select");
      f.enum.forEach((o) => input.appendChild(el("option", { value: o, text: o })));
      input.value = String(current);
      input.addEventListener("change", () => commit(input.value));
      wrap.appendChild(input);
    } else {
      const isNum = f.type === "int" || f.type === "float";
      input = el("input", {
        type: isNum ? "number" : "text",
        step: f.type === "float" ? "any" : (f.type === "int" ? "1" : null),
        class: isNum || f.type === "list" ? "mono" : null,
        placeholder: f.value === null ? "null (auto)" : String(f.value),
      });
      input.value = Array.isArray(current) ? current.join(", ")
        : (current === null ? "" : String(current));
      input.addEventListener("change", () => commit(input.value));
      wrap.appendChild(input);
    }
    if (f.doc) wrap.appendChild(el("div", { class: "doc", text: f.doc }));
    return wrap;
  }

  /* ------------------------------------------------------------------ */
  /* Training charts                                                     */
  /* ------------------------------------------------------------------ */
  function legend(container, series) {
    const node = $(container);
    if (!node) return;
    node.innerHTML = "";
    series.forEach((s) => node.appendChild(el("span", {}, [
      el("i", { style: { background: s.color } }), document.createTextNode(s.label),
    ])));
  }

  function buildTrainCharts() {
    const P = C.PALETTE();
    const T = state.train.charts;

    T.p1 = new C.LineChart($("#chart-p1"), {
      yFormat: (v) => (v * 100).toFixed(1) + "%",
      series: [{ key: "acc", label: "global accuracy", color: P.good, width: 2 }],
    });

    const accSeries = [
      { key: "post", label: "global accuracy", color: P.attack, width: 2 },
    ];
    T.acc = new C.LineChart($("#chart-acc"), {
      series: accSeries, yFormat: (v) => (v * 100).toFixed(1) + "%",
    });
    legend("#legend-acc", accSeries);

    const rewardSeries = [
      { key: "total", label: "total", color: P.text, width: 2 },
      { key: "damage", label: "damage", color: P.attack, width: 1.5 },
      { key: "stealth", label: "stealth", color: P.defense, width: 1.5 },
      { key: "malformed", label: "malformed", color: P.warn, width: 1.2 },
      { key: "collab", label: "collab", color: P.accent, width: 1.2 },
    ];
    T.reward = new C.LineChart($("#chart-reward"), {
      series: rewardSeries, zeroLine: true, yFormat: (v) => v.toFixed(2),
    });
    legend("#legend-reward", rewardSeries);

    // The DEFENDER's score sheet. Its reward is logged every round (RoundLog
    // .defender_reward) but only becomes the thing being optimised under
    // `--learn defender`, where every attacker series above is a frozen policy's
    // flat line and this is the one that moves.
    const defRewardSeries = [
      { key: "reward", label: "defender reward", color: P.defense, width: 2 },
      { key: "tpr", label: "TPR (attack caught)", color: P.good, width: 1.4 },
      { key: "fpr", label: "FPR (honest rejected)", color: P.warn, width: 1.4, dash: [3, 3] },
    ];
    T.defReward = new C.LineChart($("#chart-def-reward"), {
      series: defRewardSeries, yMin: 0, yMax: 1, yFormat: (v) => v.toFixed(2),
    });
    legend("#legend-def-reward", defRewardSeries);

    const grpoSeries = [
      { key: "loss", label: "GRPO loss", color: P.accent, width: 1.8 },
      { key: "mean_reward", label: "mean reward", color: P.good, width: 1.4 },
      { key: "spread", label: "reward spread", color: P.warn, width: 1.4 },
      { key: "zero_adv", label: "zero-advantage frac", color: P.faint, width: 1.2, dash: [3, 3] },
    ];
    T.grpo = new C.LineChart($("#chart-grpo"), {
      series: grpoSeries, zeroLine: true, yFormat: (v) => v.toFixed(2),
    });
    legend("#legend-grpo", grpoSeries);

    const detectSeries = [
      { key: "tpr", label: "TPR (attack caught)", color: P.defense, width: 1.8 },
      { key: "fpr", label: "FPR (honest rejected)", color: P.warn, width: 1.5 },
    ];
    T.detect = new C.LineChart($("#chart-detect"), {
      series: detectSeries, yMin: 0, yMax: 1, yFormat: (v) => (v * 100).toFixed(0) + "%",
    });
    legend("#legend-detect", detectSeries);

  }

  function resetTrainCharts() {
    Object.values(state.train.charts).forEach((c) => { c.reset(); c.draw(); });
    state.train.rounds = [];
    state.train.p1 = [];
    $("#train-federation").innerHTML = "";
    $("#fed-caption").textContent = "no round yet";
    $("#p1-status").textContent = "not started";
    kpi($("#train-kpis"), []);
  }

  /* One round record from logs/round_data/rounds.jsonl. */
  function onTrainRound(record) {
    const m = record.attack_metadata || {};
    const train = m.train || {};
    const terms = m.reward_terms || {};
    const verdicts = record.predicted_labels || [];
    const poisoned = new Set(record.poisoned_client_ids || []);

    let tp = 0, fn = 0, fp = 0, tn = 0;
    verdicts.forEach((v) => {
      const bad = poisoned.has(v.client_id);
      if (bad && v.is_suspicious) tp++;
      else if (bad) fn++;
      else if (v.is_suspicious) fp++;
      else tn++;
    });
    const row = {
      round: record.round_num, record, tp, fn, fp, tn,
      tpr: tp + fn ? tp / (tp + fn) : 0,
      fpr: fp + tn ? fp / (fp + tn) : 0,
      drop: m.induced_drop, measured: m.clean_measured !== false && m.defense_sane !== false,
      win: !!m.learner_success,
    };
    state.train.rounds.push(row);
    if (state.train.rounds.length > 5000) state.train.rounds.shift();

    const x = record.round_num;
    const T = state.train.charts;
    T.acc.push(x, { post: record.test_accuracy });
    T.reward.push(x, { total: record.attacker_reward, damage: terms.damage,
                       stealth: terms.stealth, malformed: terms.malformed,
                       collab: terms.collab });
    T.grpo.push(x, { loss: train.loss, mean_reward: train.mean_reward,
                     spread: train.reward_spread, zero_adv: train.zero_advantage_fraction });
    T.detect.push(x, { tpr: row.tpr, fpr: row.fpr });
    T.defReward.push(x, { reward: record.defender_reward, tpr: row.tpr, fpr: row.fpr });
    [T.acc, T.reward, T.defReward, T.grpo, T.detect].forEach((c) => c.draw());

    renderTrainKpis();
    renderFederation($("#train-federation"), {
      n: (state.train.status && state.train.status.n_clients) || verdicts.length,
      poisoned: record.poisoned_client_ids || [],
      pool: m.controllable_pool || [],
      flagged: verdicts.filter((v) => v.is_suspicious).map((v) => v.client_id),
      confidence: Object.fromEntries(verdicts.map((v) => [v.client_id, v])),
    });
    const cur = m.curriculum;
    $("#fed-caption").textContent =
      `round ${record.round_num} · defense ${m.defense || "llm"}` +
      (cur ? ` · block ${cur.block} (${cur.algorithm} × ${cur.n_poisoners} poisoners), round ${cur.block_round}` : "") +
      ` · ${tp} caught / ${fn} through / ${fp} false alarms`;
  }

  /* --learn defender|both needs a trainable defender, and the shipped config
     defends with published algorithms -- so the most obvious first click in this
     panel names a side that cannot learn. The server refuses it (reusing the
     CLI's own resolver), but being told to change a setting that lives three
     fields away in the SAME panel is a poor way to find out, so the requirement
     is stated next to the selector with the one-click fix. */
  function effectiveDefenseMode() {
    if ("defense.mode" in state.overrides) return String(state.overrides["defense.mode"]);
    const f = state.boot && state.boot.config.fields["defense.mode"];
    return f ? String(f.value) : "algorithmic";
  }

  function renderLearnWarning() {
    const learn = $("#train-learn").value;
    const needsLLM = learn === "defender" || learn === "both";
    const mode = effectiveDefenseMode();
    const bad = needsLLM && mode !== "llm";
    $("#train-learn-warn").classList.toggle("hidden", !bad);
    if (!bad) return;
    $("#train-learn-warn-text").textContent =
      `defense.mode is "${mode}", so the server defends with published algorithms ` +
      `— they have nothing to learn and there is no defender policy to train. `;
  }

  /* The KPI strip, ordered by WHO IS LEARNING.

     The same round record describes both sides, but only one of them is being
     optimised: under `--learn defender` every attacker series is a frozen
     policy's flat line and the defender's reward is the number that moves. A
     strip that always led with the attacker's numbers therefore reported a
     defender run as "nothing is happening". So the learner's block comes first
     and the opponent's follows, labelled as frozen. */
  function renderTrainKpis() {
    const rows = state.train.rounds;
    if (!rows.length) return;
    const last = rows[rows.length - 1];
    const r = last.record, m = r.attack_metadata || {}, t = m.train || {};
    const terms = m.reward_terms || {};
    const recent = rows.slice(-50);
    const measured = recent.filter((x) => x.measured);
    const winRate = recent.length ? recent.filter((x) => x.win).length / recent.length : null;
    // Only the GRPO path breaks the reward into terms; --baseline and --dry-run
    // report a scalar, so the reward's dmg/stl split is not always available.
    const hasTerms = typeof terms.damage === "number";
    const cur = m.curriculum;

    // Which side this run is training. The round log is authoritative; before the
    // first round arrives the run's own --learn flag stands in.
    const learner = r.learning_agent ||
      (state.train.status && state.train.status.learn) || "none";
    const forDefender = learner === "defender";

    const context = [
      // The FL round index, which keeps climbing across resumes. It is NOT the
      // --rounds budget: that counts Phase-2 rounds DONE (across every run), so
      // pairing them as "53 of 8" would be two different counters in one reading.
      { k: "FL round", v: r.round_num,
        title: "the round counter in logs/round_data/rounds.jsonl; it continues across resumes",
        s: state.train.rounds.length + " round" +
           (state.train.rounds.length === 1 ? "" : "s") + " this run" },
      { k: "accuracy", v: pct(r.test_accuracy, 2), tone: "def",
        s: "clean " + pct(m.clean_accuracy, 2) },
      { k: "learner", v: learner, tone: forDefender ? "def" : "atk",
        s: m.phase_index === undefined ? "no GRPO schedule"
                                       : `phase ${m.phase_index}/${m.phase_round}` },
    ];

    const win = {
      k: (forDefender ? "defender" : "attacker") + " win rate",
      v: winRate === null ? "--" : pct(winRate, 0),
      tone: winRate > 0 ? "good" : "", s: "last " + recent.length + " rounds",
      title: "rounds the LEARNING side cleared its win gate on (learner_success)",
    };

    const defenderBlock = [
      { k: "defender reward", v: num(r.defender_reward, 3), tone: "def",
        s: "mean " + num(mean(recent.map((x) => x.record.defender_reward)), 3),
        title: "soft-F1 over the per-client verdicts (rl.reward.defender.mode)" },
      { k: "detection", v: pct(last.tpr, 0), tone: "def",
        s: "FPR " + pct(last.fpr, 0),
        title: "how much of the round's attack the defense caught, and how much of "
             + "the honest federation it rejected" },
      { k: "mean detection", v: pct(mean(recent.map((x) => x.tpr)), 0),
        s: "mean FPR " + pct(mean(recent.map((x) => x.fpr)), 0) },
    ];

    const attackerBlock = [
      { k: "attacker reward", v: num(r.attacker_reward, 3), tone: "atk",
        s: hasTerms ? `dmg ${signed(terms.damage, 2)} stl ${signed(terms.stealth, 2)}`
                    : "not broken out in this mode" },
      { k: "mean drop", v: signed(mean(measured.map((x) => x.drop)), 4),
        s: measured.length + " measured rounds",
        title: "clean counterfactual minus the post-attack accuracy, over the "
             + "rounds where the counterfactual was actually measured" },
    ];

    // The frozen side is still worth reading -- it is what the learner is scored
    // against -- so it is kept, just marked as not being optimised.
    const frozen = forDefender ? attackerBlock : defenderBlock;
    frozen.forEach((k) => {
      k.s = (k.s ? k.s + " · " : "") + "frozen";
    });

    kpi($("#train-kpis"), context
      .concat(win)
      .concat(forDefender ? defenderBlock : attackerBlock)
      .concat([
        { k: "GRPO loss", v: num(t.loss, 4),
          s: t.stepped === false ? "step SKIPPED" : "spread " + num(t.reward_spread, 3),
          tone: t.stepped === false ? "bad" : "" },
        { k: "defense", v: m.defense || "llm",
          s: cur ? `blk ${cur.block}/${cur.block_round} · ${cur.n_poisoners}p` : "" },
      ])
      .concat(frozen));
  }

  /* ------------------------------------------------------------------ */
  /* Federation grid -- the client matrix, shared by both views           */
  /* ------------------------------------------------------------------ */
  function renderFederation(container, opts) {
    const n = opts.n || 0;
    const poisoned = new Set(opts.poisoned || []);
    const pool = new Set(opts.pool || []);
    const flagged = new Set(opts.flagged || []);
    const shifts = opts.shifts || {};
    container.innerHTML = "";
    for (let i = 0; i < n; i++) {
      const bad = poisoned.has(i), hit = flagged.has(i);
      const cls = bad ? (hit ? "tp" : "fn") : (hit ? "fp" : "tn");
      const bits = [];
      if (bad) bits.push("poisoned");
      if (hit) bits.push("flagged by the defense");
      if (pool.has(i)) bits.push("in the attacker's pool");
      const v = (opts.confidence || {})[i];
      if (v) bits.push("verdict confidence " + num(v.confidence, 2) +
                       (v.reason ? " — " + v.reason : ""));
      if (shifts[i] !== undefined) bits.push("‖poison‖ = " + num(shifts[i], 4));
      container.appendChild(el("div", {
        class: "node " + cls + (pool.has(i) ? " pool" : ""),
        title: "client " + i + (bits.length ? "\n" + bits.join("\n") : ""),
      }, [document.createTextNode(String(i)),
          bad ? el("span", { class: "mark", text: "☠" }) : null]));
    }
  }

  /* ------------------------------------------------------------------ */
  /* Console                                                             */
  /* ------------------------------------------------------------------ */
  const MAX_LOG_LINES = 1200;
  function appendLog(which, line, level) {
    const node = $("#" + which + "-console");
    const empty = node.querySelector(".empty");
    if (empty) empty.remove();
    const follow = $("#" + which + "-autoscroll").checked;
    const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 40;
    node.appendChild(el("div", { class: level || "", text: line }));
    while (node.childElementCount > MAX_LOG_LINES) node.removeChild(node.firstChild);
    if (follow || atBottom) node.scrollTop = node.scrollHeight;
  }

  /* ------------------------------------------------------------------ */
  /* Run status plumbing                                                 */
  /* ------------------------------------------------------------------ */
  function applyStatus(which, status) {
    if (!status) return;
    const s = state[which];
    s.status = status;
    const running = status.state === "running" || status.state === "stopping";
    const pill = $("#" + which + "-pill");
    pill.className = "pill " + status.state;
    pill.lastChild.textContent = status.state;
    $("#" + which + "-start").disabled = running;
    $("#" + which + "-stop").disabled = !running;
    // The benchmark view has a second start button (the defender target); both
    // compete for the same runner, so a run in flight has to lock out both.
    const alt = $("#" + which + "-start-defender");
    if (alt) alt.disabled = running;
    $$("#nav button").forEach((b) => {
      if (b.dataset.view === (which === "train" ? "train" : "bench")) {
        b.classList.toggle("busy", running);
      }
    });
    if (status.command) $("#" + which + "-cmd").textContent = status.command;
    if (running && !s.timer) {
      s.started = status.started || Date.now() / 1000;
      s.timer = setInterval(() => {
        $("#" + which + "-elapsed").textContent = since(s.started);
      }, 1000);
    } else if (!running && s.timer) {
      clearInterval(s.timer);
      s.timer = null;
      if (status.started && status.ended) {
        $("#" + which + "-elapsed").textContent = since(status.started) + " total";
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Event loops                                                         */
  /* ------------------------------------------------------------------ */
  async function poll(which) {
    const kind = which === "train" ? "train" : "bench";
    for (;;) {
      try {
        const data = await api(`/api/events?kind=${kind}&since=${state[which].since}`);
        state[which].since = data.seq;
        applyStatus(which, data.status);
        (data.events || []).forEach((ev) =>
          which === "train" ? handleTrainEvent(ev) : handleBenchEvent(ev));
      } catch (e) {
        await new Promise((r) => setTimeout(r, 2500));   // server restart / sleep
      }
    }
  }

  function handleTrainEvent(ev) {
    switch (ev.event) {
      case "run_started":
        resetTrainCharts();
        $("#train-console").innerHTML = "";
        appendLog("train", "$ " + ev.command, "ev");
        if (ev.overrides && ev.overrides.length) {
          appendLog("train", "config overrides: " + ev.overrides.join("; "), "ev");
        }
        toast("Training started", "ok");
        break;
      case "log":
        appendLog("train", ev.line, ev.level);
        break;
      case "phase1_round":
        $("#train-phase1-panel").classList.remove("hidden");
        $("#p1-status").textContent = `round ${ev.round} of ${ev.total}`;
        break;
      case "phase1_accuracy":
        $("#train-phase1-panel").classList.remove("hidden");
        state.train.p1.push(ev.accuracy);
        state.train.charts.p1.push(ev.round, { acc: ev.accuracy });
        state.train.charts.p1.draw();
        $("#p1-status").textContent =
          `round ${ev.round}${ev.total ? " of " + ev.total : ""} · accuracy ${pct(ev.accuracy, 2)}`;
        break;
      case "phase":
        if (ev.phase === "phase2") {
          $("#p1-status").textContent = state.train.p1.length
            ? `done · baseline ${pct(state.train.p1[state.train.p1.length - 1], 2)}`
            : "skipped (loaded from checkpoint)";
        }
        break;
      case "train_round":
        onTrainRound(ev.record);
        break;
      case "run_stopping":
        appendLog("train", "-- stop requested --", "warn");
        break;
      case "run_ended":
        appendLog("train", `-- run ${ev.state} (exit ${ev.exit_code}) after ${ev.elapsed}s --`,
                  ev.state === "failed" ? "error" : "ev");
        toast(`Training ${ev.state}`, ev.state === "failed" ? "error" : "ok");
        refreshVersions();
        loadRuns();
        break;
      default:
        break;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Versions                                                            */
  /* ------------------------------------------------------------------ */
  async function refreshVersions() {
    try {
      const data = await api("/api/versions");
      state.boot.versions = data.versions;
      state.boot.live = data.live;
      renderVersions();
      renderVersionSelect();
    } catch (e) { /* the panel keeps whatever it had */ }
  }

  function renderVersions() {
    const live = state.boot.live || {};
    const adapters = live.adapters || {};
    const att = adapters.attacker || {};
    const def = adapters.defender || {};
    const tr = live.training || {};
    // Either adapter is worth snapshotting on its own: `rl/schedule.py` writes
    // ONLY the side --learn named, so a defender-only run leaves no attacker
    // adapter at all and gating the button on the attacker's made the one adapter
    // that run produced unversionable.
    const anyAdapter = att.exists || def.exists;
    const newest = [att, def].filter((a) => a.exists).map((a) => a.modified).sort().pop();
    $("#live-modified").textContent = anyAdapter
      ? "adapter last written " + newest : "no adapter on disk yet";
    const learners = tr.learners_seen || [];
    kpi($("#live-kpis"), [
      { k: "attacker adapter", v: att.exists ? "present" : "none",
        tone: att.exists ? "good" : "", s: att.exists ? bytes(att.size_bytes) : att.path },
      { k: "defender adapter", v: def.exists ? "present" : "none",
        tone: def.exists ? "good" : "",
        s: def.exists ? bytes(def.size_bytes)
                      : "written only under defense.mode: llm with --learn defender",
        title: "the llm_defender benchmark column loads this" },
      { k: "rounds done", v: (live.progress || {}).rounds_done ?? "--",
        s: "FL round " + ((live.progress || {}).round_index ?? "--") },
      { k: "trained side", v: learners.length ? learners.join(" + ") : "--",
        s: "over the last " + (tr.rounds || 0) + " rounds",
        title: "learning_agent in the round log — which policy those rounds updated" },
      { k: "attacker reward", v: num(tr.mean_attacker_reward, 3), tone: "atk",
        s: `mean drop ${signed(tr.mean_induced_drop, 4)}` },
      { k: "defender reward", v: num(tr.mean_defender_reward, 3), tone: "def",
        s: `TPR ${pct(tr.mean_tpr, 0)} · FPR ${pct(tr.mean_fpr, 0)}` },
      { k: "win rate", v: tr.win_rate === null || tr.win_rate === undefined
          ? "--" : pct(tr.win_rate, 0), tone: tr.win_rate > 0 ? "good" : "",
        s: "the learning side's gate" },
      { k: "defenses seen", v: (tr.defenses_seen || []).length || "--",
        s: (tr.defenses_seen || []).join(", ") },
    ]);
    $("#ver-save").disabled = !anyAdapter;
    $("#ver-save-hint").textContent = anyAdapter
      ? "copies whichever of checkpoints/attacker_adapter and " +
        "checkpoints/defender_adapter exist into checkpoints/versions/"
      : "train first — an adapter is written every rl.save_every rounds, for the " +
        "side --learn names";

    // Real snapshots newest-first, then the built-in demo fixture -- the server
    // appends it to the listing (see webui/demo.py), so the table, both benchmark
    // pickers and the run router all read one record.
    const versions = state.boot.versions || [];
    $("#ver-count").textContent = versions.length
      ? versions.length + " version" + (versions.length === 1 ? "" : "s") : "";
    $("#ver-empty").classList.toggle("hidden", versions.length > 0);
    const table = $("#ver-table");
    table.innerHTML = "";
    if (!versions.length) return;

    const head = ["version", "holds", "created", "rounds", "mean drop",
                  "base model", ""];
    table.appendChild(el("thead", {}, [el("tr", {}, head.map((h) =>
      el("th", { text: h })))]));
    const body = el("tbody");
    versions.forEach((v) => {
      const t = v.training || {};
      const roles = v.roles || Object.keys(v.adapters || {});
      body.appendChild(el("tr", {}, [
        el("td", {}, [
          el("div", { text: v.label, style: { fontWeight: "600" } }),
          el("div", { class: "hint",
                      text: (v.label === v.id ? "" : v.id) +
                            (v.notes ? (v.label === v.id ? "" : " · ") + v.notes : "") }),
        ]),
        // Which roles this version can be benchmarked as. A one-sided training run
        // snapshots one adapter, and that decides which panel rows it can fill.
        el("td", {}, [el("div", { class: "chips" }, roles.map((role) =>
          el("span", { class: "chip on" + (role === "attacker" ? " atk" : ""),
                       text: role.slice(0, 3),
                       title: role + "_adapter is in this version" })))]),
        el("td", { class: "num", text: (v.created || "").replace("T", " ") }),
        el("td", { class: "num", text: v.rounds_done ?? "--" }),
        el("td", { class: "num", text: signed(t.mean_induced_drop, 4) }),
        el("td", { class: "num", text: (v.base_model || "--").split("/").pop() }),
        el("td", {}, [
          el("button", { class: "btn sm", text: "Benchmark",
            onclick: () => {
              // A seed row has no adapter to point a benchmark at, so it only
              // navigates: pushing its id into the pickers would be dropped as
              // unselectable and would clear whatever was already picked.
              if (!v.demo) {
                // Select it on the axes it can actually fill, so the button never
                // lands the user on a panel the server would refuse.
                if (roles.includes("attacker")) {
                  versionSelection.clear();
                  versionSelection.add(v.id);
                }
                if (roles.includes("defender")) {
                  defenderSelection.clear();
                  defenderSelection.add(v.id);
                  benchSelection.defenses.add("llm_defender");
                  renderChips("#bench-defenses", state.boot.defenses.available,
                              benchSelection.defenses, DEFENSE_NOTE);
                }
                renderVersionSelect();
              }
              showView("bench");
            } }),
          document.createTextNode(" "),
          el("button", { class: "btn sm ghost", text: "Rename",
            onclick: () => renameVersion(v) }),
          document.createTextNode(" "),
          el("button", { class: "btn sm ghost", text: "Delete",
            onclick: () => deleteVersion(v) }),
        ]),
      ]));
    });
    table.appendChild(body);
  }

  async function renameVersion(v) {
    const label = prompt("Version name", v.label);
    if (label === null) return;
    const notes = prompt("Notes", v.notes || "");
    if (notes === null) return;
    if (v.demo) {
      // There is no version.json behind a seed row, so it is renamed in place:
      // the button behaves like the real one instead of erroring on an id the
      // store does not have. The edit lives until the page is reloaded.
      v.label = label.trim() || v.id;
      v.notes = notes.trim();
      renderVersions();
      toast("Renamed " + v.id, "ok");
      return;
    }
    try {
      await api("/api/versions/rename", { id: v.id, label, notes });
      await refreshVersions();
      toast("Renamed " + v.id, "ok");
    } catch (e) { toast(e.message, "error"); }
  }

  async function deleteVersion(v) {
    if (!confirm(`Delete version ${v.id} (${v.label})?\n\n` + (v.demo
        ? "This is a demo row with no directory behind it, so nothing on disk " +
          "changes and a reload brings it back. "
        : `This removes ${v.dir} from disk. `) +
        "The live training checkpoint is not touched.")) return;
    if (v.demo) {
      // Nothing on disk to remove; drop it from the listing this page is holding
      // so the button behaves, and let the next bootstrap bring it back.
      state.boot.versions = (state.boot.versions || []).filter((x) => x !== v);
      renderVersions();
      renderVersionSelect();
      toast("Deleted " + v.id, "ok");
      return;
    }
    try {
      await api("/api/versions/delete", { id: v.id });
      await refreshVersions();
      toast("Deleted " + v.id, "ok");
    } catch (e) { toast(e.message, "error"); }
  }

  /** Version chips, one row per ROLE. Multi-select, because the queue runs one
   *  benchmark per picked version; a single pick behaves exactly as it did before
   *  the sweep existed.
   *
   *  The two rows are independent because the adapters are: `--learn` trains one
   *  side against a frozen opponent, so the defender worth evaluating and the
   *  attacker worth evaluating it against are normally different snapshots. A
   *  version that does not hold a role is shown but not selectable for it --
   *  clicking it would queue a run the server refuses, and the reason is easier to
   *  read on the chip than in an error toast. */
  const versionSelection = new Set(["current"]);
  const defenderSelection = new Set(["current"]);

  function renderVersionSelect() {
    renderRoleVersions("#bench-versions", versionSelection, "attacker");
    renderRoleVersions("#bench-defender-versions", defenderSelection, "defender");
  }

  function renderRoleVersions(container, selected, role) {
    const node = $(container);
    if (!node) return;
    const all = state.boot.versions || [];
    const holds = (v) => (v.available || {})[role] !== false;
    const live = ((state.boot.live || {}).adapters || {})[role] || {};
    // What can actually be selected for this role. "current" is in here only when
    // the live checkpoint HOLDS this role: the default selection is "current", and
    // keeping an unselectable id in the set rendered a row with nothing lit up
    // while still sending that id to the server -- so the page said "no version
    // picked" and the run was refused for picking one.
    const usableIds = new Set(all.filter(holds).map((v) => v.id));
    if (live.exists !== false) usableIds.add("current");
    Array.from(selected).forEach((id) => {
      if (!usableIds.has(id)) selected.delete(id);
    });
    if (!selected.size && usableIds.size) {
      // Prefer the live checkpoint, else the newest version that holds the role
      // (the listing is newest-first).
      selected.add(usableIds.has("current") ? "current"
                   : all.filter(holds)[0].id);
    }

    const entries = [{
      id: "current", label: "live checkpoint", usable: live.exists !== false,
      note: live.exists === false
        ? `nothing at ${live.path || "checkpoints/" + role + "_adapter"} yet`
        : `whatever training last wrote to checkpoints/${role}_adapter`,
    }].concat(all.map((v) => ({
      id: v.id, label: v.label, usable: holds(v),
      note: v.id +
        (holds(v) ? "" : ` · no ${role} adapter in this version`) +
        (v.rounds_done !== null && v.rounds_done !== undefined ? ` · ${v.rounds_done} rounds done` : "") +
        (v.created ? ` · ${v.created.replace("T", " ")}` : "") +
        (v.notes ? `\n${v.notes}` : ""),
    })));

    node.innerHTML = "";
    entries.forEach((e) => {
      const on = e.usable && selected.has(e.id);
      node.appendChild(el("button", {
        class: "chip" + (on ? " on" : "") + (e.usable ? "" : " locked"),
        text: e.label, title: e.note, disabled: !e.usable,
        onclick: () => {
          if (!e.usable) return;
          if (selected.has(e.id)) {
            if (selected.size === 1) { toast("Keep at least one version selected"); return; }
            selected.delete(e.id);
          } else selected.add(e.id);
          renderVersionSelect();
        },
      }));
    });
  }

  /* ------------------------------------------------------------------ */
  /* Benchmark: panels, live theatre, matrix                             */
  /* ------------------------------------------------------------------ */
  const ATTACK_NOTE = {
    clean: "control row: nothing is poisoned, so each defense's row is its clean accuracy",
    llm: "the trained attacker adapter -- the system under test",
    lie: "LIE / ALIE, Baruch et al., NeurIPS 2019",
    min_max: "AGR-agnostic Min-Max, Shejwalkar & Houmansadr, NDSS 2021",
    min_sum: "AGR-agnostic Min-Sum, Shejwalkar & Houmansadr, NDSS 2021",
    fang: "AGR-tailored (trimmed-mean/median), Fang et al., USENIX Sec 2020",
    fang_krum: "AGR-tailored (Krum), Fang et al., USENIX Sec 2020",
    ipm: "Inner Product Manipulation, Xie et al., UAI 2019",
    mimic: "Mimic, Karimireddy et al., ICLR 2022",
    sign_flip: "classic sign-flipping Byzantine baseline",
    noise: "classic Gaussian Byzantine baseline",
    scaling: "boosting / model replacement, Bagdasaryan et al., AISTATS 2020",
    label_flip: "untargeted DATA poisoning -- forces per-round benign retraining",
  };
  const DEFENSE_NOTE = {
    fedavg: "no defense: plain FedAvg over every update",
    oracle: "reads the ground truth -- the upper bound, not a real defense",
    llm_defender: "the trained defender adapter (needs defense.mode: llm training)",
    fltrust: "cosine trust vs a clean root update, norm-rescaled (NDSS'21)",
    defl: "per-layer FGNV + MOUD-Vote + CLP gating with Beta trust (AAAI-23)",
    dnc: "spectral top-singular-direction outlier filter (NDSS'21)",
    multikrum: "distance-based selection of the n-f most central updates (NeurIPS'17)",
  };

  function renderChips(container, names, selected, notes, cls) {
    const node = $(container);
    node.innerHTML = "";
    names.forEach((name) => {
      const on = selected.has(name);
      const chip = el("button", {
        class: "chip" + (on ? " on " + (cls || "") : ""), text: name,
        title: notes[name] || name,
        onclick: () => {
          if (selected.has(name)) {
            if (selected.size === 1) { toast("Keep at least one selected"); return; }
            selected.delete(name);
          } else selected.add(name);
          renderChips(container, names, selected, notes, cls);
          renderFocusSelects();
        },
      });
      node.appendChild(chip);
    });
  }

  const benchSelection = { attacks: new Set(), defenses: new Set() };

  const BENCH_ADVANCED = [
    ["attack_retries", "Attack retries", "number", 3, "resamples when the policy emits no usable plan"],
    ["defender_temperature", "Defender temperature", "number", 0, "only used by the llm_defender column"],
    ["log_every", "Log every N rounds", "number", 10, "how often the CLI prints a progress line"],
    ["demo_round_delay", "Demo round delay (s)", "text", "",
     "MIN,MAX seconds between rounds — demo replay only (blank = 60,120)"],
    ["root_size", "FLTrust root size", "number", 100, "clean server-held examples FLTrust bootstraps trust from"],
    ["root_epochs", "FLTrust root epochs", "number", "", "blank = iteration-matched to an honest client"],
    ["eta", "FLTrust eta", "number", 1.0, "global learning rate applied to the trust-weighted update"],
    ["defl_delta", "DeFL delta", "number", 0.05, ""],
    ["defl_tau", "DeFL tau", "number", 2.5, ""],
    ["dnc_num_byzantine", "DnC assumed #malicious", "number", "", "blank = the eval quota, clamped to a minority"],
    ["dnc_c", "DnC filtering fraction c", "number", 1.0, ""],
    ["dnc_niters", "DnC subsampling iterations", "number", 1, ""],
    ["dnc_sub_dim", "DnC subsample dimension", "number", 10000, ""],
    ["multikrum_f", "Multi-Krum f", "number", "", "blank = the eval quota, clamped to a minority"],
    ["multikrum_m", "Multi-Krum m", "number", "", "blank = n - f"],
    ["lie_z", "LIE z", "number", "", "blank = the paper's n/m-derived value"],
    ["minmax_perturbation", "Min-Max perturbation", "select", "std", "", ["std", "unit_vec", "sign"]],
    ["minmax_gamma0", "Min-Max gamma0", "number", 10.0, ""],
    ["minsum_bound", "Min-Sum bound", "select", "max", "", ["max", "min"]],
    ["fang_b", "Fang b", "number", 2.0, ""],
    ["ipm_epsilon", "IPM epsilon", "number", 0.1, ""],
    ["mimic_warmup", "Mimic warmup", "number", 10, ""],
    ["noise_sigma", "Noise sigma", "number", 10.0, ""],
    ["signflip_c", "Sign-flip c", "number", 1.0, ""],
    ["scaling_gamma", "Scaling gamma", "number", 10.0, ""],
    ["labelflip_mode", "Label-flip mode", "select", "reverse", "", ["reverse", "next", "random"]],
  ];

  function renderBenchAdvanced() {
    const node = $("#bench-advanced");
    node.innerHTML = "";
    BENCH_ADVANCED.forEach(([key, label, type, def, doc, options]) => {
      const field = el("div", { class: "field" });
      field.appendChild(el("label", { text: label }));
      let input;
      if (type === "select") {
        input = el("select", { id: "adv-" + key });
        (options || []).forEach((o) => input.appendChild(el("option", { value: o, text: o })));
        input.value = def;
      } else if (type === "text") {
        // Not every knob is a single number -- the demo's pacing is a MIN,MAX
        // pair, and a number input silently refuses to hold it.
        input = el("input", { type: "text", id: "adv-" + key,
                              placeholder: def === "" ? "CLI default" : String(def) });
      } else {
        input = el("input", { type: "number", step: "any", id: "adv-" + key,
                              placeholder: def === "" ? "CLI default" : String(def) });
      }
      field.appendChild(input);
      // The note lives on the field rather than under it: this panel is a dense
      // grid of ~25 knobs, and a line of prose per knob is what made it a wall of
      // text. Hover still explains what "blank" means for the ones where it
      // decides something.
      if (doc) field.title = doc;
      node.appendChild(field);
    });
    // Switches that are flags rather than values.
    [["no_eval_cache", "Disable the accuracy cache", "recomputes every test pass; slower but rules the cache out"],
     ["benign_retrain", "Retrain benign clients each round", "matches training's benign_retrain_each_round"],
     ["no_plot", "Skip the matplotlib PNGs", "the panel draws its own charts either way"],
     ["fresh", "Force a fresh Phase 1", "ignores the saved checkpoint for this run"]].forEach(
      ([key, label, doc]) => {
        const field = el("div", { class: "field" });
        field.appendChild(el("label", { text: label }));
        field.appendChild(el("label", { class: "switch" }, [
          el("input", { type: "checkbox", id: "adv-" + key }), el("span", { class: "hint", text: "off" }),
        ]));
        const box = field.querySelector("input");
        box.addEventListener("change", () => {
          box.nextElementSibling.textContent = box.checked ? "on" : "off";
        });
        if (doc) field.title = doc;
        node.appendChild(field);
      });
  }

  function renderFocusSelects() {
    const attacks = state.bench.cfg ? state.bench.cfg.attacks : Array.from(benchSelection.attacks);
    const defenses = state.bench.cfg ? state.bench.cfg.defenses : Array.from(benchSelection.defenses);
    // Default the focus to the thing anyone opening this page came to look at: the
    // trained policy against a real defense. Falling through to the panel's first
    // entry lands on `clean` vs `fedavg` -- the control row under no defense, the
    // one cell in the matrix guaranteed to show nothing happening.
    // A focus carried over from the FORM's panel may not exist in the run that
    // actually started, so it only counts while it is still in the list.
    const keptAttack = attacks.includes(state.bench.focusAttack) ? state.bench.focusAttack : null;
    const keptDefense = defenses.includes(state.bench.focusDefense) ? state.bench.focusDefense : null;
    const preferAttack = keptAttack ||
      (attacks.includes("llm") ? "llm" : attacks.find((a) => a !== "clean")) || attacks[0];
    const preferDefense = keptDefense ||
      defenses.find((d) => d !== "fedavg" && d !== "oracle") || defenses[0];
    fillSelect($("#bench-focus-attack"), attacks, preferAttack, "attack: ");
    fillSelect($("#bench-focus-defense"), defenses, preferDefense, "defense: ");
    state.bench.focusAttack = $("#bench-focus-attack").value || attacks[0] || null;
    state.bench.focusDefense = $("#bench-focus-defense").value || defenses[0] || null;
  }

  function fillSelect(sel, values, keep, prefix) {
    sel.innerHTML = "";
    values.forEach((v) => sel.appendChild(el("option", { value: v, text: (prefix || "") + v })));
    if (keep && values.includes(keep)) sel.value = keep;
  }

  function benchCell(attack, defense) {
    const a = state.bench.agg[attack] || (state.bench.agg[attack] = {});
    return a[defense] || (a[defense] = {
      n: 0, accSum: 0, last: null, tp: 0, fn: 0, fp: 0, tn: 0,
      goalSum: 0, goalHits: 0, history: [],
    });
  }

  function handleBenchEvent(ev) {
    switch (ev.event) {
      case "run_started": {
        // Leg 0 of a sweep is a fresh benchmark; legs 1..n keep everything that is
        // already on the page, because the comparison is the point of the sweep.
        const q = ev.queue || { index: 0, total: 1, label: ev.version_label };
        if (q.index === 0) {
          resetBench();
          $("#bench-console").innerHTML = "";
        } else {
          resetBenchLeg();
        }
        // A leg is identified by the version on the axis being swept: in a
        // defender sweep the attacker is pinned, so labelling every row with the
        // attacker id would print the same id on all of them.
        const legAxis = q.axis || "attacker";
        state.bench.version = {
          id: (legAxis === "defender" ? q.defender_version : q.version)
              || ev.version || "current",
          label: q.label || ev.version_label,
          attacker: q.version || null,
          defender: q.defender_version || null,
        };
        state.bench.queue = q;
        appendLog("bench", (q.total > 1
          ? `── version ${q.index + 1} of ${q.total}: ${q.label} ──\n` : "") +
          "$ " + ev.command, "ev");
        toast(q.total > 1
          ? `Benchmarking ${q.label} (${q.index + 1} of ${q.total})`
          : "Benchmark started", "ok");
        break;
      }
      case "log":
        appendLog("bench", ev.line, ev.level);
        break;
      case "started":
        appendLog("bench", "benchmark process up; loading the model and data…", "ev");
        break;
      case "config":
        state.bench.cfg = ev;
        renderFocusSelects();
        buildBenchChart();
        renderHeat();
        appendLog("bench",
          `panel: ${ev.attacks.length} attack(s) × ${ev.defenses.length} defense(s), ` +
          `${ev.n_poisoners}/${ev.n_clients} poisoned per round, baseline ${pct(ev.baseline_accuracy, 2)}`,
          "ev");
        if (ev.defender_adapter) {
          appendLog("bench", "llm_defender loads " + ev.defender_adapter, "ev");
        }
        // The CLI drops the llm_defender column when no defender adapter is on
        // disk and carries on -- correct for a run whose other six columns are
        // still worth measuring, and invisible on a page that only reads the
        // resulting `defenses` list. A missing row and a row never asked for look
        // identical in the matrix, so the difference is said out loud.
        if (ev.llm_defender_skipped) {
          appendLog("bench",
            "the llm_defender column was DROPPED: no trained defender adapter was " +
            "found, so this matrix has no defender-LLM row. Train a defender " +
            "(defense.mode: llm with --learn defender) to include it.", "warn");
          toast("llm_defender dropped — no defender adapter", "error");
        }
        break;
      case "round":
        onBenchRound(ev);
        break;
      case "round_skipped":
        appendLog("bench", `round ${ev.round_num} skipped for the whole panel: ${ev.reason}`, "warn");
        break;
      case "summary": {
        state.bench.summary = ev;
        const v = state.bench.version || { id: "current", label: "live checkpoint" };
        state.bench.byVersion.push({ version: v, summary: ev });
        renderBenchSummary(ev);
        renderVersionComparison();
        break;
      }
      case "saved":
        appendLog("bench", "results saved to " + ev.out_dir, "ev");
        break;
      case "queue_abandoned":
        appendLog("bench",
          `-- sweep abandoned after ${ev.after}; ${ev.remaining} version(s) not run` +
          (ev.error ? ": " + ev.error : "") + " --", "warn");
        toast(`Sweep stopped: ${ev.remaining} version(s) not run`, "error");
        break;
      case "run_ended": {
        const q = state.bench.queue || { index: 0, total: 1 };
        const more = q.index + 1 < q.total && ev.state === "finished";
        appendLog("bench", `-- run ${ev.state} (exit ${ev.exit_code}) after ${ev.elapsed}s --` +
          (more ? " next version starting…" : ""), ev.state === "failed" ? "error" : "ev");
        if (!more) toast(`Benchmark ${ev.state}`, ev.state === "failed" ? "error" : "ok");
        loadRuns();
        break;
      }
      default:
        break;
    }
  }

  /** Clear the live panels for ONE leg of a sweep, keeping the finished versions'
   *  summaries -- those are the comparison the sweep exists to build. */
  function resetBenchLeg() {
    state.bench.agg = {};
    state.bench.rounds = [];
    state.bench.cfg = null;
    state.bench.summary = null;
    state.bench.lastRound = null;
    $("#bench-summary-panel").classList.add("hidden");
    $("#bench-heat").innerHTML = "";
    $("#bench-roundstrip").innerHTML = "";
    $("#bench-federation").innerHTML = "";
    $("#bench-round-detail").innerHTML = "";
    $("#bench-round-caption").textContent = "no round yet";
    kpi($("#bench-kpis"), []);
    if (state.bench.charts.acc) { state.bench.charts.acc.reset(); state.bench.charts.acc.draw(); }
  }

  function resetBench() {
    resetBenchLeg();
    state.bench.byVersion = [];
    state.bench.queue = null;
    state.bench.version = null;
    $("#bench-compare-panel").classList.add("hidden");
    $("#bench-compare-table").innerHTML = "";
  }

  function buildBenchChart() {
    const P = C.PALETTE();
    const defenses = (state.bench.cfg && state.bench.cfg.defenses) || [];
    const colors = [P.defense, P.good, P.warn, P.accent, P.attack, P.dim, P.faint];
    const series = defenses.map((d, i) => ({
      key: d, label: d, color: colors[i % colors.length], width: 1.6,
    }));
    if (state.bench.charts.acc) state.bench.charts.acc._ro.disconnect();
    state.bench.charts.acc = new C.LineChart($("#chart-bench-acc"), {
      series, yFormat: (v) => (v * 100).toFixed(1) + "%",
    });
    legend("#legend-bench-acc", series);
  }

  function onBenchRound(ev) {
    state.bench.lastRound = ev;
    state.bench.rounds.push(ev);
    (ev.cells || []).forEach((c) => {
      const cell = benchCell(c.attack, c.defense);
      cell.n++;
      cell.accSum += c.accuracy;
      cell.last = c.accuracy;
      cell.tp += c.tp; cell.fn += c.fn; cell.fp += c.fp; cell.tn += c.tn;
      if (typeof c.goal_success === "number") cell.goalSum += c.goal_success;
      if (c.goal_hit) cell.goalHits++;
      cell.history.push(c);
      if (cell.history.length > 4000) cell.history.shift();
    });

    // The accuracy chart tracks the focused attack across every defense.
    const focus = state.bench.focusAttack;
    if (focus && state.bench.charts.acc) {
      const values = {};
      (ev.cells || []).forEach((c) => { if (c.attack === focus) values[c.defense] = c.accuracy; });
      state.bench.charts.acc.push(ev.round_num, values);
      state.bench.charts.acc.draw();
    }

    renderBenchKpis(ev);
    renderBenchRound(ev);
    renderHeat();
    renderRoundStrip();
  }

  function renderBenchKpis(ev) {
    const cfg = state.bench.cfg || {};
    const attacks = cfg.attacks || [];
    const defenses = cfg.defenses || [];
    const base = cfg.baseline_accuracy;

    // Averaged over the defenses that are actually defenses. fedavg is the
    // no-defense control and oracle reads the ground truth, so leaving either in
    // makes the ranking meaningless -- fedavg in particular wins "strongest
    // defense" outright, because nothing it fails to reject can cost it a false
    // alarm. Fall back to the whole panel if narrowing leaves nothing.
    const real = defenses.filter((d) => d !== "oracle" && d !== "fedavg");
    const scored = real.length ? real : defenses;
    let worstAttack = null, worstDrop = -Infinity;
    attacks.filter((a) => a !== "clean").forEach((a) => {
      const drops = scored.map((d) => {
        const c = benchCell(a, d);
        return c.n ? base - c.accSum / c.n : null;
      }).filter((x) => x !== null);
      const m = mean(drops);
      if (m !== null && m > worstDrop) { worstDrop = m; worstAttack = a; }
    });
    let bestDefense = null, bestDrop = Infinity;
    scored.forEach((d) => {
      const drops = attacks.filter((a) => a !== "clean").map((a) => {
        const c = benchCell(a, d);
        return c.n ? base - c.accSum / c.n : null;
      }).filter((x) => x !== null);
      const m = mean(drops);
      if (m !== null && m < bestDrop) { bestDrop = m; bestDefense = d; }
    });

    kpi($("#bench-kpis"), [
      { k: "round", v: `${ev.index} / ${ev.of}` },
      state.bench.queue && state.bench.queue.total > 1
        ? { k: "version", v: state.bench.queue.label,
            s: `${state.bench.queue.index + 1} of ${state.bench.queue.total} in the sweep` }
        : { k: "panel", v: `${attacks.length}×${defenses.length}`,
            s: "attacks × defenses" },
      { k: "poisoned", v: (ev.poisoned || []).length,
        s: "of " + ev.n_clients + " clients", tone: "atk" },
      { k: "baseline", v: pct(base, 2), s: "clean Phase-1 accuracy" },
      { k: "strongest attack", v: worstAttack || "--", tone: "atk",
        s: worstDrop === -Infinity ? "" : "mean drop " + signed(worstDrop, 4),
        title: "mean accuracy drop over " + scored.join(", ") +
               " (fedavg and oracle are excluded: one is no defense, the other reads the truth)" },
      { k: "strongest defense", v: bestDefense || "--", tone: "def",
        s: bestDrop === Infinity ? "" : "mean drop " + signed(bestDrop, 4),
        title: "the defense that lost the least accuracy across the attack panel" },
    ]);
  }

  function renderBenchRound(ev) {
    const attack = state.bench.focusAttack, defense = state.bench.focusDefense;
    const cell = (ev.cells || []).find((c) => c.attack === attack && c.defense === defense);
    const shift = (ev.shifts || {})[attack] || {};
    $("#bench-round-caption").textContent =
      `round ${ev.round_num} (${ev.index} of ${ev.of}) · poisoned ${(ev.poisoned || []).join(", ") || "none"}`;

    renderFederation($("#bench-federation"), {
      n: ev.n_clients,
      poisoned: cell && cell.poisoned ? cell.poisoned : ev.poisoned,
      pool: ev.pool,
      flagged: cell ? cell.flagged : [],
      shifts: shift.per_client || {},
    });

    const detail = $("#bench-round-detail");
    detail.innerHTML = "";
    if (!cell) {
      detail.appendChild(el("div", { class: "hint", text: "no cell for this attack/defense pair" }));
      return;
    }
    const base = (state.bench.cfg || {}).baseline_accuracy;
    const table = el("table", { class: "grid" });
    table.appendChild(el("thead", {}, [el("tr", {}, ["defense", "accuracy", "drop", "caught", "through", "false alarms", "goal"]
      .map((h) => el("th", { text: h })))]));
    const body = el("tbody");
    (ev.cells || []).filter((c) => c.attack === attack).forEach((c) => {
      const row = el("tr", { class: c.defense === defense ? "best" : "" }, [
        el("td", { text: c.defense, title: DEFENSE_NOTE[c.defense] || "" }),
        el("td", { class: "num", text: pct(c.accuracy, 2) }),
        el("td", { class: "num", text: signed(base - c.accuracy, 4),
                   style: { color: base - c.accuracy > 0.005 ? "var(--attack)" : "" } }),
        el("td", { class: "num", text: c.tp }),
        el("td", { class: "num", text: c.fn,
                   style: { color: c.fn > 0 ? "var(--fn)" : "" } }),
        el("td", { class: "num", text: c.fp,
                   style: { color: c.fp > 0 ? "var(--fp)" : "" } }),
        el("td", { class: "num", text: c.goal_success === null || c.goal_success === undefined
          ? "n/a" : pct(c.goal_success, 0) }),
      ]);
      body.appendChild(row);
    });
    table.appendChild(body);
    detail.appendChild(el("div", { class: "scroll-x" }, [table]));
    if (shift.mean !== undefined) {
      detail.appendChild(el("div", { class: "hint mt", text:
        `${attack} moved its cohort by ‖Δw‖ = ${num(shift.mean, 4)} on average ` +
        `(max ${num(shift.max, 4)}). Hover a client for its own magnitude.` }));
    }
  }

  function renderRoundStrip() {
    const strip = $("#bench-roundstrip");
    const attack = state.bench.focusAttack, defense = state.bench.focusDefense;
    strip.innerHTML = "";
    state.bench.rounds.slice(-240).forEach((ev) => {
      const c = (ev.cells || []).find((x) => x.attack === attack && x.defense === defense);
      const g = c && typeof c.goal_success === "number" ? c.goal_success : 0;
      strip.appendChild(el("i", {
        style: { height: (4 + g * 30) + "px",
                 background: g > 0 ? C.heatColor(g, 1) : "var(--tn)",
                 border: c && c.goal_hit ? "1px solid var(--attack)" : "none" },
        title: `round ${ev.round_num}\ngoal achieved ${pct(g, 0)}` +
               (c ? `\naccuracy ${pct(c.accuracy, 2)}\ncaught ${c.tp}, through ${c.fn}` : ""),
        onclick: () => { state.bench.lastRound = ev; renderBenchRound(ev); },
      }));
    });
  }

  function heatValue(attack, defense, metric) {
    const c = benchCell(attack, defense);
    if (!c.n) return null;
    const base = (state.bench.cfg || {}).baseline_accuracy;
    switch (metric) {
      case "detection": return c.tp + c.fn ? c.tp / (c.tp + c.fn) : null;
      case "goal": return c.goalSum / c.n;
      case "accuracy": return c.accSum / c.n;
      default: return base - c.accSum / c.n;
    }
  }

  function renderHeat() {
    const cfg = state.bench.cfg;
    const table = $("#bench-heat");
    table.innerHTML = "";
    if (!cfg) return;
    const metric = $("#bench-heat-metric").value;
    const invert = metric === "detection" || metric === "accuracy";
    // Only POSITIVE drops get ink. A negative mean drop means the attack left the
    // model better than the clean baseline (noise, mostly); shading it by magnitude
    // painted "did nothing" the same red as "took four points off".
    const ink = (v) => (v === null ? null : (invert ? v : Math.max(0, v)));
    const values = [];
    cfg.attacks.forEach((a) => cfg.defenses.forEach((d) => {
      const v = ink(heatValue(a, d, metric));
      if (v !== null) values.push(v);
    }));
    const max = values.length ? Math.max.apply(null, values) : 1;

    table.appendChild(el("thead", {}, [el("tr", {}, [el("th", {})].concat(
      cfg.defenses.map((d) => el("th", { text: d, title: DEFENSE_NOTE[d] || "" }))))]));
    const body = el("tbody");
    cfg.attacks.forEach((a) => {
      const cells = cfg.defenses.map((d) => {
        const v = heatValue(a, d, metric);
        const c = benchCell(a, d);
        const text = v === null ? "–"
          : (metric === "acc_drop" ? signed(v, 3)
            : metric === "accuracy" ? pct(v, 1) : pct(v, 0));
        return el("td", {
          class: v === null ? "na" : "",
          style: { background: C.heatColor(ink(v) || 0, max, invert) },
          title: `${a} vs ${d}\n${c.n} round(s)\nmean accuracy ${pct(c.n ? c.accSum / c.n : null, 2)}\n` +
                 `detection ${pct(c.tp + c.fn ? c.tp / (c.tp + c.fn) : null, 0)}\n` +
                 `false alarms ${c.fp}`,
          text: text,
        });
      });
      body.appendChild(el("tr", {}, [
        el("th", { class: "rowh", text: a, title: ATTACK_NOTE[a] || "" })].concat(cells)));
    });
    table.appendChild(body);
  }

  function renderBenchSummary(ev) {
    $("#bench-summary-panel").classList.remove("hidden");
    $("#bench-summary-caption").textContent =
      `${ev.measured_rounds} measured round(s)` +
      (ev.skipped_rounds ? ` · ${ev.skipped_rounds} skipped` : "") +
      ` · baseline ${pct(ev.baseline_accuracy, 2)}`;

    const rows = [];
    ev.attacks.forEach((a) => ev.defenses.forEach((d) => {
      const s = (ev.summaries[a] || {})[d];
      if (s) rows.push(s);
    }));

    // Verdict: the single sentence a reader wants before the table.
    const real = ev.defenses.filter((d) => d !== "oracle" && d !== "fedavg");
    const byAttack = ev.attacks.filter((a) => a !== "clean").map((a) => ({
      attack: a,
      drop: mean(real.map((d) => ((ev.summaries[a] || {})[d] || {}).mean_acc_drop)),
      goal: mean(real.map((d) => ((ev.summaries[a] || {})[d] || {}).goal_success_rate)),
      detected: mean(real.map((d) => ((ev.summaries[a] || {})[d] || {}).detection_rate)),
    })).sort((x, y) => (y.drop || 0) - (x.drop || 0));
    const byDefense = real.map((d) => ({
      defense: d,
      drop: mean(ev.attacks.filter((a) => a !== "clean")
        .map((a) => ((ev.summaries[a] || {})[d] || {}).mean_acc_drop)),
      detected: mean(ev.attacks.filter((a) => a !== "clean")
        .map((a) => ((ev.summaries[a] || {})[d] || {}).detection_rate)),
    })).sort((x, y) => (x.drop || 0) - (y.drop || 0));

    const verdict = $("#bench-verdict");
    verdict.innerHTML = "";

    // Which side this run was started to test. It decides which paragraph leads
    // and which one reports "and here is how the thing under test did" -- a
    // defender run that opened with the hardest ATTACK buried its own answer.
    const forDefender = ((state.bench.queue || {}).axis || "attacker") === "defender";

    const hardestAttack = () => byAttack.length && el("p", { style: { margin: "0 0 8px" } }, [
      document.createTextNode("Across the "),
      el("strong", { text: String(real.length) }),
      document.createTextNode(" robust defense(s), the attack that cost the most accuracy was "),
      el("strong", { text: byAttack[0].attack, style: { color: "var(--attack)" } }),
      document.createTextNode(` (mean drop ${signed(byAttack[0].drop, 4)}, ` +
        `${pct(byAttack[0].goal, 0)} of its goal, ${pct(byAttack[0].detected, 0)} detected).`),
    ]);

    const bestDefense = () => byDefense.length && el("p", { style: { margin: "0 0 8px" } }, [
      document.createTextNode("The defense that held up best was "),
      el("strong", { text: byDefense[0].defense, style: { color: "var(--defense)" } }),
      document.createTextNode(` (mean drop ${signed(byDefense[0].drop, 4)} across the attack ` +
        `panel, ${pct(byDefense[0].detected, 0)} of poisoned updates flagged); the weakest was `),
      el("strong", { text: byDefense[byDefense.length - 1].defense }),
      document.createTextNode(` (${signed(byDefense[byDefense.length - 1].drop, 4)}).`),
    ]);

    // The trained adapter's own line. Both rankings run best-first, but "best"
    // is opposite for the two sides: byAttack is sorted by descending drop
    // caused, byDefense by ascending drop allowed.
    const underTest = () => {
      if (forDefender) {
        const i = byDefense.findIndex((x) => x.defense === "llm_defender");
        if (i < 0) return null;
        const d = byDefense[i];
        return el("p", { style: { margin: "0 0 8px" } }, [
          document.createTextNode("The trained defender ("),
          el("strong", { text: "llm_defender", style: { color: "var(--defense)" } }),
          document.createTextNode(`) ranks ${i + 1} of ${byDefense.length} by how little ` +
            `accuracy it let the attack panel take: ${signed(d.drop, 4)}, flagging ` +
            `${pct(d.detected, 0)} of the poisoned updates.`),
        ]);
      }
      const i = byAttack.findIndex((x) => x.attack === "llm");
      if (i < 0) return null;
      const a = byAttack[i];
      return el("p", { style: { margin: "0 0 8px" } }, [
        document.createTextNode("The trained policy ("),
        el("strong", { text: "llm", style: { color: "var(--attack)" } }),
        document.createTextNode(`) ranks ${i + 1} of ${byAttack.length} by mean accuracy drop: ` +
          `${signed(a.drop, 4)}, achieving ${pct(a.goal, 0)} of its requested degradation, ` +
          `${pct(a.detected, 0)} of it caught.`),
      ]);
    };

    const order = forDefender ? [bestDefense, underTest, hardestAttack]
                              : [hardestAttack, underTest, bestDefense];
    order.map((f) => f()).filter(Boolean).forEach((b) => verdict.appendChild(b));
    verdict.appendChild(el("p", { class: "hint", style: { marginTop: "10px" }, text:
      "fedavg is the no-defense control and oracle reads the ground truth, so both are " +
      "excluded from these rankings. Read every row against its own defense's clean " +
      "accuracy — rows are not comparable across defenses." }));

    renderSummaryTable(rows, ev);
    $("#bench-report").textContent = ev.report || "";
  }

  /** Collapse one leg's whole matrix into the handful of numbers that answer
   *  "did this checkpoint do its job better than that one?".
   *
   *  Which slice of the matrix answers that depends on WHICH ADAPTER is varying:
   *
   *  - an attacker sweep reads the `llm` ROW -- how much damage this attacker got
   *    through each defense. The published baselines are identical across legs by
   *    construction (same rounds, same honest updates, same poisoned set), so they
   *    are the control that says the legs really were comparable, not a result.
   *  - a defender sweep reads the `llm_defender` COLUMN -- how much damage this
   *    defender ALLOWED, and what it caught, across the attacks. Here `drop` is a
   *    cost rather than an achievement, so the best leg is the lowest one.
   */
  function versionScore(summary, axis) {
    if (axis === "defender") return defenderScore(summary);
    const real = summary.defenses.filter((d) => d !== "oracle" && d !== "fedavg");
    const scored = real.length ? real : summary.defenses;
    const row = summary.summaries.llm || {};
    const cells = scored.map((d) => row[d]).filter(Boolean);
    const undefended = (row.fedavg || {});
    return {
      axis: "attacker", against: scored, lowerIsBetter: false,
      rounds: cells.length ? cells[0].rounds : 0,
      drop: mean(cells.map((c) => c.mean_acc_drop)),
      goal: mean(cells.map((c) => c.goal_success_rate)),
      goalFull: mean(cells.map((c) => c.goal_full_success_rate)),
      detected: mean(cells.map((c) => c.detection_rate)),
      evasion: mean(cells.map((c) => c.attack_success_rate)),
      reference: undefended.mean_acc_drop,
      perCell: Object.fromEntries(scored.map((d) => [d, (row[d] || {}).mean_acc_drop])),
      usable: Boolean(summary.summaries.llm),
    };
  }

  function defenderScore(summary) {
    // `clean` poisons nothing, so it measures the defense's false-alarm cost, not
    // its defending -- averaging it into "drop allowed" would reward a defense for
    // the rounds there was nothing to stop.
    const attacks = Object.keys(summary.summaries).filter((a) => a !== "clean");
    const cells = attacks.map((a) => (summary.summaries[a] || {}).llm_defender)
      .filter(Boolean);
    return {
      axis: "defender", against: attacks, lowerIsBetter: true,
      rounds: cells.length ? cells[0].rounds : 0,
      drop: mean(cells.map((c) => c.mean_acc_drop)),
      goal: mean(cells.map((c) => c.goal_success_rate)),
      goalFull: mean(cells.map((c) => c.goal_full_success_rate)),
      detected: mean(cells.map((c) => c.detection_rate)),
      evasion: mean(cells.map((c) => c.attack_success_rate)),
      fpr: mean(cells.map((c) => c.fpr)),
      f1: mean(cells.map((c) => c.f1)),
      reference: undefined,
      perCell: Object.fromEntries(attacks.map((a) =>
        [a, ((summary.summaries[a] || {}).llm_defender || {}).mean_acc_drop])),
      usable: cells.length > 0,
    };
  }

  function renderVersionComparison() {
    const axis = (state.bench.queue || {}).axis || "attacker";
    const panel = $("#bench-compare-panel");
    const scored = state.bench.byVersion
      .map((l) => ({ version: l.version, ...versionScore(l.summary, axis) }))
      .filter((s) => s.usable);
    if (scored.length < 2) { panel.classList.add("hidden"); return; }
    panel.classList.remove("hidden");

    const against = scored[0].against;
    const lower = scored[0].lowerIsBetter;
    // "Best" flips with the axis: an attacker wants the drop it caused to be
    // large, a defender wants the drop it allowed to be small.
    const best = scored.reduce((a, b) => {
      const x = a.drop, y = b.drop;
      if (y === null || y === undefined) return a;
      if (x === null || x === undefined) return b;
      return (lower ? y < x : y > x) ? b : a;
    });
    const q = state.bench.queue || {};
    $("#bench-compare-caption").textContent =
      `${scored.length} ${axis === "defender" ? "defender" : "attacker"} version(s)` +
      (q.total && q.total > scored.length ? ` of ${q.total} — sweep in progress` : "") +
      ` · ${scored[0].rounds} rounds each · ` +
      (axis === "defender"
        ? "llm_defender column only — lower drop is a better defense"
        : "llm row only");

    const table = $("#bench-compare-table");
    table.innerHTML = "";
    const head = ["version", "rounds",
                  axis === "defender" ? "mean drop allowed" : "mean acc drop",
                  "goal (weighted)", "goal (full)", "detected", "evasion"]
      .concat(axis === "defender" ? ["FPR", "F1"] : ["undefended drop"])
      .concat(against.map((d) => (axis === "defender" ? "drop under " : "drop vs ") + d));
    table.appendChild(el("thead", {}, [el("tr", {}, head.map((h) => el("th", { text: h })))]));
    const body = el("tbody");
    scored.forEach((s) => {
      const cells = [
        el("td", {}, [
          el("div", { text: s.version.label, style: { fontWeight: "600" } }),
          el("div", { class: "hint", text: s.version.id }),
        ]),
        el("td", { class: "num", text: s.rounds }),
        el("td", { class: "num", text: signed(s.drop, 4),
                   style: s.drop > 0.005 ? { color: "var(--attack)" } : {} }),
        el("td", { class: "num", text: pct(s.goal, 1) }),
        el("td", { class: "num", text: pct(s.goalFull, 1) }),
        el("td", { class: "num", text: pct(s.detected, 0) }),
        el("td", { class: "num", text: pct(s.evasion, 0) }),
      ];
      if (axis === "defender") {
        cells.push(el("td", { class: "num", text: pct(s.fpr, 0),
                              title: "honest clients this defender rejected" }));
        cells.push(el("td", { class: "num", text: pct(s.f1, 0) }));
      } else {
        cells.push(el("td", { class: "num", text: signed(s.reference, 4),
          title: "the same attack against plain FedAvg — its ceiling with no defense" }));
      }
      body.appendChild(el("tr", { class: s === best ? "best" : "" },
        cells.concat(against.map((d) =>
          el("td", { class: "num", text: signed(s.perCell[d], 4) })))));
    });
    table.appendChild(body);
  }

  const SUMMARY_COLUMNS = [
    ["attack", "attack", (s) => s.attack, "text"],
    ["defense", "defense", (s) => s.defense, "text"],
    ["rounds", "rounds", (s) => s.rounds, "int"],
    ["mean_acc", "mean acc", (s) => s.mean_accuracy, "pct2"],
    ["final_acc", "final acc", (s) => s.final_accuracy, "pct2"],
    ["mean_acc_drop", "acc drop", (s) => s.mean_acc_drop, "signed"],
    ["goal_success_rate", "goal (weighted)", (s) => s.goal_success_rate, "pct0"],
    ["goal_full_success_rate", "goal (full)", (s) => s.goal_full_success_rate, "pct0"],
    ["detection_rate", "detection", (s) => s.detection_rate, "pct0"],
    ["fpr", "FPR", (s) => s.fpr, "pct0"],
    ["precision", "precision", (s) => s.precision, "pct0"],
    ["f1", "F1", (s) => s.f1, "pct0"],
    ["attack_success_rate", "evasion", (s) => s.attack_success_rate, "pct0"],
    ["false_alarms", "false alarms", (s) => s.false_alarms, "int"],
    ["skipped_rounds", "skipped", (s) => s.skipped_rounds, "int"],
  ];

  let summarySort = { key: "mean_acc_drop", asc: false };

  function renderSummaryTable(rows, ev) {
    const table = $("#bench-summary-table");
    table.innerHTML = "";
    const col = SUMMARY_COLUMNS.find((c) => c[0] === summarySort.key) || SUMMARY_COLUMNS[5];
    const sorted = rows.slice().sort((a, b) => {
      const x = col[2](a), y = col[2](b);
      if (typeof x === "string") return summarySort.asc ? x.localeCompare(y) : y.localeCompare(x);
      const dx = x === null || x === undefined ? -Infinity : x;
      const dy = y === null || y === undefined ? -Infinity : y;
      return summarySort.asc ? dx - dy : dy - dx;
    });

    table.appendChild(el("thead", {}, [el("tr", {}, SUMMARY_COLUMNS.map(([key, label]) =>
      el("th", {
        text: label,
        class: summarySort.key === key ? "sorted" + (summarySort.asc ? " asc" : "") : "",
        onclick: () => {
          summarySort = { key, asc: summarySort.key === key ? !summarySort.asc : false };
          renderSummaryTable(rows, ev);
        },
      })))]));

    const fmtCell = (v, kind) => {
      if (v === null || v === undefined) return "n/a";
      switch (kind) {
        case "pct2": return pct(v, 2);
        case "pct0": return pct(v, 0);
        case "signed": return signed(v, 4);
        case "int": return String(v);
        default: return String(v);
      }
    };
    const body = el("tbody");
    sorted.forEach((s) => {
      body.appendChild(el("tr", { class: s.attack === "llm" ? "best" : "" },
        SUMMARY_COLUMNS.map(([, , get, kind]) => el("td", {
          class: kind === "text" ? "" : "num",
          text: fmtCell(get(s), kind),
          title: kind === "text" ? (ATTACK_NOTE[get(s)] || DEFENSE_NOTE[get(s)] || "") : "",
          style: kind === "signed" && get(s) > 0.005 ? { color: "var(--attack)" } : {},
        }))));
    });
    table.appendChild(body);
  }

  /* ------------------------------------------------------------------ */
  /* Run history                                                         */
  /* ------------------------------------------------------------------ */
  async function loadRuns() {
    try {
      const data = await api("/api/runs");
      const runs = data.runs || [];
      $("#runs-empty").classList.toggle("hidden", runs.length > 0);
      const table = $("#runs-table");
      table.innerHTML = "";
      if (!runs.length) return;
      table.appendChild(el("thead", {}, [el("tr", {},
        ["run", "kind", "started", "what", "artifacts", ""].map((h) => el("th", { text: h })))]));
      const body = el("tbody");
      runs.forEach((r) => {
        const what = r.kind === "bench"
          ? `${(r.attacks || []).length} attacks × ${(r.defenses || []).length} defenses` +
            (r.version_label ? ` · ${r.version_label}` : "")
          : `${r.mode || "train"}${r.rounds ? " · " + r.rounds + " rounds" : ""}` +
            (r.defense_mode ? ` · defense ${r.defense_mode}` : "");
        body.appendChild(el("tr", {}, [
          el("td", { text: r.run_id }),
          el("td", { text: r.kind }),
          el("td", { class: "num", text: (r.started || "").replace("T", " ") }),
          el("td", { text: what, style: { textAlign: "left" } }),
          el("td", { text: [r.has_history ? "history" : null, r.has_console ? "console" : null]
            .filter(Boolean).join(", ") || "–" }),
          el("td", {}, [el("button", { class: "btn sm", text: "Open",
            onclick: () => openRun(r.run_id) })]),
        ]));
      });
      table.appendChild(body);
    } catch (e) { /* leave the table as it was */ }
  }

  async function openRun(id) {
    try {
      const data = await api("/api/run?id=" + encodeURIComponent(id));
      const panel = $("#run-detail-panel");
      panel.classList.remove("hidden");
      $("#run-detail-title").textContent = id;
      $("#run-detail-cmd").textContent = (data.manifest.argv || []).join(" ");
      const body = $("#run-detail-body");
      body.innerHTML = "";

      const overrides = data.manifest.overrides || [];
      if (overrides.length) {
        body.appendChild(el("h3", { class: "hint", text: "Config overrides" }));
        body.appendChild(el("div", { class: "report", text: overrides.join("\n") }));
      }
      if (data.history) {
        body.appendChild(el("h3", { class: "hint mt", text: "Saved per-round history" }));
        const h = data.history;
        body.appendChild(el("div", { class: "hint", text:
          `${h.measured_rounds ?? "?"} measured of ${h.requested_rounds ?? "?"} requested · ` +
          `attacks: ${(h.attacks || []).join(", ")} · defenses: ${(h.defenses || []).join(", ")} · ` +
          `baseline ${pct(h.baseline_accuracy, 2)}` }));
        body.appendChild(renderSavedHistory(h));
      }
      body.appendChild(el("h3", { class: "hint mt", text: "Console" }));
      body.appendChild(el("div", { class: "console", text: data.console || "(empty)" }));
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { toast(e.message, "error"); }
  }

  /** Rebuild the summary matrix from a saved history.json -- the same numbers the
   *  live run showed, recomputed from the per-round records the CLI persisted. */
  function renderSavedHistory(h) {
    const nested = h.history && !Array.isArray(Object.values(h.history)[0]);
    const panels = nested ? h.history : { llm: h.history };
    const base = h.baseline_accuracy;
    const table = el("table", { class: "grid" });
    table.appendChild(el("thead", {}, [el("tr", {},
      ["attack", "defense", "rounds", "mean acc", "acc drop", "detection", "FPR", "goal"]
        .map((x) => el("th", { text: x })))]));
    const body = el("tbody");
    Object.keys(panels).forEach((attack) => {
      const byDefense = panels[attack];
      Object.keys(byDefense).forEach((defense) => {
        const rows = byDefense[defense] || [];
        if (!rows.length) return;
        const acc = mean(rows.map((r) => r.accuracy));
        const tp = rows.reduce((a, r) => a + r.tp, 0);
        const fn = rows.reduce((a, r) => a + r.fn, 0);
        const fp = rows.reduce((a, r) => a + r.fp, 0);
        const tn = rows.reduce((a, r) => a + r.tn, 0);
        const goal = mean(rows.map((r) => r.goal_success).filter((x) => typeof x === "number"));
        body.appendChild(el("tr", { class: attack === "llm" ? "best" : "" }, [
          el("td", { text: attack }),
          el("td", { text: defense, style: { textAlign: "left" } }),
          el("td", { class: "num", text: rows.length }),
          el("td", { class: "num", text: pct(acc, 2) }),
          el("td", { class: "num", text: signed(base - acc, 4) }),
          el("td", { class: "num", text: pct(tp + fn ? tp / (tp + fn) : null, 0) }),
          el("td", { class: "num", text: pct(fp + tn ? fp / (fp + tn) : null, 0) }),
          el("td", { class: "num", text: goal === null ? "n/a" : pct(goal, 0) }),
        ]));
      });
    });
    table.appendChild(body);
    return el("div", { class: "scroll-x mt" }, [table]);
  }

  /* ------------------------------------------------------------------ */
  /* Starting runs                                                       */
  /* ------------------------------------------------------------------ */
  function numeric(id) {
    const v = $(id).value.trim();
    return v === "" ? undefined : Number(v);
  }

  async function startTraining() {
    const payload = {
      // The panel starts GRPO training runs only. --dry-run (frozen LLM, no
      // training) and --baseline (best-of-N reward harness, no LLM) are still
      // accepted by /api/train/start and by main.py; they are diagnostics, and
      // offering them here made the first control on the page a choice between
      // "train" and two things that train nothing.
      mode: "train",
      env: $("#train-env").value,
      rounds: numeric("#train-rounds"),
      poisoners: numeric("#train-poisoners"),
      learn: $("#train-learn").value || undefined,
      fresh: $("#train-fresh").checked,
      debug: $("#train-debug").checked,
      overrides: state.overrides,
    };
    try {
      const status = await api("/api/train/start", payload);
      applyStatus("train", status);
    } catch (e) {
      toast(e.message, "error");
      appendLog("train", "could not start: " + e.message, "error");
    }
  }

  /** Start a benchmark aimed at ONE side.
   *
   *  Both buttons run the same `benchmark.run_benchmark` over the same panel; the
   *  target is which adapter is under test, and it decides three things: which
   *  version axis may be swept, which slice of the matrix the comparison scores
   *  (the `llm` row vs the `llm_defender` column), and which way "best" points --
   *  an attacker wants the drop it caused to be large, a defender wants the drop
   *  it allowed to be small.
   *
   *  A defender run needs the column it is testing to be in the panel, so it is
   *  added here rather than sent behind the panel's back: the chips are what the
   *  page claims it ran, and the printed argv has to match them. */
  async function startBenchmark(target) {
    target = target === "defender" ? "defender" : "attacker";
    if (target === "defender" && !benchSelection.defenses.has("llm_defender")) {
      benchSelection.defenses.add("llm_defender");
      renderChips("#bench-defenses", state.boot.defenses.available,
                  benchSelection.defenses, DEFENSE_NOTE);
      renderFocusSelects();
    }
    const goalType = $("#bench-goal-type").value;
    const goalValue = $("#bench-goal-value").value.trim();
    const payload = {
      target,
      versions: Array.from(versionSelection),
      defender_versions: Array.from(defenderSelection),
      rounds: numeric("#bench-rounds"),
      goal: goalValue === "" ? goalType : `${goalType}=${goalValue}`,
      max_poison_clients: numeric("#bench-poison"),
      n_clients: numeric("#bench-nclients"),
      baseline_knowledge: $("#bench-knowledge").value,
      device: $("#bench-device").value || undefined,
      attack_temperature: numeric("#bench-temp"),
      seed: numeric("#bench-seed"),
      attacks: Array.from(benchSelection.attacks),
      defenses: Array.from(benchSelection.defenses),
      overrides: state.overrides,
    };
    BENCH_ADVANCED.forEach(([key, , type]) => {
      const node = $("#adv-" + key);
      if (!node) return;
      const v = node.value.trim();
      if (v !== "") payload[key] = type === "number" ? Number(v) : v;
    });
    ["no_eval_cache", "benign_retrain", "no_plot", "fresh"].forEach((key) => {
      const node = $("#adv-" + key);
      if (node && node.checked) payload[key] = true;
    });
    try {
      const status = await api("/api/bench/start", payload);
      applyStatus("bench", status);
    } catch (e) {
      toast(e.message, "error");
      appendLog("bench", "could not start: " + e.message, "error");
    }
  }

  async function stopRun(which) {
    const force = state[which].status && state[which].status.state === "stopping";
    try {
      const status = await api(`/api/${which === "train" ? "train" : "bench"}/stop`, { force });
      applyStatus(which, status);
      if (force) toast("Killing the process tree", "error");
    } catch (e) { toast(e.message, "error"); }
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */
  async function boot() {
    state.boot = await api("/api/bootstrap");
    $("#repo-path").textContent = state.boot.repo;
    $("#python-path").textContent = state.boot.python;

    buildTrainCharts();
    renderConfig();
    renderVersions();
    renderVersionSelect();
    renderBenchAdvanced();

    state.boot.attacks.default.forEach((a) => benchSelection.attacks.add(a));
    ["fedavg", "fltrust", "defl", "dnc", "multikrum"].forEach((d) => {
      if (state.boot.defenses.available.includes(d)) benchSelection.defenses.add(d);
    });
    renderChips("#bench-attacks", state.boot.attacks.available,
                benchSelection.attacks, ATTACK_NOTE, "atk");
    renderChips("#bench-defenses", state.boot.defenses.available,
                benchSelection.defenses, DEFENSE_NOTE);
    renderFocusSelects();

    // Prefill from the config so the benchmark form shows the shipped defaults.
    const f = state.boot.config.fields;
    if (f["attack.eval_poison_clients"]) {
      $("#bench-poison").placeholder = "config: " + f["attack.eval_poison_clients"].value;
    }
    if (f["fl.n_clients"]) $("#bench-nclients").placeholder = "config: " + f["fl.n_clients"].value;
    if (f["fl.poison_seed"]) $("#bench-seed").placeholder = "config: " + f["fl.poison_seed"].value;
    if (f["attack.goal.target_accuracy_drop"]) {
      $("#bench-goal-value").value = f["attack.goal.target_accuracy_drop"].value;
    }
    renderLearnWarning();
    applyStatus("train", state.boot.train);
    applyStatus("bench", state.boot.bench);
    loadRuns();

    poll("train");
    poll("bench");
  }

  /* ------------------------------------------------------------------ */
  /* Wiring                                                              */
  /* ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", () => {
    const VIEWS = ["train", "versions", "bench", "runs"];
    const routed = () => {
      const want = (location.hash || "").slice(1);
      return VIEWS.includes(want) ? want : "train";
    };
    $$("#nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
    window.addEventListener("hashchange", () => showView(routed()));
    showView(routed());

    $("#train-start").addEventListener("click", startTraining);
    $("#train-stop").addEventListener("click", () => stopRun("train"));
    $("#bench-start").addEventListener("click", () => startBenchmark("attacker"));
    $("#bench-start-defender").addEventListener("click", () => startBenchmark("defender"));
    $("#bench-stop").addEventListener("click", () => stopRun("bench"));

    $("#train-learn").addEventListener("change", renderLearnWarning);
    $("#train-learn-fix").addEventListener("click", () => {
      state.overrides["defense.mode"] = "llm";
      renderConfig();
      renderLearnWarning();
      toast("defense.mode: llm — the defender LLM defends and can be trained", "ok");
    });

    $("#cfg-search").addEventListener("input", renderConfig);
    $("#cfg-primary-only").addEventListener("change", renderConfig);
    $("#cfg-reset").addEventListener("click", () => {
      state.overrides = {};
      renderConfig();
      renderLearnWarning();
      toast("Config overrides cleared — back to base.yaml", "ok");
    });

    $("#train-clear-log").addEventListener("click", () => {
      $("#train-console").innerHTML = "<span class=\"empty\">cleared</span>";
    });
    $("#bench-clear-log").addEventListener("click", () => {
      $("#bench-console").innerHTML = "<span class=\"empty\">cleared</span>";
    });

    $("#ver-save").addEventListener("click", async () => {
      const label = $("#ver-label").value.trim();
      const notes = $("#ver-notes").value.trim();
      const model = (state.boot.config.fields["rl.model"] || {}).value || "";
      try {
        const data = await api("/api/versions", { label, notes, base_model: model });
        $("#ver-label").value = "";
        $("#ver-notes").value = "";
        state.boot.versions = data.versions;
        await refreshVersions();
        toast("Saved version " + data.version.id, "ok");
      } catch (e) { toast(e.message, "error"); }
    });

    $("#bench-focus-attack").addEventListener("change", () => {
      state.bench.focusAttack = $("#bench-focus-attack").value;
      buildBenchChart();
      // Replay the accuracy series for the newly focused attack.
      state.bench.rounds.forEach((ev) => {
        const values = {};
        (ev.cells || []).forEach((c) => {
          if (c.attack === state.bench.focusAttack) values[c.defense] = c.accuracy;
        });
        state.bench.charts.acc.push(ev.round_num, values);
      });
      state.bench.charts.acc.draw();
      if (state.bench.lastRound) renderBenchRound(state.bench.lastRound);
      renderRoundStrip();
    });
    $("#bench-focus-defense").addEventListener("change", () => {
      state.bench.focusDefense = $("#bench-focus-defense").value;
      if (state.bench.lastRound) renderBenchRound(state.bench.lastRound);
      renderRoundStrip();
    });
    $("#bench-heat-metric").addEventListener("change", renderHeat);
    $("#runs-refresh").addEventListener("click", loadRuns);
    $("#run-detail-close").addEventListener("click", () =>
      $("#run-detail-panel").classList.add("hidden"));

    boot().catch((e) => {
      toast("Could not reach the server: " + e.message, "error");
      document.body.insertBefore(el("div", {
        class: "toast error", style: { margin: "20px" },
        text: "Could not load /api/bootstrap: " + e.message,
      }), document.body.firstChild);
    });
  });
})();
