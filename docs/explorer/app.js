/* CUHK-X Observatory — data-driven explorer for the CUHK-S subset */
(() => {
"use strict";

const DATA = "data/";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const MODC = { depth: "#2ee6c8", ir: "#ff6a3c", thermal: "#ffc23d", rgb: "#e8eef4",
               radar: "#79f08e", imu: "#b88cff", skeleton: "#58a6ff" };

const fmt = n => n.toLocaleString("en-US");
const pad = n => String(n).padStart(2, "0");
const tc = s => `${pad(Math.floor(s / 60))}:${pad(Math.floor(s % 60))}.${Math.floor((s % 1) * 10)}`;
const mmss = s => `${pad(Math.floor(s / 60))}:${pad(Math.floor(s % 60))}`;

/* ---------------------------------------------------- synced video group */
class SyncGroup {
  constructor(videos, { onTick, loop = true } = {}) {
    this.v = videos.filter(Boolean);
    this.master = this.v[0];
    this.onTick = onTick;
    this.playing = false;
    this._raf = null;
    this._lastFix = 0;
    if (loop && this.master) {
      this.master.addEventListener("ended", () => { this.seek(0); this.play(); });
    }
  }
  load(srcs) {
    this.pause();
    this.v.forEach((vid, i) => { if (srcs[i]) { vid.src = srcs[i]; vid.load(); } });
  }
  play() {
    this.playing = true;
    this.v.forEach(v => v.play().catch(() => {}));
    this._tick();
  }
  pause() {
    this.playing = false;
    cancelAnimationFrame(this._raf);
    this.v.forEach(v => v.pause());
  }
  toggle() { this.playing ? this.pause() : this.play(); return this.playing; }
  seek(t) {
    this.v.forEach(v => { try { v.currentTime = t; } catch (e) {} });
  }
  _tick() {
    cancelAnimationFrame(this._raf);
    const step = (ts) => {
      if (!this.playing) return;
      const m = this.master;
      if (ts - this._lastFix > 600) {
        this._lastFix = ts;
        for (let i = 1; i < this.v.length; i++) {
          const v = this.v[i];
          if (v.readyState > 1 && Math.abs(v.currentTime - m.currentTime) > 0.18)
            v.currentTime = m.currentTime;
        }
      }
      this.onTick && this.onTick(m.currentTime, m.duration || 0);
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
}

/* ---------------------------------------------------- boot */
let stats = null, manifest = null;

// scroll reveal (shared observer — sections now, injected tiles later)
const ro = new IntersectionObserver(es => es.forEach(en => {
  if (en.isIntersecting) { en.target.classList.add("in"); ro.unobserve(en.target); }
}), { threshold: 0.06 });

async function boot() {
  try {
    const [s, m] = await Promise.all([
      fetch(DATA + "stats.json").then(r => { if (!r.ok) throw 0; return r.json(); }),
      fetch(DATA + "manifest.json").then(r => { if (!r.ok) throw 0; return r.json(); }),
    ]);
    stats = s; manifest = m;
  } catch (e) {
    console.warn("Observatory: data feed unavailable", e);
    offline();
    return;
  }
  // each module initializes only when its page hosts the element
  if ($(".gauge-num")) gauges();
  if ($("#ticker-track")) ticker();
  if ($("#sweep-stage")) sweep();
  if ($("#lab-rail")) lab();
  if ($("#atlas-grid")) atlas();
  if ($("#quiz-video")) quiz();
  if ($("#matrix-canvas")) matrixChart();
  if ($("#dur-harn")) durationCharts();
  if ($("#pace-bars")) paceAndChainCharts();
  if ($("#top-chain")) topChainChart();
  if ($("#inventory")) inventoryChart();
  if ($("#stations")) stationCards();
  benchBars();
  const fb = $("#foot-build");
  if (fb) fb.textContent = `MANIFEST ${stats.generated} · ${stats.source}`;
  const pl = $("#parent-line");
  if (pl) pl.textContent =
    `${fmt(stats.parent.samples)} samples · ${stats.parent.modalities} modalities · ${stats.parent.subjects} subjects`;
}

function offline() {
  const so = $("#sweep-offline"); if (so) so.hidden = false;
  const lc = $("#lab-caption"); if (lc) lc.textContent = "Sample feed unavailable — open via the published site to stream clips.";
  const qa = $("#quiz-action"); if (qa) qa.textContent = "OFFLINE";
  if ($("#ticker-track")) ticker(true);
  benchBars();
}

/* ---------------------------------------------------- stat gauges */
function gauges() {
  const t = stats.totals;
  const vals = { clips: t.clips, sensorHours: t.sensorHours, users: t.users,
                 actions: t.actions, scenes: t.scenes, labelRows: t.labelRows };
  $$(".gauge-num").forEach(el => {
    const target = vals[el.dataset.stat] ?? 0;
    countUp(el, target, el.dataset.stat === "sensorHours" ? "≈%s h" : "%s");
  });
}

function countUp(el, target, tpl) {
  if (REDUCED) { el.textContent = tpl.replace("%s", fmt(target)); return; }
  const t0 = performance.now(), dur = 1400;
  const step = now => {
    const p = Math.min((now - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = tpl.replace("%s", fmt(Math.round(target * eased)));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------------------------------------------- action ticker */
function ticker(fallback) {
  const names = fallback || !stats
    ? ["WASH FACE", "BRUSH TEETH", "FOLD CLOTHES", "DRINK WATER", "DO SQUATS", "WALK", "READ DOCUMENTS"]
    : stats.actions.map(a => a.name.toUpperCase());
  const seq = names.map((n, i) => `<b>${pad(i)}</b> ${n}`).join('<span> · </span>');
  $("#ticker-track").innerHTML = seq + '<span> · </span>' + seq; // doubled for seamless loop
}

/* ---------------------------------------------------- hero spectral sweep */
function sweep() {
  const set = manifest.hauSets.find(s => s.id === manifest.hero.set) || manifest.hauSets[0];
  if (!set || !set.trials.length) { $("#sweep-offline").hidden = false; return; }
  const trial = set.trials.find(t => t.trial === manifest.hero.trial) || set.trials[0];
  $("#sweep-session").textContent = `${set.user.toUpperCase()} · S${set.scene}E${set.env} · ${trial.pace.toUpperCase()}`;

  const stage = $("#sweep-stage");
  const layers = { depth: $('[data-mod="depth"] video', stage),
                   ir: $('[data-mod="ir"] video', stage),
                   thermal: $('[data-mod="thermal"] video', stage) };
  const group = new SyncGroup([layers.depth, layers.ir, layers.thermal], {
    onTick: t => $("#sweep-tc").textContent = tc(t),
  });
  group.load([DATA + trial.clips.depth, DATA + trial.clips.ir, DATA + trial.clips.thermal]);

  const btn = $("#sweep-play");
  btn.addEventListener("click", e => { e.stopPropagation(); btn.textContent = group.toggle() ? "❚❚" : "▶"; });

  // autoplay once visible
  new IntersectionObserver((es, ob) => {
    if (es[0].isIntersecting) { group.play(); btn.textContent = "❚❚"; ob.disconnect(); }
  }, { threshold: 0.35 }).observe(stage);

  // draggable dividers
  const pos = [33.3, 66.6];
  const handles = $$(".sweep-handle", stage);
  const irLayer = $('[data-mod="ir"]', stage), thLayer = $('[data-mod="thermal"]', stage);
  const tagIr = $(".tag-ir", stage), tagTh = $(".tag-thermal", stage);
  const apply = () => {
    irLayer.style.clipPath = `inset(0 0 0 ${pos[0]}%)`;
    thLayer.style.clipPath = `inset(0 0 0 ${pos[1]}%)`;
    handles[0].style.left = pos[0] + "%";
    handles[1].style.left = pos[1] + "%";
    tagIr.style.left = `calc(${pos[0]}% + 12px)`;
    tagTh.style.left = `calc(${pos[1]}% + 12px)`;
    tagIr.style.opacity = pos[1] - pos[0] < 13 ? 0 : 1;
  };
  apply();
  let drag = -1;
  const move = clientX => {
    const r = stage.getBoundingClientRect();
    let p = ((clientX - r.left) / r.width) * 100;
    if (drag === 0) p = Math.min(Math.max(p, 4), pos[1] - 6);
    else p = Math.min(Math.max(p, pos[0] + 6), 96);
    pos[drag] = p; apply();
  };
  handles.forEach((h, i) => {
    h.addEventListener("pointerdown", e => { drag = i; h.setPointerCapture(e.pointerId); e.preventDefault(); });
    h.addEventListener("pointermove", e => { if (drag === i) move(e.clientX); });
    h.addEventListener("pointerup", () => drag = -1);
    h.addEventListener("keydown", e => {
      const d = e.key === "ArrowLeft" ? -2 : e.key === "ArrowRight" ? 2 : 0;
      if (d) { drag = i; move(stage.getBoundingClientRect().left + (pos[i] + d) / 100 * stage.clientWidth); drag = -1; }
    });
  });
  stage.addEventListener("pointerdown", e => {
    if (e.target.closest(".sweep-handle") || e.target.closest(".sweep-play")) return;
    const r = stage.getBoundingClientRect();
    const p = ((e.clientX - r.left) / r.width) * 100;
    drag = Math.abs(p - pos[0]) < Math.abs(p - pos[1]) ? 0 : 1;
    move(e.clientX); drag = -1;
  });
}

/* ---------------------------------------------------- sequence lab */
function lab() {
  const sets = manifest.hauSets.filter(s => s.trials.length);
  if (!sets.length) return;
  const vids = { depth: $('#lab-stage [data-mod="depth"]'),
                 ir: $('#lab-stage [data-mod="ir"]'),
                 thermal: $('#lab-stage [data-mod="thermal"]') };
  let cur = { set: 0, trial: 0 };
  let typeTimer = null;

  const group = new SyncGroup([vids.depth, vids.ir, vids.thermal], {
    onTick: (t, d) => {
      $("#lab-fill").style.width = d ? (t / d * 100) + "%" : "0%";
      $("#lab-time").textContent = `${mmss(t)} / ${mmss(d || 0)}`;
    },
  });

  // set rail
  const rail = $("#lab-rail");
  rail.innerHTML = sets.map((s, i) => `
    <button class="set-card${i === 0 ? " active" : ""}" data-i="${i}">
      <span class="sc-id">SCENE ${s.scene} · ${s.user.toUpperCase()}</span>
      <span class="sc-sub">ENV ${s.env} · 3 PACES · ${Math.round(s.trials[0].dur)}s · ×3 MODALITIES</span>
    </button>`).join("");
  rail.addEventListener("click", e => {
    const b = e.target.closest(".set-card"); if (!b) return;
    $$(".set-card", rail).forEach(x => x.classList.toggle("active", x === b));
    cur.set = +b.dataset.i; cur.trial = 0;
    renderTabs(); loadTrial(true);
  });

  const PACE_COLOR = { Leisurely: MODC.depth, Calmly: MODC.skeleton, Hastily: MODC.ir };
  function renderTabs() {
    const trials = sets[cur.set].trials;
    $("#pace-tabs").innerHTML = trials.map((t, i) =>
      `<button class="pace-tab${i === cur.trial ? " active" : ""}" data-i="${i}" role="tab">${(t.pace || "TAKE " + t.trial).toUpperCase()}</button>`).join("");
  }
  $("#pace-tabs").addEventListener("click", e => {
    const b = e.target.closest(".pace-tab"); if (!b) return;
    cur.trial = +b.dataset.i;
    $$(".pace-tab").forEach(x => x.classList.toggle("active", x === b));
    loadTrial(true);
  });

  function typewriter(el, text) {
    clearInterval(typeTimer);
    if (REDUCED) { el.textContent = text; return; }
    el.innerHTML = '<span class="cursor"></span>';
    let i = 0;
    typeTimer = setInterval(() => {
      i = Math.min(i + 3, text.length);
      el.innerHTML = text.slice(0, i) + (i < text.length ? '<span class="cursor"></span>' : "");
      if (i >= text.length) clearInterval(typeTimer);
    }, 24);
  }

  function loadTrial(autoplay) {
    const s = sets[cur.set], t = s.trials[cur.trial];
    group.load([DATA + t.clips.depth, DATA + t.clips.ir, DATA + t.clips.thermal]);
    $("#lab-meta").textContent = `SESSION ${s.user.toUpperCase()}/${t.sess} · ${t.dur}s · PACE: ${t.pace.toUpperCase()}`;
    typewriter($("#lab-caption"), t.caption);
    $("#lab-chain").innerHTML = t.chain.map((c, i) =>
      `${i ? '<span class="chain-sep">▸</span>' : ""}<span class="chain-step" style="animation-delay:${0.15 + i * 0.12}s">${c.toUpperCase()}</span>`).join("");
    $("#lab-play").textContent = autoplay ? "❚❚" : "▶";
    if (autoplay) group.play();
  }

  $("#lab-play").addEventListener("click", () => {
    $("#lab-play").textContent = group.toggle() ? "❚❚" : "▶";
  });
  $("#lab-rail-bar").addEventListener("click", e => {
    const r = e.currentTarget.getBoundingClientRect();
    const d = group.master.duration || 0;
    group.seek((e.clientX - r.left) / r.width * d);
  });

  renderTabs();
  // lazy start when scrolled into view
  new IntersectionObserver((es, ob) => {
    if (es[0].isIntersecting) { loadTrial(true); ob.disconnect(); }
  }, { threshold: 0.25 }).observe($("#sequence-lab"));
  loadTrial(false);
}

/* ---------------------------------------------------- action atlas */
function atlas() {
  const grid = $("#atlas-grid");
  $("#atlas-line").textContent = `${manifest.atlas.length} classes of everyday actions`;
  const tile = (a, i) => {
    const bs = `background-size:${a.frames * 100}% 100%`;
    const label = `<span class="tile-label"><span class="tl-name">${a.name}</span><span class="tl-meta">${a.dur}s · ${a.user.toUpperCase()}</span></span>`;
    if (a.stripIr) {
      return `
    <button class="tile dual reveal" data-i="${i}" style="transition-delay:${(i % 8) * 40}ms">
      <span class="tile-view is-ir" style="background-image:url('${DATA + a.stripIr}');${bs}"></span>
      <span class="tile-view is-depth pip" style="background-image:url('${DATA + a.strip}');${bs}"></span>
      <span class="tile-mod m-ir">INFRARED ⊕ DEPTH</span>
      <span class="tile-scan"></span>
      <span class="tl-id">${pad(a.id)}</span>
      ${label}
    </button>`;
    }
    return `
    <button class="tile reveal" data-i="${i}" style="transition-delay:${(i % 8) * 40}ms">
      <span class="tile-view is-depth" style="background-image:url('${DATA + a.strip}');${bs}"></span>
      <span class="tile-scan"></span>
      <span class="tl-id">${pad(a.id)}</span>
      ${label}
    </button>`;
  };
  let html = "", lastCat = null;
  manifest.atlas.forEach((a, i) => {
    if (a.category && a.category !== lastCat) {
      lastCat = a.category;
      const n = manifest.atlas.filter(x => x.category === a.category).length;
      html += `<div class="atlas-cat"><h3>${a.category}</h3><span>${n} ACTIONS</span></div>`;
    }
    html += tile(a, i);
  });
  grid.innerHTML = html;

  $$(".tile", grid).forEach(t => ro.observe(t));

  grid.addEventListener("pointermove", e => {
    const t = e.target.closest(".tile"); if (!t) return;
    const a = manifest.atlas[+t.dataset.i];
    if (!a || a.frames < 2) return;
    const r = t.getBoundingClientRect();
    const idx = Math.max(0, Math.min(a.frames - 1, Math.floor((e.clientX - r.left) / r.width * a.frames)));
    const pos = `${(idx / (a.frames - 1)) * 100}% 0`;
    $$(".tile-view", t).forEach(v => v.style.backgroundPosition = pos);
  });
  grid.addEventListener("click", e => {
    const t = e.target.closest(".tile"); if (!t) return;
    inspect(manifest.atlas[+t.dataset.i]);
  });
}

/* ---------------------------------------------------- inspector modal */
let inspGroup = null;
function inspect(a) {
  const box = $("#inspector");
  box.hidden = false;
  document.body.style.overflow = "hidden";
  $("#insp-title").textContent = `${pad(a.id)} · ${a.name.toUpperCase()} — DEPTH ⊕ INFRARED`;
  const vd = $('#inspector [data-mod="depth"]'), vi = $('#inspector [data-mod="ir"]');
  vd.poster = vi.poster = DATA + a.poster;
  inspGroup = new SyncGroup([vd, vi]);
  inspGroup.load([DATA + a.clips.depth, DATA + a.clips.ir]);
  inspGroup.play();
  $("#insp-meta").innerHTML = [
    `CLASS <b>${pad(a.id)}</b>`,
    a.category ? `CATEGORY <b>${a.category.toUpperCase()}</b>` : "",
    `SUBJECT <b>${a.user.toUpperCase()}</b>`,
    `SESSION <b>${a.sess}</b>`, `LENGTH <b>${a.dur}s</b>`,
    a.logic ? `NEXT-ACTION GT <b>${a.logic.next.toUpperCase()}</b>` : `NEXT-ACTION GT <b>—</b>`,
  ].filter(Boolean).map(x => `<span>${x}</span>`).join("");
}
function closeInspector() {
  const box = $("#inspector");
  if (!box) return;
  if (inspGroup) inspGroup.pause();
  box.hidden = true;
  document.body.style.overflow = "";
}
document.addEventListener("click", e => {
  if (e.target.id === "insp-close" || e.target.id === "inspector") closeInspector();
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeInspector(); });

/* ---------------------------------------------------- reasoning quiz */
function quiz() {
  const qs = manifest.quiz;
  if (!qs.length) return;
  let i = 0, score = 0, answered = 0;
  const vid = $("#quiz-video");

  function load() {
    const q = qs[i];
    vid.src = DATA + q.clip; vid.poster = DATA + q.poster;
    vid.play().catch(() => {});
    $("#quiz-progress").textContent = `ROUND ${i + 1} / ${qs.length} · DEPTH STREAM ONLY`;
    $("#quiz-action").textContent = q.action.toUpperCase();
    $("#quiz-verdict").textContent = ""; $("#quiz-verdict").className = "quiz-verdict";
    $("#quiz-options").innerHTML = q.candidates.map((c, k) =>
      `<button class="quiz-opt" data-k="${k}">▹ ${c.toUpperCase()}</button>`).join("");
  }
  $("#quiz-options").addEventListener("click", e => {
    const b = e.target.closest(".quiz-opt"); if (!b || b.disabled) return;
    const q = qs[i];
    const isAns = k => q.candidates[k].toUpperCase() === q.answer.toUpperCase();
    const correct = isAns(+b.dataset.k);
    $$(".quiz-opt").forEach(x => {
      x.disabled = true;
      if (isAns(+x.dataset.k)) x.classList.add("correct");
    });
    if (!correct) b.classList.add("wrong");
    answered++; if (correct) score++;
    const v = $("#quiz-verdict");
    v.textContent = correct ? "✓ MATCH — GROUND TRUTH AGREES." : `✗ GROUND TRUTH: ${q.answer.toUpperCase()}`;
    v.classList.add(correct ? "good" : "nope");
    $("#quiz-score").textContent = `SCORE ${score}/${answered}`;
  });
  $("#quiz-next").addEventListener("click", () => { i = (i + 1) % qs.length; load(); });
  new IntersectionObserver((es, ob) => {
    if (es[0].isIntersecting) { load(); ob.disconnect(); }
  }, { threshold: 0.3 }).observe($("#console"));
}

/* ---------------------------------------------------- charts */
function barRow(label, val, max, color, valText) {
  return `<div class="bar-row" style="--bc:${color}">
    <span class="br-label" title="${label}">${label}</span>
    <span class="br-track"><span class="br-fill" data-w="${max ? (val / max * 100) : 0}"></span></span>
    <span class="br-val">${valText ?? fmt(val)}</span></div>`;
}
function animateBars(root) {
  new IntersectionObserver((es, ob) => {
    if (!es[0].isIntersecting) return;
    $$(".br-fill", root).forEach((f, i) => setTimeout(() => f.style.width = f.dataset.w + "%", i * 45));
    ob.disconnect();
  }, { threshold: 0.2 }).observe(root);
}

function durationCharts() {
  const d = stats.durations;
  const dm = $("#dur-meta");
  if (dm) dm.textContent = `${d.harn.sampled + d.hau.sampled} clips sampled`;
  const mk = (h, title) => {
    const max = Math.max(...h.hist) || 1;
    return `<div><p class="chart-title">${title}</p>` + h.hist.map((v, i) => {
      const lo = h.bins[i], hi = h.bins[i + 1];
      return barRow(hi ? `${lo}–${hi}s` : `${lo}s+`, v, max, "var(--depth)");
    }).join("") + "</div>";
  };
  $("#dur-harn").innerHTML = mk(d.harn, `SINGLE ACTIONS · MEAN ${d.harn.mean}s`);
  $("#dur-hau").innerHTML = mk(d.hau, `SEQUENTIAL SCENES · MEAN ${d.hau.mean}s`);
  animateBars($("#dur-harn").closest(".instr"));
}

function paceAndChainCharts() {
  // pace — dataset uses ~50 adverb labels; show the dominant ones
  const em = Object.entries(stats.emotions).sort((a, b) => b[1] - a[1]);
  const top = em.slice(0, 8), rest = em.slice(8).reduce((n, [, v]) => n + v, 0);
  const emax = top.length ? top[0][1] : 1;
  const PAL = [MODC.depth, MODC.skeleton, MODC.imu, MODC.thermal, MODC.ir, MODC.radar, MODC.rgb, MODC.depth];
  $("#pace-bars").innerHTML = top.map(([k, v], i) => barRow(k.toUpperCase(), v, emax, PAL[i]))
    .join("") + barRow(`+${em.length - 8} OTHERS`, rest, emax, "var(--faint)");

  const cl = stats.chainLengths, clmax = Math.max(...cl.map(x => x[1])) || 1;
  $("#chainlen-bars").innerHTML = cl.map(([len, n]) => barRow(`${len} STEPS`, n, clmax, "var(--thermal)")).join("");
  animateBars($("#pace-bars").closest(".instr"));
}

function topChainChart() {
  const ca = stats.chainActions.slice(0, 12), cmax = ca.length ? ca[0].count : 1;
  $("#top-chain").innerHTML = ca.map(c => barRow(c.name.toUpperCase(), c.count, cmax, "var(--imu)")).join("");
  animateBars($("#top-chain"));
}

function inventoryChart() {
  const b = stats.benchmarks;
  const rows = [
    ["HARN · DEPTH", b.HARn.modalities.Depth, MODC.depth],
    ["HARN · IR", b.HARn.modalities.IR, MODC.ir],
    ["HAU · DEPTH", b.HAU.modalities.Depth, MODC.depth],
    ["HAU · IR", b.HAU.modalities.IR, MODC.ir],
    ["HAU · THERMAL", b.HAU.modalities.Thermal, MODC.thermal],
  ];
  const imax = Math.max(...rows.map(r => r[1]));
  $("#inventory").innerHTML = rows.map(([l, v, c]) => barRow(l, v, imax, c)).join("");
  const im = $("#inv-meta");
  if (im) im.textContent = `${fmt(stats.totals.clips)} clips · ${stats.totals.sizeGB} GB compressed · 320×240`;
  animateBars($("#inventory"));
}

function stationCards() {
  const b = stats.benchmarks;
  const map = {
    lab: `${fmt(b.HAU.sequences)} TAKES · ${b.HAU.scenes} SCENES · 3 SYNCED STREAMS`,
    atlas: `${b.HARn.actionClasses} CLASSES · ${fmt(b.HARn.clips)} CLIPS · DEPTH ⊕ IR`,
    console: `${fmt(b.HARn.logicLabels)} LOGIC LABELS · BEAT THE VLM BASELINE`,
  };
  $$("#stations [data-fill]").forEach(el => {
    if (map[el.dataset.fill]) el.textContent = map[el.dataset.fill];
  });
}

function matrixChart() {
  const cv = $("#matrix-canvas"), tip = $("#matrix-tip");
  const M = stats.matrix;
  const dpr = devicePixelRatio || 1;

  function draw() {
    if (!M.users.length || !M.actions.length) return;
    const W = cv.clientWidth - 30, H = 300;
    cv.width = cv.clientWidth * dpr; cv.height = (H + 40) * dpr;
    cv.style.height = (H + 40) + "px";
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    const left = 56, top = 8;
    const cw = (W - left) / M.actions.length, ch = (H - top) / M.users.length;
    let max = 0;
    M.counts.forEach(r => r.forEach(v => max = Math.max(max, v)));
    max = max || 1;
    ctx.font = "9px 'IBM Plex Mono', monospace";
    M.users.forEach((u, y) => {
      ctx.fillStyle = "#5f7689";
      ctx.textAlign = "right";
      if (M.users.length <= 24 || y % 2 === 0) ctx.fillText(u.replace("user", "U"), left - 6, top + y * ch + ch * 0.7);
      M.counts[y].forEach((v, x) => {
        if (!v) { ctx.fillStyle = "rgba(22,34,47,0.55)"; }
        else {
          const a = 0.12 + 0.88 * Math.sqrt(v / max);
          ctx.fillStyle = `rgba(46, 230, 200, ${a.toFixed(3)})`;
        }
        ctx.fillRect(left + x * cw, top + y * ch, Math.max(cw - 1.5, 1), Math.max(ch - 1.5, 1));
      });
    });
    ctx.fillStyle = "#5f7689"; ctx.textAlign = "center";
    for (let x = 0; x < M.actions.length; x += 4)
      ctx.fillText(pad(M.actions[x]), left + x * cw + cw / 2, H + 22);
    cv._geom = { left, top, cw, ch, max };
  }
  draw();
  addEventListener("resize", draw);

  cv.addEventListener("pointermove", e => {
    const g = cv._geom; if (!g) return;
    const r = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left - g.left) / g.cw);
    const y = Math.floor((e.clientY - r.top - g.top) / g.ch);
    if (x < 0 || y < 0 || x >= stats.matrix.actions.length || y >= stats.matrix.users.length) { tip.hidden = true; return; }
    const action = stats.actions.find(a => a.id === stats.matrix.actions[x]);
    tip.hidden = false;
    tip.textContent = `${stats.matrix.users[y].toUpperCase()} × ${action ? action.name.toUpperCase() : x} — ${stats.matrix.counts[y][x]} CLIPS`;
    tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY - 10) + "px";
  });
  cv.addEventListener("pointerleave", () => tip.hidden = true);
  $("#matrix-meta").textContent = `${stats.matrix.users.length} subjects × ${stats.matrix.actions.length} actions · depth clips`;
}

/* ---------------------------------------------------- benchmark bars */
function benchBars() {
  const root = $("#bench-har");
  if (!root) return;
  const HAR = [["THERMAL", 92.57, MODC.thermal], ["RGB", 90.89, MODC.rgb],
               ["DEPTH", 90.46, MODC.depth], ["IR", 90.22, MODC.ir],
               ["SKELETON", 79.08, MODC.skeleton], ["RADAR", 46.63, MODC.radar],
               ["IMU", 45.52, MODC.imu]];
  root.innerHTML = HAR.map(([l, v, c]) => barRow(l, v, 100, c, v.toFixed(1) + "%")).join("");
  animateBars(root);
}

/* ---------------------------------------------------- misc */
const copyBtn = $("#copy-bib");
if (copyBtn) copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#bibtex").textContent);
    copyBtn.textContent = "COPIED ✓";
    setTimeout(() => copyBtn.textContent = "COPY", 1500);
  } catch (e) {}
});

// scroll reveal for blocks
$$(".block, .instr, .gauge").forEach(el => { el.classList.add("reveal"); ro.observe(el); });

boot();
})();
