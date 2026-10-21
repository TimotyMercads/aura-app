/*
 * aura-viz.js - switchable visualization panels for AURA.
 *
 * Every element with class "viz" and a data-viz JSON config (built by
 * utils/viz.py, placed by templates/_viz.html) becomes a panel with:
 *   - a view switch:   Chart | Statistics | Both
 *   - a chart-type dropdown (only the types that make sense for that data)
 *   - the chart (Chart.js) and/or a statistics table of the exact values
 * A page-level switch (.viz-global) changes every panel on the page at once.
 * Choices are remembered per panel in the viewer's browser (if allowed).
 */
(function () {
    "use strict";

    var SURFACE = "#1B3358";
    var INK = "#D7DEEA", MUTED = "#93A1BC", GRID = "rgba(255,255,255,.07)";
    var TYPE_NAMES = {
        bar: "Bar", hbar: "Horizontal bar", doughnut: "Doughnut", pie: "Pie", polarArea: "Polar area",
        line: "Line", radar: "Radar", stacked: "Stacked bar", grouped: "Grouped bar",
        hstacked: "Horizontal stacked bar", hgrouped: "Horizontal grouped bar"
    };
    var VIEWS = [["chart", "📊 Chart"], ["stats", "🔢 Statistics"], ["both", "▦ Both"]];
    var panels = [];

    // ---------- small helpers ----------
    function store(key, value) {
        try {
            if (value === undefined) { return JSON.parse(window.localStorage.getItem("aura-viz:" + key) || "null"); }
            window.localStorage.setItem("aura-viz:" + key, JSON.stringify(value));
        } catch (e) { return null; }
        return null;
    }
    function el(tag, cls, text) {
        var n = document.createElement(tag);
        if (cls) { n.className = cls; }
        if (text !== undefined && text !== null) { n.textContent = text; }
        return n;
    }
    function fmt(cfg, v) {
        if (v === null || v === undefined || isNaN(v)) { return "—"; }
        var d = cfg.decimals || 0;
        var s = Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
        if (cfg.signed && v > 0) { s = "+" + s; }
        return s + (cfg.unit && cfg.unit.trim() === "%" ? "%" : "");
    }
    function unitWord(cfg) { return cfg.unit && cfg.unit.trim() !== "%" ? cfg.unit : ""; }
    function alpha(hex, a) {
        var h = hex.replace("#", "");
        var r = parseInt(h.substr(0, 2), 16), g = parseInt(h.substr(2, 2), 16), b = parseInt(h.substr(4, 2), 16);
        return "rgba(" + r + "," + g + "," + b + "," + a + ")";
    }
    function sum(arr) { return arr.reduce(function (a, b) { return a + (Number(b) || 0); }, 0); }

    // ---------- error whiskers (± SD) for bar charts ----------
    var whiskers = {
        id: "auraWhiskers",
        afterDatasetsDraw: function (chart) {
            var ctx = chart.ctx;
            chart.data.datasets.forEach(function (ds, di) {
                if (!ds.errors) { return; }
                var meta = chart.getDatasetMeta(di);
                if (meta.type !== "bar" || meta.hidden) { return; }
                var horizontal = chart.options.indexAxis === "y";
                var vScale = horizontal ? chart.scales.x : chart.scales.y;
                ctx.save();
                ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
                meta.data.forEach(function (bar, i) {
                    var v = ds.data[i], e = ds.errors[i];
                    if (v === null || e === null || e === undefined) { return; }
                    var lo = vScale.getPixelForValue(Math.max(v - e, vScale.min));
                    var hi = vScale.getPixelForValue(Math.min(v + e, vScale.max));
                    var c = horizontal ? bar.y : bar.x, cap = 5;
                    ctx.beginPath();
                    if (horizontal) {
                        ctx.moveTo(lo, c); ctx.lineTo(hi, c);
                        ctx.moveTo(lo, c - cap); ctx.lineTo(lo, c + cap);
                        ctx.moveTo(hi, c - cap); ctx.lineTo(hi, c + cap);
                    } else {
                        ctx.moveTo(c, lo); ctx.lineTo(c, hi);
                        ctx.moveTo(c - cap, lo); ctx.lineTo(c + cap, lo);
                        ctx.moveTo(c - cap, hi); ctx.lineTo(c + cap, hi);
                    }
                    ctx.stroke();
                });
                ctx.restore();
            });
        }
    };

    // ---------- chart ----------
    function buildChart(p) {
        var cfg = p.cfg, type = p.type;
        if (p.chart) { p.chart.destroy(); p.chart = null; }
        if (typeof window.Chart === "undefined") { return; }

        var multi = cfg.series.length > 1;
        var round = ["doughnut", "pie", "polarArea"].indexOf(type) >= 0;
        var horizontal = ["hbar", "hstacked", "hgrouped"].indexOf(type) >= 0;
        var stacked = ["stacked", "hstacked"].indexOf(type) >= 0;
        var cjsType = round ? type : (type === "line" ? "line" : (type === "radar" ? "radar" : "bar"));

        var datasets = cfg.series.map(function (s) {
            var colors = s.colors || cfg.labels.map(function () { return s.color; });
            var ds = { label: s.name, data: s.data, errors: s.errors || null };
            if (round) {
                ds.backgroundColor = colors; ds.borderColor = SURFACE; ds.borderWidth = 2; ds.hoverOffset = 6;
            } else if (cjsType === "line") {
                ds.borderColor = s.color; ds.backgroundColor = s.color; ds.borderWidth = 2;
                ds.pointBackgroundColor = colors; ds.pointBorderColor = SURFACE; ds.pointBorderWidth = 2;
                ds.pointRadius = 5; ds.pointHoverRadius = 7; ds.tension = 0.25; ds.fill = false;
            } else if (cjsType === "radar") {
                ds.borderColor = s.color; ds.backgroundColor = alpha(s.color, multi ? 0.12 : 0.22);
                ds.borderWidth = 2; ds.pointBackgroundColor = colors; ds.pointBorderColor = SURFACE; ds.pointRadius = 4;
            } else {
                ds.backgroundColor = colors; ds.borderColor = SURFACE;
                ds.borderWidth = stacked ? { top: 2, bottom: 0, left: 0, right: 0 } : 0;
                ds.borderRadius = stacked ? 0 : 4; ds.borderSkipped = cfg.signed ? false : "start";
                ds.maxBarThickness = 46; ds.categoryPercentage = 0.72; ds.barPercentage = multi ? 0.9 : 0.8;
                if (horizontal && stacked) { ds.borderWidth = { right: 2, top: 0, bottom: 0, left: 0 }; }
            }
            return ds;
        });

        var valueAxis = {
            grid: { color: GRID }, border: { display: false }, ticks: { color: MUTED },
            beginAtZero: cfg.min === null || cfg.min === undefined || cfg.signed, stacked: stacked
        };
        if (cfg.max !== null && cfg.max !== undefined && !stacked) { valueAxis.suggestedMax = cfg.max; valueAxis.max = cfg.max; }
        if (cfg.min !== null && cfg.min !== undefined && !cfg.signed && !stacked) { valueAxis.min = cfg.min; }
        if (cfg.signed) { valueAxis.grid = { color: function (c) { return c.tick.value === 0 ? MUTED : GRID; } }; }
        var catAxis = { grid: { display: false }, border: { color: GRID }, ticks: { color: MUTED, autoSkip: false, maxRotation: 40 }, stacked: stacked };

        var totals = cfg.series.map(function (s) { return sum(s.data); });
        var options = {
            maintainAspectRatio: false, responsive: true, animation: { duration: 350 },
            interaction: { mode: round || cjsType === "radar" ? "nearest" : "index", intersect: round },
            plugins: {
                legend: { display: multi || round, position: "bottom", labels: { color: INK, boxWidth: 12, padding: 14, usePointStyle: true } },
                tooltip: {
                    backgroundColor: "#0B1A33", borderColor: "rgba(255,255,255,.12)", borderWidth: 1,
                    titleColor: "#fff", bodyColor: INK, padding: 10,
                    callbacks: {
                        label: function (item) {
                            var s = cfg.series[item.datasetIndex], v = s.data[item.dataIndex];
                            var txt = (multi ? s.name + ": " : (round ? item.label + ": " : "")) + fmt(cfg, v) + unitWord(cfg);
                            if (s.errors && s.errors[item.dataIndex] !== null) { txt += " ± " + fmt(cfg, s.errors[item.dataIndex]); }
                            if (cfg.percent) {
                                var base = multi ? sum(cfg.series.map(function (x) { return x.data[item.dataIndex]; })) : totals[item.datasetIndex];
                                if (base) { txt += "  (" + (100 * v / base).toFixed(1) + "%" + (multi ? " of " + item.label : "") + ")"; }
                            }
                            return txt;
                        }
                    }
                }
            }
        };
        if (!round && cjsType !== "radar") {
            options.indexAxis = horizontal ? "y" : "x";
            options.scales = horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis };
        } else if (cjsType === "radar") {
            options.scales = { r: { angleLines: { color: GRID }, grid: { color: GRID }, pointLabels: { color: INK, font: { size: 11 } },
                ticks: { color: MUTED, backdropColor: "transparent", showLabelBackdrop: false },
                suggestedMin: cfg.min !== null && cfg.min !== undefined ? cfg.min : 0,
                suggestedMax: cfg.max !== null && cfg.max !== undefined ? cfg.max : undefined } };
        } else if (type === "polarArea") {
            options.scales = { r: { grid: { color: GRID }, ticks: { color: MUTED, backdropColor: "transparent" } } };
        }
        p.chart = new window.Chart(p.canvas.getContext("2d"), { type: cjsType, data: { labels: cfg.labels, datasets: datasets }, options: options, plugins: [whiskers] });
    }

    // ---------- statistics ----------
    function buildStats(p) {
        var cfg = p.cfg, box = p.statsBox;
        box.innerHTML = "";
        var multi = cfg.series.length > 1;
        var s0 = cfg.series[0];

        var chips = [];
        (cfg.extra || []).forEach(function (pair) { chips.push(pair); });
        if (!multi && cfg.summary) {
            var vals = s0.data.filter(function (v) { return v !== null; });
            if (vals.length) {
                var mx = Math.max.apply(null, vals), mn = Math.min.apply(null, vals);
                if (cfg.percent) { chips.push(["Total", fmt(cfg, sum(vals)) + unitWord(cfg)]); }
                chips.push(["Highest", cfg.labels[s0.data.indexOf(mx)] + " · " + fmt(cfg, mx)]);
                chips.push(["Lowest", cfg.labels[s0.data.indexOf(mn)] + " · " + fmt(cfg, mn)]);
                if (!cfg.percent) { chips.push(["Average", fmt(Object.assign({}, cfg, { decimals: Math.max(cfg.decimals, 2) }), sum(vals) / vals.length)]); }
            }
        } else if (multi && cfg.percent) {
            chips.push(["Total", fmt(cfg, sum(cfg.series.map(function (s) { return sum(s.data); }))) + unitWord(cfg)]);
        }
        if (chips.length) {
            var row = el("div", "viz-chips");
            chips.forEach(function (c) {
                var chip = el("div", "viz-chip");
                chip.appendChild(el("span", null, c[0])); chip.appendChild(el("strong", null, c[1]));
                row.appendChild(chip);
            });
            box.appendChild(row);
        }

        var wrap = el("div", "table-scroll-x");
        var t = el("table", "columns-table viz-table");
        var thead = el("thead"), hr = el("tr");
        hr.appendChild(el("th", null, ""));
        cfg.series.forEach(function (s) {
            var th = el("th");
            if (multi) { var sw = el("span", "viz-swatch"); sw.style.background = s.color; th.appendChild(sw); }
            th.appendChild(document.createTextNode(multi ? s.name : (cfg.valueLabel || s.name)));
            hr.appendChild(th);
            if (s.errors) { hr.appendChild(el("th", null, "± SD")); }
        });
        if (cfg.percent && !multi) { hr.appendChild(el("th", null, "% of total")); }
        if (cfg.percent && multi) { hr.appendChild(el("th", null, cfg.totalLabel || "Total")); }
        thead.appendChild(hr); t.appendChild(thead);

        var tb = el("tbody"), total0 = sum(s0.data);
        cfg.labels.forEach(function (label, i) {
            var tr = el("tr"), td0 = el("td");
            if (!multi && s0.colors) { var sw = el("span", "viz-swatch"); sw.style.background = s0.colors[i]; td0.appendChild(sw); }
            td0.appendChild(document.createTextNode(label)); tr.appendChild(td0);
            cfg.series.forEach(function (s) {
                var v = s.data[i];
                var cell = el("td", "num", fmt(cfg, v));
                if (multi && cfg.percent) {
                    var rowTotal = sum(cfg.series.map(function (x) { return x.data[i]; }));
                    if (rowTotal) { cell.appendChild(el("em", null, " " + (100 * v / rowTotal).toFixed(0) + "%")); }
                }
                tr.appendChild(cell);
                if (s.errors) { tr.appendChild(el("td", "num", fmt(cfg, s.errors[i]))); }
            });
            if (cfg.percent && !multi) { tr.appendChild(el("td", "num", total0 ? (100 * s0.data[i] / total0).toFixed(1) + "%" : "—")); }
            if (cfg.percent && multi) { tr.appendChild(el("td", "num", fmt(cfg, sum(cfg.series.map(function (x) { return x.data[i]; }))))); }
            tb.appendChild(tr);
        });
        if (cfg.percent) {
            var tr = el("tr", "viz-total-row");
            tr.appendChild(el("td", null, cfg.totalLabel || "Total"));
            cfg.series.forEach(function (s) { tr.appendChild(el("td", "num", fmt(cfg, sum(s.data)))); });
            if (!multi) { tr.appendChild(el("td", "num", "100%")); }
            else { tr.appendChild(el("td", "num", fmt(cfg, sum(cfg.series.map(function (s) { return sum(s.data); }))))); }
            tb.appendChild(tr);
        }
        t.appendChild(tb); wrap.appendChild(t); box.appendChild(wrap);
    }

    // ---------- panel ----------
    function apply(p) {
        p.root.setAttribute("data-view", p.view);
        p.viewButtons.forEach(function (b) { b.classList.toggle("active", b.getAttribute("data-view") === p.view); b.setAttribute("aria-pressed", b.getAttribute("data-view") === p.view); });
        if (p.select) { p.select.value = p.type; p.select.disabled = p.view === "stats"; }
        var showChart = p.view !== "stats" && p.chartOk;
        p.chartBox.hidden = !showChart;
        p.statsBox.hidden = p.view === "chart" && p.chartOk;
        if (showChart) { buildChart(p); } else if (p.chart) { p.chart.destroy(); p.chart = null; }
        store(p.cfg.id, { view: p.view, type: p.type });
    }

    function init(root) {
        var cfg;
        try { cfg = JSON.parse(root.getAttribute("data-viz")); } catch (e) { return; }
        var saved = store(cfg.id) || {};
        var p = {
            root: root, cfg: cfg, chartOk: typeof window.Chart !== "undefined",
            view: saved.view || store("global") || cfg.defaultView || "chart",
            type: cfg.types.indexOf(saved.type) >= 0 ? saved.type : cfg["default"]
        };

        var bar = el("div", "viz-toolbar");
        var seg = el("div", "viz-seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "How to show this");
        p.viewButtons = VIEWS.map(function (v) {
            var b = el("button", "viz-seg-btn", v[1]); b.type = "button"; b.setAttribute("data-view", v[0]);
            b.addEventListener("click", function () { p.view = v[0]; apply(p); });
            seg.appendChild(b); return b;
        });
        bar.appendChild(seg);
        if (cfg.types.length > 1) {
            var lab = el("label", "viz-type");
            lab.appendChild(el("span", null, "Chart type"));
            p.select = el("select");
            cfg.types.forEach(function (t) { var o = el("option", null, TYPE_NAMES[t] || t); o.value = t; p.select.appendChild(o); });
            p.select.addEventListener("change", function () { p.type = p.select.value; if (p.view === "stats") { p.view = "chart"; } apply(p); });
            lab.appendChild(p.select); bar.appendChild(lab);
        }
        root.appendChild(bar);

        p.chartBox = el("div", "chart-wrap chart-wrap-" + (cfg.height || "sm") + " viz-chart");
        p.canvas = el("canvas"); p.canvas.setAttribute("role", "img");
        p.canvas.setAttribute("aria-label", (cfg.title || "Chart") + ". Switch to Statistics for the exact values.");
        p.chartBox.appendChild(p.canvas); root.appendChild(p.chartBox);
        p.statsBox = el("div", "viz-stats"); root.appendChild(p.statsBox);
        if (!p.chartOk) { root.appendChild(el("p", "viz-note", "The chart library couldn't load, so the exact values are shown instead.")); }
        if (cfg.note) { root.appendChild(el("p", "viz-note", cfg.note)); }

        buildStats(p);
        panels.push(p);
        apply(p);
    }

    function initGlobal(box) {
        var seg = el("div", "viz-seg");
        box.appendChild(el("span", "viz-global-label", "Show all visuals as"));
        var current = store("global");
        VIEWS.forEach(function (v) {
            var b = el("button", "viz-seg-btn" + (current === v[0] ? " active" : ""), v[1]); b.type = "button";
            b.addEventListener("click", function () {
                store("global", v[0]);
                seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("active", x === b); });
                panels.forEach(function (p) { p.view = v[0]; apply(p); });
            });
            seg.appendChild(b);
        });
        box.appendChild(seg);
    }

    function start() {
        if (window.Chart) {
            window.Chart.defaults.color = MUTED;
            window.Chart.defaults.font.family = "'Poppins', sans-serif";
        }
        document.querySelectorAll(".viz[data-viz]").forEach(init);
        document.querySelectorAll(".viz-global").forEach(initGlobal);
    }
    if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
