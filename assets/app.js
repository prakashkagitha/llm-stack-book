/* The LLM Stack — client runtime: theme, search, nav, copy, progress, TOC scrollspy */
(function () {
  "use strict";

  // ---------- Theme ----------
  var root = document.documentElement;
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    try { localStorage.setItem("llmbook-theme", t); } catch (e) {}
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = t === "dark" ? "☀" : "☾";
  }
  window.__toggleTheme = function () {
    applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  };

  // ---------- Sidebar: collapse parts, persist, mark active ----------
  function initSidebar() {
    var saved;
    try { saved = JSON.parse(localStorage.getItem("llmbook-collapsed") || "{}"); } catch (e) { saved = {}; }
    document.querySelectorAll(".nav-part").forEach(function (part) {
      var key = part.getAttribute("data-part");
      var title = part.querySelector(".nav-part-title");
      var hasActive = part.querySelector("a.active");
      if (saved[key] === true && !hasActive) part.classList.add("collapsed");
      if (title) title.addEventListener("click", function () {
        part.classList.toggle("collapsed");
        saved[key] = part.classList.contains("collapsed");
        try { localStorage.setItem("llmbook-collapsed", JSON.stringify(saved)); } catch (e) {}
      });
    });
    // scroll active chapter into view in sidebar
    var active = document.querySelector(".sidebar a.active");
    if (active) { var sb = document.querySelector(".sidebar"); if (sb) { var r = active.getBoundingClientRect(); if (r.top < 120 || r.top > window.innerHeight - 120) active.scrollIntoView({ block: "center" }); } }
  }

  // ---------- Mobile menu ----------
  function initMobileMenu() {
    var t = document.getElementById("menu-toggle");
    var sb = document.querySelector(".sidebar");
    var scrim = document.querySelector(".scrim");
    function close() { if (sb) sb.classList.remove("open"); if (scrim) scrim.classList.remove("open"); }
    if (t && sb) t.addEventListener("click", function () { sb.classList.toggle("open"); if (scrim) scrim.classList.toggle("open"); });
    if (scrim) scrim.addEventListener("click", close);
  }

  // ---------- Copy buttons on code blocks ----------
  function initCopy() {
    document.querySelectorAll(".highlight").forEach(function (block) {
      var pre = block.querySelector("pre");
      if (!pre) return;
      var btn = document.createElement("button");
      btn.className = "copy-btn"; btn.type = "button"; btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var code = block.querySelector("code") || pre;
        var text = code.innerText.replace(/\n$/, "");
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = "Copied!"; btn.classList.add("copied");
          setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
        });
      });
      block.appendChild(btn);
    });
  }

  // ---------- Reading progress ----------
  function initProgress() {
    var bar = document.querySelector(".progress-bar");
    if (!bar) return;
    function upd() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    }
    document.addEventListener("scroll", upd, { passive: true }); upd();
  }

  // ---------- Right TOC scrollspy ----------
  function initScrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc-side a"));
    if (!links.length) return;
    var map = {};
    var heads = links.map(function (a) {
      var id = decodeURIComponent(a.getAttribute("href").slice(1));
      var el = document.getElementById(id); if (el) map[id] = a; return el;
    }).filter(Boolean);
    function upd() {
      var pos = window.scrollY + 90; var cur = null;
      for (var i = 0; i < heads.length; i++) { if (heads[i].offsetTop <= pos) cur = heads[i]; else break; }
      links.forEach(function (a) { a.classList.remove("active"); });
      if (cur && map[cur.id]) map[cur.id].classList.add("active");
    }
    document.addEventListener("scroll", upd, { passive: true }); upd();
  }

  // ---------- Search ----------
  function initSearch() {
    var input = document.getElementById("search-input");
    var resultsEl = document.getElementById("search-results");
    if (!input || !resultsEl) return;
    var index = null, loading = false, base = input.getAttribute("data-base") || "";
    var indexName = input.getAttribute("data-index") || "search-index.json";
    var sel = -1, items = [];

    function load() {
      if (index || loading) return;
      loading = true;
      fetch(base + "assets/" + indexName).then(function (r) { return r.json(); }).then(function (d) { index = d; });
    }
    input.addEventListener("focus", load);

    function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
    function highlight(text, q) {
      try { return text.replace(new RegExp("(" + esc(q) + ")", "gi"), "<mark>$1</mark>"); } catch (e) { return text; }
    }
    function search(q) {
      if (!index) { return []; }
      q = q.trim().toLowerCase(); if (q.length < 2) return [];
      var terms = q.split(/\s+/);
      var scored = [];
      index.forEach(function (doc) {
        var hay = (doc.title + " " + doc.part + " " + doc.headings + " " + doc.text).toLowerCase();
        var score = 0, ok = true;
        terms.forEach(function (t) {
          var n = hay.split(t).length - 1; if (n === 0) ok = false;
          score += n;
          if (doc.title.toLowerCase().indexOf(t) >= 0) score += 50;
          if (doc.headings.toLowerCase().indexOf(t) >= 0) score += 15;
        });
        if (ok) scored.push({ doc: doc, score: score });
      });
      scored.sort(function (a, b) { return b.score - a.score; });
      return scored.slice(0, 12).map(function (s) { return s.doc; });
    }
    function snippet(text, q) {
      var t = text.toLowerCase(), i = t.indexOf(q.split(/\s+/)[0]);
      if (i < 0) i = 0; var start = Math.max(0, i - 60);
      return (start > 0 ? "…" : "") + text.slice(start, start + 160) + "…";
    }
    function render(q) {
      items = search(q);
      sel = -1;
      if (!q || q.trim().length < 2) { resultsEl.classList.remove("open"); resultsEl.innerHTML = ""; return; }
      if (!items.length) { resultsEl.innerHTML = '<div class="sr-empty">No results for “' + q + '”' + (index ? "" : " (loading…)") + "</div>"; resultsEl.classList.add("open"); return; }
      resultsEl.innerHTML = items.map(function (d) {
        return '<a class="sr-item" href="' + base + d.url + '">' +
          '<div class="sr-title">' + highlight(d.title, q) + "</div>" +
          '<div class="sr-part">' + d.part + "</div>" +
          '<div class="sr-snippet">' + highlight(snippet(d.text, q), q) + "</div></a>";
      }).join("");
      resultsEl.classList.add("open");
    }
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (e) {
      var els = resultsEl.querySelectorAll(".sr-item");
      if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(sel + 1, els.length - 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(sel - 1, 0); }
      else if (e.key === "Enter") { if (els[sel]) { window.location.href = els[sel].getAttribute("href"); } else if (els[0]) { window.location.href = els[0].getAttribute("href"); } return; }
      else if (e.key === "Escape") { input.blur(); resultsEl.classList.remove("open"); return; }
      els.forEach(function (el, i) { el.classList.toggle("sel", i === sel); });
      if (els[sel]) els[sel].scrollIntoView({ block: "nearest" });
    });
    document.addEventListener("click", function (e) { if (!resultsEl.contains(e.target) && e.target !== input) resultsEl.classList.remove("open"); });
    // keyboard shortcut "/" focuses search
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input && !/input|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); input.focus(); }
    });
  }

  // ---------- Figure animations: trigger when scrolled into view ----------
  function initFigures() {
    var figs = Array.prototype.slice.call(document.querySelectorAll(".viz"));
    if (!figs.length) return;
    var reduce = false;
    try { reduce = matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) {}
    if (reduce || !("IntersectionObserver" in window)) {
      // Show final state immediately; CSS uses .in-view to gate animations.
      figs.forEach(function (f) { f.classList.add("in-view", "no-anim"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in-view"); io.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.25 });
    figs.forEach(function (f) { io.observe(f); });
    // Click-to-replay: restart CSS animations by toggling the class.
    figs.forEach(function (f) {
      var replay = f.querySelector(".viz-replay");
      if (replay) replay.addEventListener("click", function () {
        f.classList.remove("in-view"); void f.offsetWidth; f.classList.add("in-view");
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initSidebar(); initMobileMenu(); initCopy(); initProgress(); initScrollSpy(); initSearch(); initFigures();
    var tt = document.getElementById("theme-toggle");
    if (tt) tt.addEventListener("click", window.__toggleTheme);
  });
})();
