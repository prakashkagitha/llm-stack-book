#!/usr/bin/env python3
"""Aggregate concept/*.json (per-chapter records from gen_concept_workflow) into:
  concept/glossary.json  — deduped term -> {definition, home chapter, used_in}
  concept/graph.json     — ordered chapter records + resolved prerequisite links
  concept/tags.json      — tag -> [chapter ids]
build.py renders /glossary and /map from these. Run after the concept workflow.
Usage: python3 scripts/build_concept_pages.py
"""
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPT_DIR = os.path.join(ROOT, "concept")


def norm(term):
    t = term.strip().lower()
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()   # drop parenthetical expansion for the key
    t = re.sub(r"\s+", " ", t)
    return t


def chapter_order(book):
    order, pn = {}, 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            cid = f"{p['dir']}/{c['file']}"
            order[cid] = {"num": "" if front else f"{pn}.{i}", "title": c["title"],
                          "part": p["title"], "part_dir": p["dir"], "url": f"{p['dir']}/{c['file']}.html",
                          "sort": (0 if front and p['dir'][:2] == '00' else (99 if p['dir'][:2] == '99' else pn), i)}
    return order


def main():
    book = json.load(open(os.path.join(ROOT, "book.json")))
    order = chapter_order(book)
    recs = {}
    for f in sorted(glob.glob(os.path.join(CONCEPT_DIR, "*.json"))):
        if os.path.basename(f) in ("glossary.json", "graph.json", "tags.json", "tracks.json"):
            continue
        try:
            r = json.load(open(f))
        except ValueError:
            print("  bad json:", f); continue
        cid = r.get("chapter_id") or os.path.basename(f)[:-5].replace("__", "/")
        if cid not in order:
            # tolerate id drift; try filename
            cid2 = os.path.basename(f)[:-5].replace("__", "/")
            cid = cid2 if cid2 in order else cid
        recs[cid] = r

    # ---- glossary: term -> best definition + home chapter ----
    terms = {}   # key -> {term, entries:[(cid, definition, introduced_bool)]}
    for cid, r in recs.items():
        intro = {norm(x) for x in (r.get("introduces") or [])}
        for kt in (r.get("key_terms") or []):
            term, dfn = kt.get("term", "").strip(), kt.get("definition", "").strip()
            if not term or not dfn:
                continue
            k = norm(term)
            terms.setdefault(k, {"term": term, "entries": []})
            # prefer the longest surface form as display name
            if len(term) > len(terms[k]["term"]):
                terms[k]["term"] = term
            terms[k]["entries"].append((cid, dfn, k in intro))

    def csort(cid):
        return order.get(cid, {}).get("sort", (100, 0))

    glossary = []
    for k, v in terms.items():
        entries = v["entries"]
        # home chapter: prefer one that "introduces" it; else earliest; definition = longest from home
        homes = [e for e in entries if e[2]] or entries
        homes.sort(key=lambda e: csort(e[0]))
        home_cid = homes[0][0]
        # pick the longest definition among the home chapter's entries for this term
        home_defs = [e[1] for e in entries if e[0] == home_cid] or [homes[0][1]]
        definition = max(home_defs, key=len)
        used = sorted({e[0] for e in entries}, key=csort)
        glossary.append({
            "term": v["term"], "key": k, "definition": definition,
            "home": home_cid, "home_num": order.get(home_cid, {}).get("num", ""),
            "home_title": order.get(home_cid, {}).get("title", home_cid),
            "home_url": order.get(home_cid, {}).get("url", "#"),
            "used_in": used, "used_count": len(used),
        })
    glossary.sort(key=lambda g: g["key"])

    # map term-key -> home chapter, for resolving prerequisite phrases to links
    term_home = {g["key"]: g for g in glossary}

    def resolve_prereq(phrase):
        k = norm(phrase)
        if k in term_home:
            g = term_home[k]
            return {"text": phrase, "url": g["home_url"], "num": g["home_num"]}
        # substring match against known terms (longest match wins)
        cands = [g for kk, g in term_home.items() if kk and (kk in k or k in kk) and len(kk) > 3]
        if cands:
            g = max(cands, key=lambda g: len(g["key"]))
            return {"text": phrase, "url": g["home_url"], "num": g["home_num"]}
        return {"text": phrase, "url": None, "num": ""}

    # ---- graph: ordered chapter records with resolved prereqs ----
    graph = []
    for cid, r in recs.items():
        o = order.get(cid, {})
        graph.append({
            "id": cid, "num": o.get("num", ""), "title": o.get("title", cid),
            "part": o.get("part", ""), "part_dir": o.get("part_dir", ""), "url": o.get("url", "#"),
            "one_liner": r.get("one_liner", ""), "difficulty": r.get("difficulty", "core"),
            "tags": r.get("tags", []),
            "prereqs": [resolve_prereq(p) for p in (r.get("prerequisites") or [])],
        })
    graph.sort(key=lambda g: order.get(g["id"], {}).get("sort", (100, 0)))

    # ---- tags -> chapters ----
    tags = {}
    for g in graph:
        for t in g["tags"]:
            tags.setdefault(t.lower().strip(), []).append({"id": g["id"], "num": g["num"], "title": g["title"], "url": g["url"]})

    json.dump(glossary, open(os.path.join(CONCEPT_DIR, "glossary.json"), "w"), indent=1)
    json.dump(graph, open(os.path.join(CONCEPT_DIR, "graph.json"), "w"), indent=1)
    json.dump(tags, open(os.path.join(CONCEPT_DIR, "tags.json"), "w"), indent=1)
    linked = sum(1 for g in graph for p in g["prereqs"] if p["url"])
    nprq = sum(len(g["prereqs"]) for g in graph)
    print(f"records: {len(recs)} | glossary terms: {len(glossary)} | tags: {len(tags)} | "
          f"prereq links resolved: {linked}/{nprq}")


if __name__ == "__main__":
    main()
