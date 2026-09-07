#!/usr/bin/env python3
"""Build web assets for the CUHK-X Dataset Observatory (docs/explorer).

Runs next to the CUHK-S zips. Reads HARn.zip / HAU.zip,
computes dataset statistics, selects a curated set of sample clips, transcodes
them to browser-friendly H.264, renders posters / hover-scrub film strips, and
emits stats.json + manifest.json. Everything lands in OUT_DIR, then gets
tarred for transfer to the static site.

Usage: python3 build_explorer_assets.py
"""
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("CUHKX_EXPLORER_ROOT", "/path/to/CUHK-S")
HARN_ZIP = os.path.join(ROOT, "HARn", "HARn.zip")
HAU_ZIP = os.path.join(ROOT, "HAU", "HAU.zip")
WORK = os.environ.get("CUHKX_EXPLORER_WORKDIR", "/tmp/cuhkx_web_build")
RAW = os.path.join(WORK, "raw")
OUT = os.path.join(WORK, "out")
CLIPS = os.path.join(OUT, "clips")

random.seed(42)

N_HAU_SETS = 4          # tri-pace, tri-modal showcase sets
N_QUIZ = 10             # next-action reasoning quiz clips
N_PROBE_EXTRA = 120     # extra random clips probed per benchmark for duration stats
FFMPEG_WORKERS = 12

# ---------------------------------------------------------------- helpers


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kw)


def ffprobe_duration(path):
    try:
        p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration,width,height,r_frame_rate",
                 "-of", "json", path])
        st = json.loads(p.stdout)["streams"][0]
        num, den = st.get("r_frame_rate", "0/1").split("/")
        fps = (float(num) / float(den)) if float(den) else 0.0
        return float(st.get("duration", 0.0)), int(st["width"]), int(st["height"]), fps
    except Exception as e:
        print("ffprobe failed:", path, e)
        return 0.0, 0, 0, 0.0


def transcode(src, dst, fps=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "27",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
    run(cmd)


def poster(src, dst, dur):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-ss", "%.2f" % max(0.1, dur * 0.3),
         "-i", src, "-frames:v", "1", "-q:v", "4", dst])


def strip(src, dst, dur, frames=12, w=180):
    """Horizontal film strip of `frames` frames for hover scrubbing."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rate = max(frames / max(dur, 0.5), 0.2)
    run(["ffmpeg", "-y", "-v", "error", "-i", src,
         "-vf", "fps=%.6f,scale=%d:-2,tile=%dx1" % (rate, w, frames),
         "-frames:v", "1", "-q:v", "5", dst])


def extract(zf, member, dest_root):
    """Extract one member preserving its relative path; returns abs path."""
    target = os.path.join(dest_root, member)
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(member) as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return target


def read_csv_from_zip(zf, member):
    with zf.open(member) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        return list(csv.DictReader(text))


def pretty_action(name):
    """'15_Wipe_windows_and_tables' -> (15, 'Wipe windows and tables')"""
    m = re.match(r"^(\d+)_(.+)$", name)
    if not m:
        return -1, name.replace("_", " ")
    return int(m.group(1)), m.group(2).replace("_", " ")


# The MobiSys '26 paper defines exactly 40 actions in 7 categories.
# CUHK-S ships 4 extra recorded classes (ids 40-43) outside that
# taxonomy; the explorer follows the paper and skips them.
N_PAPER_ACTIONS = 40
CATEGORIES = [
    (range(0, 6), "Personal Care"),
    (range(6, 12), "Eating & Drinking"),
    (range(12, 17), "Household"),
    (range(17, 23), "Working"),
    (range(23, 28), "Socializing & Leisure"),
    (range(28, 37), "Sports & Exercises"),
    (range(37, 40), "Caring & Helping"),
]


def category_of(aid):
    for r, name in CATEGORIES:
        if aid in r:
            return name
    return "Other"


# ---------------------------------------------------------------- scan zips
print("== scanning zips")
zharn = zipfile.ZipFile(HARN_ZIP)
zhau = zipfile.ZipFile(HAU_ZIP)

harn_infos = [i for i in zharn.infolist() if i.filename.endswith(".mp4")]
hau_infos = [i for i in zhau.infolist() if i.filename.endswith(".mp4")]

# HARn/data/<Mod>/<idx_Action>/<user>/<sess>/<Mod>.mp4
harn = []  # dicts
for i in harn_infos:
    p = i.filename.split("/")
    if len(p) == 7 and p[1] == "data":
        harn.append(dict(path=i.filename, mod=p[2], action=p[3], user=p[4],
                         sess=p[5], size=i.file_size))

# HAU/data/<Mod>/<user>/<sess>/<Mod>.mp4
hau = []
for i in hau_infos:
    p = i.filename.split("/")
    if len(p) == 6 and p[1] == "data":
        hau.append(dict(path=i.filename, mod=p[2], user=p[3], sess=p[4],
                        size=i.file_size))

harn = [c for c in harn if 0 <= pretty_action(c["action"])[0] < N_PAPER_ACTIONS]
print("HARn clips (paper's 40 classes):", len(harn), " HAU clips:", len(hau))

# ---------------------------------------------------------------- GT labels
print("== reading GT")
logic_rows = read_csv_from_zip(zharn, "HARn/GT/SM_Depth_logic.csv")
logic_rows = [r for r in logic_rows
              if 0 <= pretty_action(r["Path"].split("/")[3])[0] < N_PAPER_ACTIONS]
logic_by_path = {r["Path"]: r for r in logic_rows}

cap_rows = read_csv_from_zip(zhau, "HAU/GT/LM_Depth_GT.csv")
emo_rows = read_csv_from_zip(zhau, "HAU/GT/LM_Depth_Emotion.csv")
seq_rows = read_csv_from_zip(zhau, "HAU/GT/LM_Depth_sequential.csv")
cap_by_path = {r["Path"]: r["GT"] for r in cap_rows}
emo_by_path = {r["Path"]: r for r in emo_rows}
seq_by_path = {r["Path"]: r for r in seq_rows}

# ---------------------------------------------------------------- statistics
print("== computing stats")
harn_by_mod = Counter(c["mod"] for c in harn)
hau_by_mod = Counter(c["mod"] for c in hau)
harn_users = sorted({c["user"] for c in harn}, key=lambda u: int(u[4:]))
hau_users = sorted({c["user"] for c in hau}, key=lambda u: int(u[4:]))
all_users = sorted({*harn_users, *hau_users}, key=lambda u: int(u[4:]))

actions = {}
for c in harn:
    if c["mod"] != "Depth":
        continue
    aid, label = pretty_action(c["action"])
    a = actions.setdefault(aid, dict(id=aid, dir=c["action"], name=label,
                                     clips=0, users=set(), bytes=0))
    a["clips"] += 1
    a["users"].add(c["user"])
    a["bytes"] += c["size"]

# user x action matrix (Depth stream)
matrix_counts = defaultdict(Counter)
for c in harn:
    if c["mod"] == "Depth":
        aid, _ = pretty_action(c["action"])
        matrix_counts[c["user"]][aid] += 1

emotions = Counter(r["GT"] for r in emo_rows)
chain_actions = Counter()
chain_lengths = Counter()
for r in seq_rows:
    steps = [s.strip() for s in r["GT"].split(",") if s.strip()]
    chain_lengths[len(steps)] += 1
    chain_actions.update(steps)

hau_sessions = {}
for c in hau:
    hau_sessions.setdefault((c["user"], c["sess"]), set()).add(c["mod"])
scenes = sorted({s.split("-")[0] for (_, s) in hau_sessions})

# ---------------------------------------------------------------- selection
print("== selecting samples")
# --- HAU showcase sets: same user+scene+env, trials 1..3, all three modalities
groups = defaultdict(dict)  # (user, scene-env) -> trial -> {mod: clip}
for c in hau:
    sc = c["sess"].split("-")
    if len(sc) != 3:
        continue
    key = (c["user"], sc[0] + "-" + sc[1])
    groups[key].setdefault(sc[2], {})[c["mod"]] = c

complete = []
for (user, se), trials in groups.items():
    if all(t in trials and len(trials[t]) == 3 for t in ("1", "2", "3")):
        # labels must exist for all trials
        ok = all(("HAU/data/Depth/%s/%s-%s/Depth.mp4" % (user, se, t)) in cap_by_path
                 for t in ("1", "2", "3"))
        if ok:
            size = sum(c["size"] for t in trials.values() for c in t.values())
            complete.append(dict(user=user, se=se, trials=trials, size=size))

complete.sort(key=lambda g: -g["size"])
chosen_sets, used_scenes, used_users = [], set(), set()
for g in complete:
    scene = g["se"].split("-")[0]
    if scene in used_scenes or g["user"] in used_users:
        continue
    chosen_sets.append(g)
    used_scenes.add(scene)
    used_users.add(g["user"])
    if len(chosen_sets) == N_HAU_SETS:
        break
print("HAU sets:", [(g["user"], g["se"]) for g in chosen_sets])

# --- HARn atlas: one Depth+IR pair per action class, median size, prefer logic label
harn_pairs = defaultdict(dict)  # (action, user, sess) -> {mod: clip}
for c in harn:
    harn_pairs[(c["action"], c["user"], c["sess"])][c["mod"]] = c

atlas_pick = {}
by_action = defaultdict(list)
for key, mods in harn_pairs.items():
    if "Depth" in mods and "IR" in mods:
        by_action[key[0]].append((key, mods))
for action, cands in by_action.items():
    cands.sort(key=lambda kv: kv[1]["Depth"]["size"])
    good = [kv for kv in cands
            if 120_000 < kv[1]["Depth"]["size"] < 2_500_000]
    pool = good or cands
    with_logic = [kv for kv in pool if kv[1]["Depth"]["path"] in logic_by_path]
    pool2 = with_logic or pool
    atlas_pick[action] = pool2[len(pool2) // 2]

# --- Quiz: distinct logic labels, decent size, not in atlas
atlas_paths = {kv[1]["Depth"]["path"] for kv in atlas_pick.values()}
quiz_pool = []
seen_logic = set()
rows = [r for r in logic_rows if r["Path"] not in atlas_paths]
random.shuffle(rows)
harn_size_by_path = {c["path"]: c["size"] for c in harn}
for r in rows:
    if r["logic"] in seen_logic:
        continue
    sz = harn_size_by_path.get(r["Path"], 0)
    if not 150_000 < sz < 2_000_000:
        continue
    seen_logic.add(r["logic"])
    quiz_pool.append(r)
    if len(quiz_pool) == N_QUIZ:
        break
print("quiz:", [r["logic"] for r in quiz_pool])

# ---------------------------------------------------------------- extraction
print("== extracting")
os.makedirs(RAW, exist_ok=True)
need_harn, need_hau = set(), set()
for g in chosen_sets:
    for t in g["trials"].values():
        for c in t.values():
            need_hau.add(c["path"])
for kv in atlas_pick.values():
    need_harn.add(kv[1]["Depth"]["path"])
    need_harn.add(kv[1]["IR"]["path"])
for r in quiz_pool:
    need_harn.add(r["Path"])

# random extras for duration statistics
probe_harn = random.sample([c["path"] for c in harn if c["mod"] == "Depth"],
                           N_PROBE_EXTRA)
probe_hau = random.sample([c["path"] for c in hau if c["mod"] == "Depth"],
                          min(N_PROBE_EXTRA, len(hau)))

for m in sorted(need_harn | set(probe_harn)):
    extract(zharn, m, RAW)
for m in sorted(need_hau | set(probe_hau)):
    extract(zhau, m, RAW)
print("extracted files:", sum(len(f) for _, _, f in os.walk(RAW) if f))

# ---------------------------------------------------------------- durations
print("== probing durations")


def probe_many(paths):
    out = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(ffprobe_duration, os.path.join(RAW, p)): p for p in paths}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


probed = probe_many(list(need_harn | set(probe_harn) | need_hau | set(probe_hau)))
harn_durs = [probed[p][0] for p in (set(probe_harn) | need_harn)
             if p.startswith("HARn") and probed[p][0] > 0]
hau_durs = [probed[p][0] for p in (set(probe_hau) | need_hau)
            if p.startswith("HAU") and probed[p][0] > 0]


def hist(durs, bins):
    h = [0] * (len(bins) - 1)
    for d in durs:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                h[i] += 1
                break
        else:
            h[-1] += 1
    return h


harn_mean = sum(harn_durs) / max(len(harn_durs), 1)
hau_mean = sum(hau_durs) / max(len(hau_durs), 1)
total_hours = (harn_mean * len(harn) + hau_mean * len(hau)) / 3600.0
scene_hours = (harn_mean * harn_by_mod["Depth"] + hau_mean * hau_by_mod["Depth"]) / 3600.0

# ---------------------------------------------------------------- transcode
print("== transcoding")
os.makedirs(CLIPS, exist_ok=True)
jobs = []  # (fn, args)

manifest = dict(hauSets=[], atlas=[], quiz=[])

for g in chosen_sets:
    set_id = "%s_%s" % (g["user"], g["se"])
    entry = dict(id=set_id, user=g["user"], scene=int(g["se"].split("-")[0]),
                 env=int(g["se"].split("-")[1]), trials=[])
    for t in ("1", "2", "3"):
        mods = g["trials"][t]
        sess = "%s-%s" % (g["se"], t)
        dpath = "HAU/data/Depth/%s/%s/Depth.mp4" % (g["user"], sess)
        rel = "clips/hau/%s/t%s" % (set_id, t)
        clips = {}
        for mod, c in mods.items():
            dst = "%s/%s.mp4" % (rel, mod.lower())
            clips[mod.lower()] = dst
            jobs.append(("video", c["path"], os.path.join(OUT, dst), None))
        jobs.append(("poster", mods["Depth"]["path"],
                     os.path.join(OUT, rel, "poster.jpg"),
                     probed[mods["Depth"]["path"]][0]))
        seq = seq_by_path.get(dpath, {})
        emo = emo_by_path.get(dpath, {})
        entry["trials"].append(dict(
            trial=int(t), sess=sess,
            pace=emo.get("GT", ""),
            paceCandidates=[s.strip() for s in emo.get("Candidates", "").split(",") if s.strip()],
            caption=cap_by_path.get(dpath, ""),
            chain=[s.strip() for s in seq.get("GT", "").split(",") if s.strip()],
            dur=round(probed[mods["Depth"]["path"]][0], 2),
            clips=clips, poster=rel + "/poster.jpg"))
    manifest["hauSets"].append(entry)

for action, (key, mods) in sorted(atlas_pick.items(),
                                  key=lambda kv: pretty_action(kv[0])[0]):
    aid, label = pretty_action(action)
    dpath = mods["Depth"]["path"]
    dur = probed[dpath][0]
    rel = "clips/harn/a%02d" % aid
    logic = logic_by_path.get(dpath)
    manifest["atlas"].append(dict(
        id=aid, name=label, category=category_of(aid),
        user=key[1], sess=key[2], dur=round(dur, 2),
        clips=dict(depth=rel + "/depth.mp4", ir=rel + "/ir.mp4"),
        poster=rel + "/poster.jpg", strip=rel + "/strip.jpg",
        stripIr=rel + "/strip_ir.jpg", frames=12,
        logic=(dict(next=logic["logic"],
                    candidates=[s.strip() for s in logic["candidate"].split(",") if s.strip()])
               if logic else None)))
    ir_dur = probed[mods["IR"]["path"]][0]
    jobs.append(("video", dpath, os.path.join(OUT, rel, "depth.mp4"), None))
    jobs.append(("video", mods["IR"]["path"], os.path.join(OUT, rel, "ir.mp4"), None))
    jobs.append(("poster", dpath, os.path.join(OUT, rel, "poster.jpg"), dur))
    jobs.append(("strip", dpath, os.path.join(OUT, rel, "strip.jpg"), dur))
    jobs.append(("strip", mods["IR"]["path"], os.path.join(OUT, rel, "strip_ir.jpg"), ir_dur))

for n, r in enumerate(quiz_pool, 1):
    aid, label = pretty_action(r["Path"].split("/")[3])
    dur = probed[r["Path"]][0]
    rel = "clips/quiz/q%02d" % n
    cands = [s.strip() for s in r["candidate"].split(",") if s.strip()] + [r["logic"]]
    random.shuffle(cands)
    manifest["quiz"].append(dict(
        id="q%02d" % n, action=label, dur=round(dur, 2),
        clip=rel + "/depth.mp4", poster=rel + "/poster.jpg",
        answer=r["logic"], candidates=cands))
    jobs.append(("video", r["Path"], os.path.join(OUT, rel, "depth.mp4"), None))
    jobs.append(("poster", r["Path"], os.path.join(OUT, rel, "poster.jpg"), dur))

# hero = longest trial among chosen sets
hero = max(((s, t) for s in manifest["hauSets"] for t in s["trials"]),
           key=lambda st: st[1]["dur"])
manifest["hero"] = dict(set=hero[0]["id"], trial=hero[1]["trial"])


def do_job(j):
    kind, src, dst, dur = j
    src_abs = os.path.join(RAW, src)
    if kind == "video":
        transcode(src_abs, dst)
    elif kind == "poster":
        poster(src_abs, dst, dur or 5)
    elif kind == "strip":
        strip(src_abs, dst, dur or 5)
    return dst


print("jobs:", len(jobs))
done = 0
with ThreadPoolExecutor(max_workers=FFMPEG_WORKERS) as ex:
    futs = [ex.submit(do_job, j) for j in jobs]
    for f in as_completed(futs):
        f.result()
        done += 1
        if done % 25 == 0:
            print("  %d/%d" % (done, len(jobs)))

# ---------------------------------------------------------------- stats.json
size_harn = sum(c["size"] for c in harn)
size_hau = sum(c["size"] for c in hau)

stats = dict(
    generated="2026-06-13",
    source="CUHK-S",
    video=dict(width=320, height=240),
    parent=dict(name="CUHK-X", samples=64267, subjects=30, modalities=7,
                actionClasses="40+", environments=2),
    totals=dict(
        clips=len(harn) + len(hau),
        users=len(all_users),
        actions=len(actions),
        scenes=len(scenes),
        sensorHours=round(total_hours, 1),
        sceneHours=round(scene_hours, 1),
        sizeGB=round((size_harn + size_hau) / 1e9, 2),
        labelRows=len(logic_rows) * 2 + (len(cap_rows) + len(emo_rows) + len(seq_rows)) * 3),
    benchmarks=dict(
        HARn=dict(clips=len(harn), modalities=dict(harn_by_mod),
                  users=len(harn_users), actionClasses=len(actions),
                  logicLabels=len(logic_rows), meanDur=round(harn_mean, 1)),
        HAU=dict(clips=len(hau), modalities=dict(hau_by_mod),
                 users=len(hau_users), sequences=len(cap_rows),
                 scenes=len(scenes), meanDur=round(hau_mean, 1))),
    actions=[dict(id=a["id"], name=a["name"], category=category_of(a["id"]),
                  clips=a["clips"], users=len(a["users"]))
             for a in sorted(actions.values(), key=lambda x: x["id"])],
    categories=[name for _, name in CATEGORIES],
    matrix=dict(users=harn_users,
                actions=[a["id"] for a in sorted(actions.values(), key=lambda x: x["id"])],
                counts=[[matrix_counts[u].get(a["id"], 0)
                         for a in sorted(actions.values(), key=lambda x: x["id"])]
                        for u in harn_users]),
    emotions=dict(emotions),
    chainActions=[dict(name=k, count=v) for k, v in chain_actions.most_common(20)],
    chainLengths=sorted(chain_lengths.items()),
    durations=dict(
        harn=dict(mean=round(harn_mean, 1), sampled=len(harn_durs),
                  bins=[0, 3, 6, 9, 12, 15, 20, 30, 60],
                  hist=hist(harn_durs, [0, 3, 6, 9, 12, 15, 20, 30, 60, 10**9])),
        hau=dict(mean=round(hau_mean, 1), sampled=len(hau_durs),
                 bins=[0, 20, 30, 40, 50, 60, 80, 100, 140],
                 hist=hist(hau_durs, [0, 20, 30, 40, 50, 60, 80, 100, 140, 10**9]))),
)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

run(["tar", "-czf", os.path.join(WORK, "cuhks_web_assets.tar.gz"), "-C", OUT, "."])
total = subprocess.run(["du", "-sh", OUT, os.path.join(WORK, "cuhks_web_assets.tar.gz")],
                       stdout=subprocess.PIPE).stdout.decode()
print("== DONE ==")
print(total)
