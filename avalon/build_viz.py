"""Build a self-contained GOD-VIEW HTML viewer for one or more Avalon games.

A dropdown picks the game; the selected game renders in the per-round format:
a vote panel (each seat approve/reject, team highlighted, tally), the mission
result, then the labeled dialogue. Hover or click a line to reveal that player's
terse chain-of-thought (gray). God-view tool — shows hidden roles + private CoT;
never share with a player. (No API cost is shown.)

    python build_viz.py --games results/game2_cot results/game_seed2 ... --out results/games_viz.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict, deque
from pathlib import Path

# "knew" annotations contain brackets (evil-team[..], M/M-pair[..]); match greedily.
STMT_RE = re.compile(r"^seat (\d+) \[.*\]: (.*)$")
COT_RE = re.compile(r"^\s+└─ CoT\[")
EVIL = {"Assassin", "Morgana", "Minion of Mordred"}


def parse(out_dir: Path):
    events = [json.loads(l) for l in (out_dir / "events.jsonl").read_text().splitlines()]
    summary = json.loads((out_dir / "summary.json").read_text())
    cots = [json.loads(l) for l in (out_dir / "private_cot.jsonl").read_text().splitlines()]
    roles = {int(k): v for k, v in summary["true_roles"].items()}

    stmts = []
    for line in (out_dir / "godview_transcript.txt").read_text().splitlines():
        if line.startswith("=== OUTCOME"):
            break
        m = STMT_RE.match(line)
        if m:
            stmts.append(m.group(2).rstrip())
        elif stmts and line.strip() and not COT_RE.match(line) and not line.startswith("seat "):
            stmts[-1] += " " + line.strip()
    sidx = 0

    cot_q = defaultdict(deque)
    for c in cots:
        cot_q[c["action"]].append(c["reasoning"])

    rounds, cur, sub, vote_i, assassin = [], None, None, 0, None
    for e in events:
        a, actor, data = e["action"], e["actor"], e["data"]
        if a == "propose":
            cur = {"leader": actor, "reason": data.get("reason"), "subrounds": [],
                   "final_team": data["team"], "votes": None, "approved": None,
                   "approves": None, "mission": None, "result": None}
            rounds.append(cur); sub = None
        elif a == "round_order":
            sub = {"team": data.get("team", cur["final_team"]), "dialogue": [],
                   "decision": None, "swapped_to": None, "revise_cot": ""}
            cur["subrounds"].append(sub); cur["final_team"] = sub["team"]
        elif a == "speak":
            text = stmts[sidx] if sidx < len(stmts) else ""
            sidx += 1
            sub["dialogue"].append({"seat": actor, "role": roles[actor], "text": text,
                                    "passed": text.strip() == "(passed)", "ready": data.get("ready"),
                                    "cot": cot_q["discuss"].popleft() if cot_q["discuss"] else ""})
        elif a in ("leader_swap", "leader_keep"):
            if sub is not None:
                sub["decision"] = "swap" if a == "leader_swap" else "keep"
                sub["swapped_to"] = data.get("to")
                sub["revise_cot"] = cot_q["revise"].popleft() if cot_q["revise"] else ""
        elif a == "tally":
            sv = summary["votes"][vote_i]; vote_i += 1
            cur["votes"] = {int(k): v for k, v in sv["votes"].items()}
            cur["approved"] = sv["approved"]; cur["approves"] = sv["approves"]; cur["mission"] = sv["mission"]
        elif a == "result":
            cur["result"] = {"failed": data["failed"], "fail_count": data["fail_count"]}
        elif a == "guess":
            assassin = {"seat": actor, "guess": data["guess_seat"],
                        "cot": cot_q["assassin"].popleft() if cot_q["assassin"] else ""}
    return summary, roles, rounds, assassin


def esc(s):
    return html.escape(str(s or ""))


def seat_chip(seat, role, vote=None, on_team=False):
    al = "evil" if role in EVIL else "good"
    cls = f"chip {al}" + (" team" if on_team else "")
    badge = f'<span class="vote {vote}">{"✓" if vote == "approve" else "✗"}</span>' if vote else ""
    return (f'<div class="{cls}" title="{esc(role)}">{badge}'
            f'<span class="s">S{seat}</span><span class="r">{esc(role)}</span></div>')


def render_game(summary, roles, rounds, assassin) -> str:
    p = []
    win = summary["winner"].upper()
    p.append(f'<div class="ghead">Seed {summary.get("seed")} · <b>{win} wins</b> '
             f'<span class="sub">{esc(summary.get("reason",""))}</span></div>')
    def turn_html(t):
        al = "evil" if t["role"] in EVIL else "good"
        ready = '<span class="ready">●ready</span>' if t.get("ready") else ""
        body = '<span class="passed">(passed)</span>' if t["passed"] else esc(t["text"])
        cot = f'<div class="cot">CoT: {esc(t["cot"])}</div>' if t["cot"] else ""
        return (f'<div class="turn" onclick="this.classList.toggle(\'show\')">'
                f'<span class="who {al}">S{t["seat"]} · {esc(t["role"])}</span>{ready}: {body}{cot}</div>')

    for rd in rounds:
        mlabel = f"Mission {rd['mission']+1}" if rd["mission"] is not None else "Proposal"
        status = ('<span class="pill ok">APPROVED</span>' if rd["approved"]
                  else '<span class="pill no">REJECTED</span>')
        res = ""
        if rd["result"] is not None:
            res = (f'<span class="pill no">MISSION FAILED ({rd["result"]["fail_count"]}×fail)</span>'
                   if rd["result"]["failed"] else '<span class="pill ok">MISSION SUCCESS</span>')
        p.append('<div class="round">')
        p.append(f'<div class="rhead"><span class="rtitle">{mlabel} · leader S{rd["leader"]} · '
                 f'final team {sorted(rd["final_team"])}</span><span>{status} {res}</span></div>')
        multi = len(rd["subrounds"]) > 1
        for si, sub in enumerate(rd["subrounds"]):
            if multi:
                p.append(f'<div class="subhead">Discussion round {si+1} · discussing team {sorted(sub["team"])}</div>')
            p.append('<div class="dlg">')
            for t in sub["dialogue"]:
                p.append(turn_html(t))
            p.append("</div>")
            if sub["decision"] == "swap":
                cot = f'<div class="cot">CoT: {esc(sub["revise_cot"])}</div>' if sub["revise_cot"] else ""
                p.append(f'<div class="decision swap turn" onclick="this.classList.toggle(\'show\')">'
                         f'↻ Leader S{rd["leader"]} SWAPPED the team → {sorted(sub["swapped_to"])} '
                         f'(another discussion round){cot}</div>')
            elif sub["decision"] == "keep":
                cot = f'<div class="cot">CoT: {esc(sub["revise_cot"])}</div>' if sub["revise_cot"] else ""
                p.append(f'<div class="decision keep turn" onclick="this.classList.toggle(\'show\')">'
                         f'✓ Leader S{rd["leader"]} KEPT this team → vote{cot}</div>')
        # final vote panel
        p.append('<div class="votehead">Team vote</div><div class="vrow">')
        for s in range(7):
            p.append(seat_chip(s, roles[s], vote=rd["votes"].get(s) if rd["votes"] else None,
                               on_team=s in rd["final_team"]))
        p.append("</div>")
        if rd["approves"] is not None:
            p.append(f'<div class="tally">{rd["approves"]}/7 approve</div>')
        p.append("</div>")
    if assassin:
        merlin = summary["merlin_seat"]; correct = assassin["guess"] == merlin
        p.append('<div class="round"><div class="rhead"><span class="rtitle">Assassin phase</span>'
                 f'<span class="pill {"no" if correct else "ok"}">'
                 f'{"CORRECT — EVIL WINS" if correct else "WRONG — GOOD WINS"}</span></div>')
        p.append(f'<div class="turn" onclick="this.classList.toggle(\'show\')">'
                 f'<span class="who evil">S{assassin["seat"]} · Assassin</span>: '
                 f'guessed Merlin = seat {assassin["guess"]} (Merlin was seat {merlin})'
                 f'<div class="cot">CoT: {esc(assassin["cot"])}</div></div></div>')
    return "".join(p)


CSS = """
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e8e8ea}
.wrap{max-width:880px;margin:0 auto;padding:24px}
h1{font-size:20px;margin-bottom:4px}
.warn{background:#3a1d1d;border:1px solid #7a2d2d;color:#ffb4b4;padding:8px 12px;border-radius:8px;font-size:13px;margin:12px 0}
.picker{margin:14px 0}
select{background:#171a21;color:#e8e8ea;border:1px solid #2c3340;border-radius:8px;padding:8px 10px;font-size:14px}
.ghead{font-size:16px;margin:8px 0 4px} .ghead .sub{color:#9aa0aa;font-size:13px;font-weight:400;margin-left:6px}
.game{display:none} .game.active{display:block}
.round{background:#171a21;border:1px solid #262b36;border-radius:12px;padding:16px;margin:18px 0}
.rhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.rtitle{font-weight:600} .pill{padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600}
.ok{background:#15351f;color:#7ee29a} .no{background:#3a1d1d;color:#ff9a9a}
.vrow{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px}
.chip{position:relative;display:flex;flex-direction:column;align-items:center;min-width:62px;padding:6px 4px;border-radius:8px;border:1px solid #2c3340;background:#11141a}
.chip.good .r{color:#7fb0ff} .chip.evil .r{color:#ff8a8a}
.chip.team{outline:2px solid #d9a93a;outline-offset:1px}
.chip .s{font-weight:700;font-size:13px} .chip .r{font-size:10px;opacity:.85}
.vote{position:absolute;top:-7px;right:-7px;width:18px;height:18px;border-radius:50%;font-size:12px;display:flex;align-items:center;justify-content:center}
.vote.approve{background:#1f7a3d;color:#fff} .vote.reject{background:#a33;color:#fff}
.tally{font-size:13px;color:#9aa0aa;margin:6px 0 2px}
.subhead{font-size:12px;color:#c8b06a;font-weight:600;margin:12px 0 2px;text-transform:uppercase;letter-spacing:.4px}
.votehead{font-size:12px;color:#9aa0aa;font-weight:600;margin:14px 0 4px;text-transform:uppercase;letter-spacing:.4px;border-top:1px solid #262b36;padding-top:10px}
.decision{margin:6px 0;padding:7px 10px;border-radius:8px;font-size:13px;font-weight:600}
.decision.swap{background:#2a2433;border:1px solid #5a4a7a;color:#cdb6ff}
.decision.keep{background:#15351f;border:1px solid #2d6a40;color:#9ce2b0}
.dlg{margin-top:6px}
.turn{padding:7px 9px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.turn:hover{background:#1c2029;border-color:#2c3340}
.who{font-weight:600} .who.good{color:#7fb0ff} .who.evil{color:#ff8a8a}
.ready{font-size:10px;color:#7ee29a;margin-left:6px} .passed{color:#888;font-style:italic}
.cot{display:none;color:#9098a4;font-style:italic;font-size:13px;margin-top:5px;padding-left:10px;border-left:2px solid #3a414e}
.turn:hover .cot,.turn.show .cot{display:block}
.hint{color:#6b727d;font-size:12px}
"""


def render_page(games) -> str:
    opts, sections = [], []
    for i, (label, inner) in enumerate(games):
        opts.append(f'<option value="{i}">{esc(label)}</option>')
        sections.append(f'<div class="game{" active" if i == 0 else ""}" id="game-{i}">{inner}</div>')
    js = ("function showGame(i){document.querySelectorAll('.game').forEach(g=>g.classList.remove('active'));"
          "document.getElementById('game-'+i).classList.add('active');}")
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>Avalon — game viewer</title>'
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            f"<h1>Avalon — god-view game viewer</h1>"
            f'<div class="warn">⚠ GOD-VIEW: shows hidden roles and private chain-of-thought. '
            f'Never share with a player. <span class="hint">Hover or click a line to reveal its CoT (gray).</span></div>'
            f'<div class="picker">Game: <select onchange="showGame(this.value)">{"".join(opts)}</select></div>'
            f'{"".join(sections)}'
            f"<script>{js}</script></div></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True, help="game result dirs")
    ap.add_argument("--out", default="results/games_viz.html")
    args = ap.parse_args()
    games = []
    for g in args.games:
        summary, roles, rounds, assassin = parse(Path(g))
        label = f"Seed {summary.get('seed')} — {summary['winner'].upper()} wins"
        games.append((label, render_game(summary, roles, rounds, assassin)))
        print(f"  {g}: {label} | {len(rounds)} rounds")
    Path(args.out).write_text(render_page(games))
    print(f"wrote {args.out} ({len(games)} games)")


if __name__ == "__main__":
    main()
