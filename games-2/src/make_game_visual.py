"""Generate the cute game-explainer figure as SVG (rendered to PDF via headless Chrome)."""
import json
import subprocess

SCR = "/private/tmp/claude-501/-Users-sandraluo-cbai-2026/8e86e0a5-57a6-4fd0-b1fc-cbc596c946e2/scratchpad"
D = json.load(open(f"{SCR}/gamedata.json"))

INK, MUTED, PANEL = "#22212A", "#77758A", "#F2F1F7"
A_COL, B_COL, WIN, NEUT = "#C75B7A", "#2E7D8A", "#E0A82E", "#D2CFDC"
A_SOFT, B_SOFT = "#F4DDE4", "#D9EAEC"

W, H = 1240, 690
p = []
add = p.append


def face(cx, cy, r, col, look=0):
    """A round character head with eyes and a smile."""
    ex = r * 0.34
    return f"""
  <g filter="url(#soft)">
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}"/>
  </g>
  <circle cx="{cx - ex + look}" cy="{cy - r*0.12}" r="{r*0.115}" fill="#fff"/>
  <circle cx="{cx + ex + look}" cy="{cy - r*0.12}" r="{r*0.115}" fill="#fff"/>
  <path d="M {cx - r*0.32} {cy + r*0.28} Q {cx} {cy + r*0.58} {cx + r*0.32} {cy + r*0.28}"
        stroke="#fff" stroke-width="{r*0.10}" fill="none" stroke-linecap="round" opacity=".92"/>"""


def bubble(cx, cy, w, h, tail, fill="#fff", stroke="#E2E0EA", sw=1.6):
    """Rounded speech bubble; tail = 'left'|'right'|None."""
    x, y = cx - w / 2, cy - h / 2
    t = ""
    if tail == "left":
        t = f'<path d="M {x} {cy-7} L {x-11} {cy} L {x} {cy+7} Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'
    elif tail == "right":
        t = f'<path d="M {x+w} {cy-7} L {x+w+11} {cy} L {x+w} {cy+7} Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(14, h/2)}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>' + t)


def star(cx, cy, r, col=WIN, op=1.0):
    import math
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.44
        a = -math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + rr*math.cos(a):.2f},{cy + rr*math.sin(a):.2f}")
    return f'<polygon points="{" ".join(pts)}" fill="{col}" opacity="{op}"/>'


add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
    <feDropShadow dx="0" dy="2.5" stdDeviation="3.5" flood-color="#2A2540" flood-opacity="0.16"/>
  </filter>
  <filter id="card" x="-10%" y="-20%" width="120%" height="150%">
    <feDropShadow dx="0" dy="2" stdDeviation="5" flood-color="#2A2540" flood-opacity="0.07"/>
  </filter>
</defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>''')

# ---------------- title ----------------
add(f'<text x="{W/2}" y="46" text-anchor="middle" font-family="Avenir Next, Helvetica Neue, sans-serif" '
    f'font-size="26" font-weight="600" fill="{INK}" letter-spacing="0.2">The convergence game</text>')

# ---------------- rules cards ----------------
CY = 152
cards = [(46, 512), (588, 268), (886, 308)]
for x, w in cards:
    add(f'<rect x="{x}" y="78" width="{w}" height="148" rx="20" fill="{PANEL}" filter="url(#card)"/>')

# card 1 — simultaneous words
add(face(120, CY, 30, A_COL, look=3))
add(face(484, CY, 30, B_COL, look=-3))
add(bubble(215, CY, 108, 56, "left"))
add(bubble(389, CY, 108, 56, "right"))
# A's bubble: scattered dots (any word)
for dx, dy, c in [(-28, -10, "#C9C6D4"), (-6, 8, "#C9C6D4"), (16, -6, "#C9C6D4"), (30, 10, "#C9C6D4"), (2, -14, "#C9C6D4")]:
    add(f'<circle cx="{215+dx}" cy="{CY+dy}" r="4.6" fill="{c}"/>')
# B's bubble: tidy row of same-colour dots
for i in range(3):
    add(f'<circle cx="{389-22+i*22}" cy="{CY}" r="5.4" fill="{B_COL}" opacity="{0.45+0.2*i}"/>')
# padlock over B = secret rule
add(f'<g transform="translate(484,{CY-52})">'
    f'<path d="M -7 0 a 7 7 0 0 1 14 0" stroke="{B_COL}" stroke-width="3" fill="none" stroke-linecap="round"/>'
    f'<rect x="-10.5" y="0" width="21" height="15" rx="4" fill="{B_COL}"/>'
    f'<circle cx="0" cy="7" r="2.4" fill="#fff"/></g>')
add(f'<text x="484" y="{CY-62}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="11.5" fill="{B_COL}">secret category</text>')
add(f'<text x="120" y="{CY-62}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="11.5" fill="{MUTED}">any word</text>')
add(f'<line x1="302" y1="{CY-26}" x2="302" y2="{CY+26}" stroke="#CFCCDC" stroke-width="1.4" stroke-dasharray="3 4"/>')
add(f'<text x="302" y="212" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="13" fill="{MUTED}">one word each, at the same time</text>')

# card 2 — match wins
add(bubble(672, CY, 84, 50, None, fill=A_SOFT, stroke="#EBCAD6"))
add(bubble(788, CY, 84, 50, None, fill=B_SOFT, stroke="#C3DEE1"))
for cx, col in ((672, A_COL), (788, B_COL)):
    for i in range(3):
        add(f'<circle cx="{cx-18+i*18}" cy="{CY}" r="5" fill="{col}" opacity=".85"/>')
add(f'<text x="730" y="{CY+7}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="22" font-weight="600" fill="{WIN}">=</text>')
add(star(730, CY - 46, 15))
add(f'<text x="722" y="212" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="13" fill="{MUTED}">same word → both win</text>')

# card 3 — no repeats
add(bubble(1006, CY - 20, 100, 44, None, fill="#fff"))
for i in range(3):
    add(f'<circle cx="{1006-18+i*18}" cy="{CY-20}" r="5" fill="{NEUT}"/>')
add(bubble(1076, CY + 24, 100, 44, None, fill="#fff", stroke="#E9C9D2"))
for i in range(3):
    add(f'<circle cx="{1076-18+i*18}" cy="{CY+24}" r="5" fill="{NEUT}"/>')
add(f'<line x1="1034" y1="{CY+44}" x2="1118" y2="{CY+4}" stroke="{A_COL}" stroke-width="3.4" stroke-linecap="round"/>')
add(f'<text x="1040" y="212" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="13" fill="{MUTED}">no word may ever repeat</text>')

# ---------------- timelines ----------------
LX, RX = 118, 1190
CW = (RX - LX) / 97.0
CH, GAP = 24, 5


def timeline(g, top, heading, sub):
    d = D[g]
    n = d["n"]
    out = []
    out.append(f'<text x="46" y="{top-44}" font-family="Avenir Next, sans-serif" font-size="15.5" '
               f'font-weight="600" fill="{INK}">{heading}</text>')
    out.append(f'<text x="{RX}" y="{top-44}" text-anchor="end" font-family="Avenir Next, sans-serif" '
               f'font-size="13" fill="{MUTED}">{sub}</text>')
    # mini avatars as row labels
    out.append(face(78, top + CH/2, 13, A_COL, look=2))
    out.append(face(78, top + CH + GAP + CH/2, 13, B_COL, look=2))
    for i in range(n):
        x = LX + i*CW
        w = CW - 1.7
        c = d["A"][i]["c"]
        col = {"fam": A_COL, "cat": B_COL, "oth": NEUT}[c]
        out.append(f'<rect x="{x:.2f}" y="{top}" width="{w:.2f}" height="{CH}" rx="2.6" fill="{col}"/>')
        out.append(f'<rect x="{x:.2f}" y="{top+CH+GAP}" width="{w:.2f}" height="{CH}" rx="2.6" fill="{B_COL}"/>')
        if d["A"][i]["meet"]:
            out.append(f'<rect x="{x-2.4:.2f}" y="{top-3}" width="{w+4.8:.2f}" height="{2*CH+GAP+6}" rx="5" '
                       f'fill="none" stroke="{WIN}" stroke-width="2.4"/>')
            out.append(star(x + w/2, top - 16, 10))
    # round ticks
    ticks = [1, 20, 40, 60, 80, n] if n > 30 else [1, 5, 10, n]
    for t in ticks:
        x = LX + (t-1)*CW + (CW-1.7)/2
        out.append(f'<text x="{x:.2f}" y="{top+2*CH+GAP+18}" text-anchor="middle" '
                   f'font-family="Avenir Next, sans-serif" font-size="10.5" fill="#A5A2B2">{t}</text>')
    return "\n".join(out)


T1 = 322
add(timeline("10", T1, "A game that gets stuck", "&#8195;97 rounds"))
# callout for the 36-round family run (indices 11..46)
x1 = LX + 11*CW
x2 = LX + 47*CW
add(f'<path d="M {x1:.1f} {T1-7} L {x1:.1f} {T1-15} L {x2:.1f} {T1-15} L {x2:.1f} {T1-7}" '
    f'stroke="{A_COL}" stroke-width="1.3" fill="none" opacity=".65"/>')
add(f'<text x="{(x1+x2)/2:.1f}" y="{T1-22}" text-anchor="middle" font-family="Avenir Next, sans-serif" '
    f'font-size="12.5" fill="{A_COL}">ripen → ripest → riper → ripeness → &#8230; → ripereds'
    f'<tspan fill="{MUTED}" font-size="11.5">&#8195;36 rounds, mostly invented words</tspan></text>')
add(f'<text x="{LX}" y="{T1+2*CH+GAP+36}" font-family="Avenir Next, sans-serif" font-size="12" fill="{B_COL}">'
    f'paris&#8195;berlin&#8195;rome&#8195;madrid&#8195;vienna&#8195;oslo&#8195;&#8230; a city every single round</text>')

T2 = 500
add(timeline("4", T2, "A game that gets solved", "&#8195;16 rounds"))
gx = LX + 16*CW
ann = [("#8C8A99", "starts with dances &#8212; ballet, tango, salsa&#8230;"),
       (B_COL, "then joins B&#8217;s cities &#8212; beijing, athens, geneva, bern"),
       (WIN, "both say &#8220;prague&#8221; &#8212; they win")]
for i, (col, txt) in enumerate(ann):
    y = T2 + 2 + i*21
    wgt = "600" if i == 2 else "400"
    add(f'<text x="{gx+54}" y="{y}" font-family="Avenir Next, sans-serif" font-size="13" '
        f'fill="{col}" font-weight="{wgt}">{txt}</text>')
add(f'<path d="M {gx+10} {T2+26} L {gx+40} {T2+26}" stroke="#CFCCDC" stroke-width="1.3"/>')

# ---------------- legend ----------------
LY = 612
items = [(B_COL, "city — B's secret category"), (A_COL, "word-family run"), (NEUT, "any other word")]
x = 300
for col, lab in items:
    add(f'<rect x="{x}" y="{LY-11}" width="15" height="15" rx="3.5" fill="{col}"/>')
    add(f'<text x="{x+23}" y="{LY+1}" font-family="Avenir Next, sans-serif" font-size="12.5" fill="{MUTED}">{lab}</text>')
    x += 30 + len(lab)*6.8
add(star(x + 6, LY - 3, 9))
add(f'<text x="{x+20}" y="{LY+1}" font-family="Avenir Next, sans-serif" font-size="12.5" fill="{MUTED}">both said the same word</text>')
add(f'<text x="{W/2}" y="{LY+34}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="12" '
    f'fill="#A5A2B2">both players are Qwen3-32B &#183; unedited transcripts</text>')

add("</svg>")
svg = "\n".join(p)
open(f"{SCR}/game_cute.svg", "w").write(svg)
open(f"{SCR}/game_cute.html", "w").write(
    f"<html><head><style>@page{{size:{W}px {H}px;margin:0}}"
    f"html,body{{margin:0;padding:0}}</style></head><body>{svg}</body></html>")
print("svg written")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={SCR}/game_cute.pdf", "--virtual-time-budget=3000",
                f"file://{SCR}/game_cute.html"], capture_output=True)
subprocess.run([CHROME, "--headless", "--disable-gpu", f"--screenshot={SCR}/game_cute.png",
                f"--window-size={W},{H}", "--virtual-time-budget=3000",
                f"file://{SCR}/game_cute.html"], capture_output=True)
print("rendered")
