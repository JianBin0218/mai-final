"""Draw the slide figures as SVG (run: python3 make_figures.py; writes next to this file).

Colors follow proposal.typ: ink-blue for latents and the model, orange for audio, gray for text inputs.
"""
from pathlib import Path

OUT = Path(__file__).parent
FONT = "New Computer Modern Sans, Nimbus Sans, Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2a33", "#55616d"
BLUE, PBLUE = "#075a9e", "#edf5fc"
ORANGE, PORANGE = "#c4571b", "#fdf1e8"
GRAY, PGRAY, LINE = "#8a96a1", "#f2f4f6", "#c5ccd3"
PURPLE = "#6b4fa0"
MARKERS = {"blue": BLUE, "orange": ORANGE, "gray": GRAY}


def svg(name, w, h, body):
    defs = "".join(
        f'<marker id="m-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
        f'markerWidth="11" markerHeight="11" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for k, c in MARKERS.items())
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-3 -3 {w + 6} {h + 6}" width="{w + 6}" height="{h + 6}" '
           f'font-family="{FONT}"><defs>{defs}</defs>{"".join(body)}</svg>\n')
    (OUT / f"{name}.svg").write_text(doc, encoding="utf-8")


def text(x, y, s, size=15, fill=INK, weight="normal", anchor="middle"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>'


def box(x, y, w, h, fill, stroke="none", rx=8, dash=None, width=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def path(d, color="blue", width=2.2, dash=None, arrow=True):
    c = MARKERS[color] if color in MARKERS else color
    m = f' marker-end="url(#m-{color})"' if arrow else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{width}"{ds}{m} stroke-linejoin="round"/>'


def wave(x, y, w, h, color=ORANGE):
    heights = [.3, .65, 1, .55, .8, .4, .9, .5, .3]
    step = w / len(heights)
    mid = y + h / 2
    return "".join(f'<line x1="{x + step * (i + .5):.1f}" y1="{mid - h * v / 2:.1f}" x2="{x + step * (i + .5):.1f}" '
                   f'y2="{mid + h * v / 2:.1f}" stroke="{color}" stroke-width="2.4" stroke-linecap="round"/>'
                   for i, v in enumerate(heights))


def latents(x, y, size=18, gap=4, fill=BLUE, stroke="none", dash=None, n=4):
    return "".join(box(x + i * (size + gap), y, size, size, fill, stroke, rx=3, dash=dash) for i in range(n))


# Text sizes are large on purpose: each figure is shown at 75-95% of the slide width.

# 1. Motivation: two paths can reach the answer.
svg("fig-motivation", 760, 205, [
    '<g transform="translate(30,0)">',
    box(10, 78, 150, 60, PORANGE, ORANGE), wave(24, 88, 40, 40), text(115, 115, "Audio", 20),
    path("M160,108 H200", "blue"),
    box(204, 78, 160, 60, PBLUE, BLUE), text(284, 103, "Model", 20, BLUE, "bold"), text(284, 126, "writes latents", 17, MUTED),
    path("M364,108 H392", "blue"),
    latents(396, 99),
    path("M484,108 H536", "blue"),
    box(540, 78, 150, 60, BLUE, BLUE), text(615, 116, "Answer", 21, "white", "bold"),
    path("M85,78 V38 H615 V74", "orange", 2.8),
    text(350, 27, "direct audio path", 19, ORANGE, "bold"),
    text(440, 165, "latent path", 19, BLUE, "bold"),
    '</g>',
    text(380, 199, "Both paths reach the answer. Which one does the model use?", 19, INK, "bold"),
])

# 2. Two-pass write-read architecture.
svg("fig-architecture", 760, 240, [
    box(0, 58, 125, 52, PORANGE, ORANGE), wave(10, 66, 34, 36), text(88, 91, "Audio", 18),
    box(0, 128, 125, 56, PGRAY, LINE), text(62, 152, "Question", 18), text(62, 174, "+ options", 18),
    path("M125,84 H156", "orange"), path("M125,156 H156", "gray"),
    box(160, 58, 172, 126, PBLUE, BLUE), text(246, 106, "Pass 1 · Write", 21, BLUE, "bold"),
    text(246, 132, "Qwen2-Audio-7B", 16, MUTED), text(246, 152, "+ shared LoRA", 16, MUTED),
    path("M332,121 H366", "blue"),
    latents(372, 112),
    text(414, 154, "K = 4 latents", 16, BLUE, "bold"),
    path("M460,121 H494", "blue"),
    box(498, 58, 172, 126, PBLUE, BLUE), text(584, 106, "Pass 2 · Answer", 21, BLUE, "bold"),
    text(584, 132, "Qwen2-Audio-7B", 16, MUTED), text(584, 152, "+ shared LoRA", 16, MUTED),
    path("M670,121 H700", "blue"),
    box(704, 96, 54, 52, BLUE, BLUE), text(731, 130, "B", 24, "white", "bold"), text(731, 168, "answer", 15, MUTED),
    path("M62,58 V32 H584 V54", "orange", 2.4, "8 5"),
    text(323, 22, "direct audio  ·  Soft: yes  ·  Mid: 50% dropped  ·  Hard: no", 16, ORANGE, "bold"),
    path("M62,184 V212 H584 V188", "gray"),
    text(323, 234, "question + options", 16, MUTED),
])

# 3. Reader conditions.
panels = []
for i, (title, audio, caption) in enumerate([("Soft · main test", "solid", "audio + latents"),
                                              ("Mid", "dashed", "audio dropped half the time"),
                                              ("Hard", "none", "latents only")]):
    px = i * 258
    panels += [box(px, 4, 244, 196, PGRAY, rx=10), text(px + 122, 34, title, 19, BLUE, "bold")]
    if audio == "none":
        panels += [box(px + 12, 52, 100, 42, "white", LINE, dash="5 4"), wave(px + 20, 58, 28, 30, LINE),
                   text(px + 82, 79, "Audio", 15, GRAY),
                   f'<path d="M{px + 118},63 l20,20 M{px + 138},63 l-20,20" stroke="{GRAY}" stroke-width="2.6" stroke-linecap="round"/>']
    else:
        panels += [box(px + 12, 52, 100, 42, PORANGE, ORANGE), wave(px + 20, 58, 28, 30), text(px + 82, 79, "Audio", 15),
                   path(f"M{px + 112},73 H{px + 144}", "orange", 2.4, "6 4" if audio == "dashed" else None)]
    panels += [latents(px + 16, 126, 18), text(px + 58, 166, "latents", 15, BLUE, "bold"),
               path(f"M{px + 104},135 H{px + 144}", "blue"),
               box(px + 148, 56, 84, 100, PBLUE, BLUE), text(px + 190, 112, "Reader", 17, BLUE, "bold"),
               text(px + 122, 190, caption, 15)]
svg("fig-readers", 760, 204, panels)

# 4. Latent interventions.
cols = []
for i, (name, sub, kw) in enumerate([
        ("Swap", "item j's latents", dict(fill=PURPLE)),
        ("Mean", "dataset mean", dict(fill="#9aa5b0")),
        ("Zero", "all zeros", dict(fill="white", stroke=GRAY, dash="4 3")),
        ("Noise", "+ Gaussian noise", dict(fill="#3f7fbf"))]):
    x0 = 196 + i * 141
    cols += [text(x0 + 60, 32, name, 19, INK, "bold"), latents(x0 + 14, 48, 20, **kw), text(x0 + 60, 98, sub, 16, MUTED)]
# White speckles on the four "Noise" squares.
nx = 196 + 3 * 141 + 14
noise = "".join(f'<circle cx="{nx + s * 24 + dx}" cy="{48 + dy}" r="1.9" fill="white"/>'
                for s in range(4) for dx, dy in [((5, 13, 9, 15)[s], (6, 5, 12, 7)[s]),
                                                 ((13, 6, 15, 5)[s], (14, 13, 6, 15)[s])])
svg("fig-interventions", 760, 168, [
    text(62, 32, "Original", 19, INK, "bold"), latents(16, 48, 20), text(62, 98, "item i's latents", 16, MUTED),
    path("M130,58 H178", "gray", 2.8),
    *cols, noise,
    box(0, 118, 760, 46, PBLUE, rx=8),
    text(380, 148, "Model, audio and question stay fixed; we count how many answers change.", 18, INK, "bold"),
])

# 5. Paired evidence swap.
svg("fig-paired-swap", 760, 252, [
    box(0, 22, 190, 56, PORANGE, ORANGE), wave(12, 31, 38, 38), text(122, 46, "Clip A", 18, INK, "bold"),
    text(122, 68, "dog barking", 15),
    box(0, 170, 190, 56, PORANGE, ORANGE), wave(12, 179, 38, 38), text(122, 194, "Clip B", 18, INK, "bold"),
    text(122, 216, "no barking", 15),
    path("M190,50 H526", "orange"), text(360, 40, "audio the reader hears", 16, ORANGE, "bold"),
    path("M190,198 H232", "orange"),
    box(236, 172, 112, 52, PBLUE, BLUE), text(292, 204, "Writer", 18, BLUE, "bold"),
    path("M348,198 H378", "blue"),
    latents(384, 189),
    text(426, 246, "latents from Clip B", 15, BLUE, "bold"),
    path("M472,198 H526", "blue"),
    box(206, 100, 300, 38, PGRAY, LINE, rx=19), text(356, 125, "Q: “Can you hear a dog barking?”", 16),
    path("M506,119 H526", "gray"), path("M292,138 V168", "gray"),
    box(530, 28, 112, 200, PBLUE, BLUE), text(586, 134, "Reader", 19, BLUE, "bold"),
    path("M642,90 H666", "blue"),
    box(670, 60, 90, 60, BLUE, BLUE), text(715, 86, "Yes − No", 16, "white", "bold"), text(715, 107, "score", 15, "white"),
    text(715, 152, "Evidence:", 15, BLUE, "bold"), text(715, 172, "score moves", 15, BLUE), text(715, 192, "toward No", 15, BLUE),
])

# 6. Optional chart: answers changed by a latent swap, per reader.
rows = [("Soft · audio", 0.2), ("Mid · audio", 19.7), ("Mid · no audio", 67.5), ("Hard · latents only", 74.0)]
x0, scale = 215, 4.5
chart = [f'<line x1="{x0 + t * scale}" y1="16" x2="{x0 + t * scale}" y2="196" stroke="#e3e7eb" stroke-width="1"/>'
         + text(x0 + t * scale, 216, f"{t}%", 15, MUTED) for t in (0, 25, 50, 75, 100)]
for i, (name, v) in enumerate(rows):
    y = 22 + i * 44
    chart += [text(200, y + 20, name, 18, INK, anchor="end"),
              f'<rect x="{x0}" y="{y}" width="{max(v * scale, 2):.1f}" height="28" rx="3" fill="{BLUE}"/>',
              text(x0 + v * scale + 9, y + 20, f"{v:.1f}%", 18, INK, "bold", "start")]
chart.append(text(x0 + 225, 242, "Answers changed when the latents are swapped", 16, MUTED))
svg("fig-swap-chart", 760, 250, chart)
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
