"""Genereert de extensie-iconen (PNG) zonder externe assets. Draai: python3 scripts/make-icons.py"""
from PIL import Image, ImageDraw
import math, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extension", "icons")
os.makedirs(OUT, exist_ok=True)
S = 1024  # render groot, dan verkleinen (anti-aliasing)

def lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

# Achtergrond: diagonale gradient indigo -> cyaan, afgeronde hoeken
bg = Image.new("RGB", (S, S))
px = bg.load()
c1, c2 = (79, 70, 229), (14, 165, 233)
for y in range(S):
    for x in range(S):
        px[x, y] = lerp(c1, c2, (x + y) / (2 * S))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=255)
icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(bg, (0, 0), mask)

d = ImageDraw.Draw(icon)

def sparkle(cx, cy, r, fill):
    """4-punts ster (AI-'sparkle')."""
    pts = []
    for i in range(8):
        ang = math.pi / 4 * i - math.pi / 2
        rad = r if i % 2 == 0 else r * 0.28
        pts.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))
    d.polygon(pts, fill=fill)

# Grote sparkle + kleine sparkle (wit)
sparkle(S * 0.44, S * 0.52, S * 0.34, (255, 255, 255, 255))
sparkle(S * 0.75, S * 0.27, S * 0.13, (255, 255, 255, 235))
# Klein "muispijltje" rechtsonder: het is een web-agent
arrow = [(0.62, 0.62), (0.86, 0.72), (0.75, 0.76), (0.82, 0.89), (0.77, 0.91), (0.70, 0.79), (0.62, 0.86)]
d.polygon([(x * S, y * S) for x, y in arrow], fill=(255, 255, 255, 255), outline=(30, 41, 59, 255), width=int(S * 0.012))

for size in (16, 32, 48, 128, 256):
    icon.resize((size, size), Image.LANCZOS).save(os.path.join(OUT, f"icon{size}.png"))
    print("✓ icon%d.png" % size)
