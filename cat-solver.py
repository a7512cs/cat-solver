#!/usr/bin/env python3
"""找貓貓（Queens 類）解題器
用法: python3 cat_solver.py 截圖.jpg [輸出圖.png]
規則：每種顏色、每列、每欄各一隻貓；貓不能相鄰（含斜角）。
"""
import sys, colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------- 1. 影像辨識 ----------------
def cluster_1d(vals, tol):
    vals = sorted(vals); groups = [[vals[0]]]
    for v in vals[1:]:
        if v - np.mean(groups[-1]) <= tol: groups[-1].append(v)
        else: groups.append([v])
    return [np.mean(g) for g in groups]

def detect_grid(img):
    import cv2
    im = img.convert("RGB")
    scale = min(1.0, 1200 / max(im.size))
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    a = np.asarray(im).astype(float) / 255
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    gray = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 20, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    mask = (edges == 0).astype(np.uint8)
    num, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 4)
    sq = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w < 12 or h < 12: continue
        if 0.75 < w / h < 1.33 and area > 0.75 * w * h:
            sq.append((x + w / 2, y + h / 2, w, h))
    if len(sq) < 10: raise SystemExit("找不到方格盤，請用清楚的正面截圖")
    # 找最多、大小一致的一群方塊
    sizes = np.array([s[2] for s in sq])
    best = max(sizes, key=lambda s: np.sum(np.abs(sizes - s) < 0.12 * s))
    sq = [s for s in sq if abs(s[2] - best) < 0.15 * best]
    xs = cluster_1d([s[0] for s in sq], best * 0.4)
    ys = cluster_1d([s[1] for s in sq], best * 0.4)
    def lattice(cs):
        d = np.diff(cs); pitch = np.median(d[d < 1.6 * best]) if np.any(d < 1.6 * best) else best * 1.1
        # 只保留彼此間距約為 pitch 整數倍、最長的那串
        k = np.round((np.array(cs) - cs[0]) / pitch)
        n = int(k.max()) + 1
        off = np.median(np.array(cs) - k * pitch)
        return [off + i * pitch for i in range(n)], pitch
    xs, px = lattice(xs); ys, py = lattice(ys)
    n = max(len(xs), len(ys))
    if len(xs) != len(ys):
        print(f"（警告：偵測到 {len(ys)} 列 x {len(xs)} 欄，可能有整排被叉掉）")
    # 用線性迴歸擬合每格中心（可抵抗照片微傾斜），再對齊到實際偵測到的方塊
    centers = {}
    for (sx, sy, w, h) in sq:
        c = int(round((sx - xs[0]) / px)); r = int(round((sy - ys[0]) / py))
        if 0 <= r < len(ys) and 0 <= c < len(xs): centers[(r, c)] = (sx, sy)
    if len(centers) >= 6:
        rc = np.array([[1, r, c] for (r, c) in centers]); xy = np.array(list(centers.values()))
        coef, *_ = np.linalg.lstsq(rc, xy, rcond=None)
    def center(r, c):
        if (r, c) in centers: return centers[(r, c)]
        if len(centers) >= 6: return tuple(np.array([1, r, c]) @ coef)
        return xs[c], ys[r]
    colors = np.zeros((len(ys), len(xs), 3))
    for r in range(len(ys)):
        for c in range(len(xs)):
            cx, cy = center(r, c)
            pts = []
            for dy, dx in ((-0.3, 0), (0.3, 0), (0, -0.3), (0, 0.3)):
                yy, xx = int(cy + dy * py), int(cx + dx * px)
                patch = a[yy - 2:yy + 3, xx - 2:xx + 3].reshape(-1, 3)
                patch = patch[(patch.max(1) - patch.min(1)) > 0.12]
                if len(patch): pts.append(patch)
            colors[r, c] = np.median(np.vstack(pts), 0) if pts else a[int(cy), int(cx)]
    if len(xs) != len(ys): raise SystemExit("列數和欄數不同，請重新截圖")
    boxes = []
    for r in range(len(ys)):
        for c in range(len(xs)):
            cx, cy = center(r, c)
            boxes.append(((cx - px * 0.45) / scale, (cy - py * 0.45) / scale,
                          (cx + px * 0.45) / scale, (cy + py * 0.45) / scale))
    return n, colors, boxes

def to_lab(c):
    def f(t): return np.where(t > 0.04045, ((t + 0.055) / 1.055) ** 2.4, t / 12.92)
    r, g, b = f(c[..., 0]), f(c[..., 1]), f(c[..., 2])
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089
    g_ = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = g_(X), g_(Y), g_(Z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)

def segment(n, colors):
    """相鄰格依色差由小到大合併，直到剩 n 個區域"""
    lab = to_lab(colors)
    parent = list(range(n * n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edges = []
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n and cc < n:
                    edges.append((np.linalg.norm(lab[r, c] - lab[rr, cc]), r * n + c, rr * n + cc))
    edges.sort()
    comps = n * n
    for d, u, v in edges:
        if comps == n: break
        a, b = find(u), find(v)
        if a != b: parent[a] = b; comps -= 1
    ids, reg = {}, np.zeros((n, n), int)
    for i in range(n * n):
        reg[i // n, i % n] = ids.setdefault(find(i), len(ids))
    return reg

PALETTE = {  # 遊戲上方那排貓的顏色（約略值）
    "咖啡": (120, 80, 50), "橘": (225, 150, 95), "黃": (200, 170, 60),
    "綠": (130, 180, 140), "青綠": (40, 150, 170), "天藍": (60, 140, 235),
    "淺藍": (145, 170, 230), "深藍": (40, 60, 170), "紫": (120, 70, 230),
    "桃紅": (190, 90, 180), "粉紅": (230, 150, 210), "紅": (220, 60, 60),
    "灰": (150, 150, 150), "淺綠": (170, 220, 120),
}

def name_regions(colors, reg, n):
    from scipy.optimize import linear_sum_assignment
    reg_rgb = np.array([colors[reg == g].mean(0) for g in range(n)])
    names = list(PALETTE)
    pal = np.array([PALETTE[k] for k in names]) / 255
    # 白平衡校正：讓辨識色的平均亮度/色偏貼近色盤
    adj = reg_rgb * (pal[:11].mean(0) / reg_rgb.mean(0))
    cost = np.linalg.norm(to_lab(adj)[:, None] - to_lab(pal)[None], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return {int(r): names[c] for r, c in zip(ri, ci)}

# ---------------- 2. 邏輯解題（模擬人類步驟） ----------------
class Board:
    def __init__(self, reg):
        self.n = len(reg); self.reg = reg
        self.cand = {(r, c) for r in range(self.n) for c in range(self.n)}
        self.cats = set()
    def copy(self):
        b = Board.__new__(Board); b.n, b.reg = self.n, self.reg
        b.cand, b.cats = set(self.cand), set(self.cats); return b
    def units(self):
        n = self.n
        for i in range(n):
            yield ("列", i), {(i, c) for c in range(n)}
            yield ("欄", i), {(r, i) for r in range(n)}
        for k in range(n):
            yield ("色", k), {(r, c) for r in range(n) for c in range(n) if self.reg[r][c] == k}
    def place(self, p):
        r, c = p
        self.cats.add(p)
        self.cand = {(rr, cc) for rr, cc in self.cand
                     if rr != r and cc != c and self.reg[rr][cc] != self.reg[r][c]
                     and not (abs(rr - r) <= 1 and abs(cc - c) <= 1)}
    def dead(self):
        for u, cells in self.units():
            if not (cells & self.cats) and not (cells & self.cand): return True
        return False
    def done(self): return len(self.cats) == self.n

def simple_step(b):
    """回傳 (說明, 放貓位置 or None, 刪除集合)"""
    # A. 只剩一格
    for u, cells in b.units():
        if cells & b.cats: continue
        cs = cells & b.cand
        if len(cs) == 1:
            return ("only", u, None), next(iter(cs)), set()
    # B. k 個區域被關在 k 列/欄 → 那些列/欄的其他格刪掉
    n = b.n
    from itertools import combinations
    for k in range(1, 4):
        for axis in (0, 1):
            free = [g for g in range(n) if not any(b.reg[r][c] == g for r, c in b.cats)]
            for grp in combinations(free, k):
                lines = {p[axis] for p in b.cand if b.reg[p[0]][p[1]] in grp}
                if len(lines) == k:
                    kill = {p for p in b.cand if p[axis] in lines and b.reg[p[0]][p[1]] not in grp}
                    if kill: return ("confine", grp, axis, lines), None, kill
            # 反向：k 條線只包含 k 種顏色
            freel = [i for i in range(n) if not any(p[axis] == i for p in b.cats)]
            for lines in combinations(freel, k):
                regs = {b.reg[p[0]][p[1]] for p in b.cand if p[axis] in lines}
                if len(regs) == k:
                    kill = {p for p in b.cand if b.reg[p[0]][p[1]] in regs and p[axis] not in lines}
                    if kill: return ("lines", lines, axis, regs), None, kill
    return None

def propagate(b, limit=200):
    for _ in range(limit):
        if b.done() or b.dead(): return
        s = simple_step(b)
        if not s: return
        _, place, kill = s
        if place: b.place(place)
        b.cand -= kill

def solve(reg, colors_of):
    b = Board(reg); steps = []
    while not b.done():
        s = simple_step(b)
        if s:
            info, place, kill = s
            if place: b.place(place)
            b.cand -= kill
            steps.append((info, place, kill))
        else:
            # D. 假設法：試放一隻貓，推下去會矛盾 → 這格不行
            found = False
            units = sorted(((len(cells & b.cand), u, cells & b.cand) for u, cells in b.units()
                            if not (cells & b.cats)), key=lambda t: t[0])
            for _, u, cs in units:
                for p in sorted(cs):
                    t = b.copy(); t.place(p); propagate(t)
                    if t.dead() and not t.done():
                        b.cand.discard(p); steps.append((("trial", p), None, {p})); found = True; break
                if found: break
            if not found: raise SystemExit("推不下去了，可能辨識顏色有誤")
        if b.dead(): raise SystemExit("矛盾，可能辨識顏色有誤")
    return b, steps

# ---------------- 3. 輸出 ----------------
def pos(p): return f"第{p[0]+1}列第{p[1]+1}欄"

def describe(steps, reg, cname):
    out, k = [], 0
    for info, place, kill in steps:
        t = info[0]
        if t == "only":
            if not place: continue
            k += 1
            unit = info[1]
            where = f"{cname[unit[1]]}色區" if unit[0] == "色" else f"第{unit[1]+1}{unit[0]}"
            out.append(f"{k}. 🐱 {where}只剩一格 → 放在 {pos(place)}（{cname[reg[place[0]][place[1]]]}）")
        else:
            if t == "confine":
                _, grp, axis, lines = info
                ax = "列" if axis == 0 else "欄"
                why = f"{'、'.join(cname[g] for g in grp)}色只能在第{'、'.join(str(l+1) for l in sorted(lines))}{ax}，這些{ax}的其他顏色不能放"
            elif t == "lines":
                _, lines, axis, regs = info
                ax = "列" if axis == 0 else "欄"
                why = f"第{'、'.join(str(l+1) for l in lines)}{ax}只剩{'、'.join(cname[g] for g in regs)}色，這些顏色在別的{ax}不能放"
            elif t == "kills":
                why = f"{pos(info[1])}放貓會讓某一區沒位置"
            else:
                why = f"假設 {pos(info[1])} 放貓，往下推會矛盾"
            out.append(f"   ✕ {why}（刪 {len(kill)} 格）")
    return "\n".join(out)

def draw(img, boxes, n, order, path):
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im)
    size = int((boxes[0][2] - boxes[0][0]) * 0.55)
    try: font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size)
    except Exception: font = ImageFont.load_default()
    for idx, (r, c) in enumerate(order, 1):
        x0, y0, x1, y1 = boxes[r * n + c]
        m = (x1 - x0) * 0.12
        d.ellipse([x0 + m, y0 + m, x1 - m, y1 - m], outline=(255, 0, 0), width=max(3, size // 6), fill=(255, 255, 255))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), str(idx), fill=(220, 0, 0), font=font, anchor="mm")
    im.save(path)

def main():
    src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "solution.png"
    img = Image.open(src)
    from PIL import ImageOps; img = ImageOps.exif_transpose(img)
    n, colors, boxes = detect_grid(img)
    reg = segment(n, colors)
    cname = name_regions(colors, reg, n)
    print(f"盤面 {n}x{n}，辨識到的顏色區域：")
    for r in range(n):
        print(" ".join(f"{cname[reg[r][c]]:　<2}" for c in range(n)))
    b, steps = solve(reg, cname)
    print("\n解題步驟：\n" + describe(steps, reg, cname))
    print("\n答案：")
    for r, c in sorted(b.cats): print(f"  {pos((r, c))}  {cname[reg[r][c]]}")
    draw(img, boxes, n, [p for _, p, _ in steps if p], out)
    print(f"\n標註圖：{out}")

if __name__ == "__main__":
    main()
