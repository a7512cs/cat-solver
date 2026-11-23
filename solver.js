// 找貓貓（Queens 類）解題器 — cat-solver.py 的瀏覽器移植版
// 規則：每種顏色、每列、每欄各一隻貓；貓不能相鄰（含斜角）。
(function (root) {
  "use strict";

  // ---------------- 小工具 ----------------
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  function median(a) {
    const s = Float64Array.from(a).sort(), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function* combinations(arr, k, start = 0, cur = []) {
    if (cur.length === k) { yield cur.slice(); return; }
    for (let i = start; i < arr.length; i++) {
      cur.push(arr[i]); yield* combinations(arr, k, i + 1, cur); cur.pop();
    }
  }

  // ---------------- 1. 影像辨識（等同 OpenCV 版本的流程） ----------------
  function cluster1d(vals, tol) {
    vals = vals.slice().sort((a, b) => a - b);
    const groups = [[vals[0]]];
    for (const v of vals.slice(1)) {
      if (v - mean(groups[groups.length - 1]) <= tol) groups[groups.length - 1].push(v);
      else groups.push([v]);
    }
    return groups.map(mean);
  }

  // cv2.cvtColor RGB2GRAY + GaussianBlur(3x3) + Canny(20, 60) + dilate(3x3)
  function edgeMask(rgb, W, H) {
    const gray = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++)
      gray[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
    const refl = (i, n) => (i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);
    const tmp = new Float32Array(W * H), blur = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      tmp[y * W + x] = 0.25 * gray[y * W + refl(x - 1, W)] + 0.5 * gray[y * W + x] + 0.25 * gray[y * W + refl(x + 1, W)];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      blur[y * W + x] = Math.round(0.25 * tmp[refl(y - 1, H) * W + x] + 0.5 * tmp[y * W + x] + 0.25 * tmp[refl(y + 1, H) * W + x]);
    const cl = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
    const g = (x, y) => blur[cl(y, H) * W + cl(x, W)];
    const dx = new Float32Array(W * H), dy = new Float32Array(W * H), mag = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const gx = (g(x + 1, y - 1) + 2 * g(x + 1, y) + g(x + 1, y + 1)) - (g(x - 1, y - 1) + 2 * g(x - 1, y) + g(x - 1, y + 1));
      const gy = (g(x - 1, y + 1) + 2 * g(x, y + 1) + g(x + 1, y + 1)) - (g(x - 1, y - 1) + 2 * g(x, y - 1) + g(x + 1, y - 1));
      const i = y * W + x; dx[i] = gx; dy[i] = gy; mag[i] = Math.abs(gx) + Math.abs(gy);
    }
    const LOW = 20, HIGH = 60, TG22 = Math.tan(Math.PI / 8), TG67 = Math.tan(3 * Math.PI / 8);
    const M = (x, y) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : mag[y * W + x]);
    const state = new Uint8Array(W * H); // 0 無, 1 弱, 2 強
    const stack = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, m = mag[i];
      if (m <= LOW) continue;
      const ax = Math.abs(dx[i]), ay = Math.abs(dy[i]);
      let keep;
      if (ay < ax * TG22) keep = m > M(x - 1, y) && m >= M(x + 1, y);
      else if (ay > ax * TG67) keep = m > M(x, y - 1) && m >= M(x, y + 1);
      else {
        const s = (dx[i] < 0) !== (dy[i] < 0) ? -1 : 1;
        keep = m > M(x - s, y - 1) && m > M(x + s, y + 1);
      }
      if (!keep) continue;
      if (m > HIGH) { state[i] = 2; stack.push(i); } else state[i] = 1;
    }
    while (stack.length) {
      const i = stack.pop(), x = i % W, y = (i / W) | 0;
      for (let yy = y - 1; yy <= y + 1; yy++) for (let xx = x - 1; xx <= x + 1; xx++) {
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (state[j] === 1) { state[j] = 2; stack.push(j); }
      }
    }
    const edges = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let e = 0;
      for (let yy = Math.max(0, y - 1); yy <= Math.min(H - 1, y + 1) && !e; yy++)
        for (let xx = Math.max(0, x - 1); xx <= Math.min(W - 1, x + 1); xx++)
          if (state[yy * W + xx] === 2) { e = 1; break; }
      edges[y * W + x] = e;
    }
    return edges;
  }

  // cv2.connectedComponentsWithStats(mask, 4)，依掃描順序編號
  function components(edges, W, H) {
    const lab = new Int32Array(W * H).fill(-1), out = [], q = new Int32Array(W * H);
    for (let s = 0; s < W * H; s++) {
      if (edges[s] || lab[s] >= 0) continue;
      const id = out.length;
      let x0 = W, y0 = H, x1 = -1, y1 = -1, area = 0, head = 0, tail = 0;
      q[tail++] = s; lab[s] = id;
      while (head < tail) {
        const i = q[head++], x = i % W, y = (i / W) | 0;
        area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        const nb = [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1];
        for (const j of nb) if (j >= 0 && !edges[j] && lab[j] < 0) { lab[j] = id; q[tail++] = j; }
      }
      out.push({ x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area });
    }
    return out;
  }

  // rgb: Uint8 RGB 陣列（已縮到最長邊 ≤ 1200），scale: 縮放比例
  function detectGrid(rgb, W, H, scale) {
    const a = (y, x, ch) => rgb[(y * W + x) * 3 + ch] / 255;
    const edges = edgeMask(rgb, W, H);
    let sq = [];
    for (const { x, y, w, h, area } of components(edges, W, H)) {
      if (w < 12 || h < 12) continue;
      if (w / h > 0.75 && w / h < 1.33 && area > 0.75 * w * h) sq.push([x + w / 2, y + h / 2, w, h]);
    }
    if (sq.length < 10) throw new Error("找不到方格盤，請用清楚的正面截圖");
    const sizes = sq.map((s) => s[2]);
    let best = sizes[0], bestCnt = -1;
    for (const s of sizes) {
      const cnt = sizes.filter((v) => Math.abs(v - s) < 0.12 * s).length;
      if (cnt > bestCnt) { bestCnt = cnt; best = s; }
    }
    sq = sq.filter((s) => Math.abs(s[2] - best) < 0.15 * best);
    function lattice(cs) {
      const d = cs.slice(1).map((v, i) => v - cs[i]).filter((v) => v < 1.6 * best);
      const pitch = d.length ? median(d) : best * 1.1;
      const k = cs.map((v) => Math.round((v - cs[0]) / pitch));
      const n = Math.max(...k) + 1;
      const off = median(cs.map((v, i) => v - k[i] * pitch));
      return [Array.from({ length: n }, (_, i) => off + i * pitch), pitch];
    }
    const [xs, px] = lattice(cluster1d(sq.map((s) => s[0]), best * 0.4));
    const [ys, py] = lattice(cluster1d(sq.map((s) => s[1]), best * 0.4));
    if (xs.length !== ys.length) throw new Error(`偵測到 ${ys.length} 列 × ${xs.length} 欄，列數和欄數不同，請重新截圖`);
    const n = xs.length;
    const centers = new Map();
    for (const [sx, sy] of sq) {
      const c = Math.round((sx - xs[0]) / px), r = Math.round((sy - ys[0]) / py);
      if (r >= 0 && r < n && c >= 0 && c < n) centers.set(r * n + c, [sx, sy]);
    }
    // 線性迴歸 [1, r, c] → (x, y)，抵抗微傾斜
    let coef = null;
    if (centers.size >= 6) {
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], B = [[0, 0], [0, 0], [0, 0]];
      for (const [k, [x, y]] of centers) {
        const v = [1, Math.floor(k / n), k % n];
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) A[i][j] += v[i] * v[j];
          B[i][0] += v[i] * x; B[i][1] += v[i] * y;
        }
      }
      coef = solve3(A, B);
    }
    const center = (r, c) => {
      if (centers.has(r * n + c)) return centers.get(r * n + c);
      if (coef) return [0, 1].map((j) => coef[0][j] + r * coef[1][j] + c * coef[2][j]);
      return [xs[c], ys[r]];
    };
    const colors = [], boxes = [];
    for (let r = 0; r < n; r++) {
      colors.push([]);
      for (let c = 0; c < n; c++) {
        const [cx, cy] = center(r, c);
        const pts = [];
        for (const [ddy, ddx] of [[-0.3, 0], [0.3, 0], [0, -0.3], [0, 0.3]]) {
          const yy = Math.trunc(cy + ddy * py), xx = Math.trunc(cx + ddx * px);
          for (let y = yy - 2; y < yy + 3; y++) for (let x = xx - 2; x < xx + 3; x++) {
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            const p = [a(y, x, 0), a(y, x, 1), a(y, x, 2)];
            if (Math.max(...p) - Math.min(...p) > 0.12) pts.push(p);
          }
        }
        colors[r].push(pts.length ? [0, 1, 2].map((ch) => median(pts.map((p) => p[ch])))
          : [0, 1, 2].map((ch) => a(Math.trunc(cy), Math.trunc(cx), ch)));
        boxes.push([(cx - px * 0.45) / scale, (cy - py * 0.45) / scale, (cx + px * 0.45) / scale, (cy + py * 0.45) / scale]);
      }
    }
    return { n, colors, boxes };
  }

  function solve3(A, B) { // 高斯消去解 A X = B（3x3, 3x2）
    const M = A.map((row, i) => row.concat(B[i]));
    for (let i = 0; i < 3; i++) {
      let p = i;
      for (let k = i + 1; k < 3; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
      [M[i], M[p]] = [M[p], M[i]];
      for (let k = 0; k < 3; k++) if (k !== i) {
        const f = M[k][i] / M[i][i];
        for (let j = i; j < 5; j++) M[k][j] -= f * M[i][j];
      }
    }
    return M.map((row, i) => [row[3] / row[i], row[4] / row[i]]);
  }

  function toLab([R, G, B]) {
    const f = (t) => (t > 0.04045 ? ((t + 0.055) / 1.055) ** 2.4 : t / 12.92);
    const r = f(R), g = f(G), b = f(B);
    const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505;
    const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089;
    const h = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    const fx = h(X), fy = h(Y), fz = h(Z);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

  // 相鄰格依色差由小到大合併，直到剩 n 個區域
  function segment(n, colors) {
    const lab = colors.map((row) => row.map(toLab));
    const parent = Array.from({ length: n * n }, (_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const edges = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        const rr = r + dr, cc = c + dc;
        if (rr < n && cc < n) edges.push([dist(lab[r][c], lab[rr][cc]), r * n + c, rr * n + cc]);
      }
    edges.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
    let comps = n * n;
    for (const [, u, v] of edges) {
      if (comps === n) break;
      const a = find(u), b = find(v);
      if (a !== b) { parent[a] = b; comps--; }
    }
    const ids = new Map(), reg = [];
    for (let i = 0; i < n * n; i++) {
      const root = find(i);
      if (!ids.has(root)) ids.set(root, ids.size);
      if (i % n === 0) reg.push([]);
      reg[Math.floor(i / n)].push(ids.get(root));
    }
    return reg;
  }

  function regionColors(colors, reg, n) {
    return Array.from({ length: n }, (_, g) => {
      const cs = [];
      reg.forEach((row, r) => row.forEach((k, c) => { if (k === g) cs.push(colors[r][c]); }));
      return [0, 1, 2].map((ch) => mean(cs.map((x) => x[ch])));
    });
  }

  // ---------------- 2. 邏輯解題（模擬人類步驟） ----------------
  // 格子用 r * n + c 表示
  class Board {
    constructor(reg) {
      this.n = reg.length; this.reg = reg;
      this.cand = new Set(Array.from({ length: this.n * this.n }, (_, i) => i));
      this.cats = new Set();
      this._units = null;
    }
    copy() {
      const b = Object.create(Board.prototype);
      Object.assign(b, this); b.cand = new Set(this.cand); b.cats = new Set(this.cats); return b;
    }
    regOf(i) { return this.reg[Math.floor(i / this.n)][i % this.n]; }
    units() {
      if (this._units) return this._units;
      const n = this.n, u = [];
      for (let i = 0; i < n; i++) {
        u.push([["列", i], Array.from({ length: n }, (_, c) => i * n + c)]);
        u.push([["欄", i], Array.from({ length: n }, (_, r) => r * n + i)]);
      }
      for (let k = 0; k < n; k++) {
        const cells = [];
        for (let i = 0; i < n * n; i++) if (this.regOf(i) === k) cells.push(i);
        u.push([["色", k], cells]);
      }
      return (this._units = u);
    }
    place(p) {
      const n = this.n, r = Math.floor(p / n), c = p % n, g = this.regOf(p);
      this.cats.add(p);
      for (const q of [...this.cand]) {
        const rr = Math.floor(q / n), cc = q % n;
        if (rr === r || cc === c || this.regOf(q) === g || (Math.abs(rr - r) <= 1 && Math.abs(cc - c) <= 1)) this.cand.delete(q);
      }
    }
    deadUnit() {
      for (const [u, cells] of this.units())
        if (!cells.some((p) => this.cats.has(p)) && !cells.some((p) => this.cand.has(p))) return u;
      return null;
    }
    dead() { return this.deadUnit() !== null; }
    done() { return this.cats.size === this.n; }
  }

  // 回傳 [info, place 或 null, kill 集合]
  function simpleStep(b) {
    const n = b.n;
    // A. 只剩一格
    for (const [u, cells] of b.units()) {
      if (cells.some((p) => b.cats.has(p))) continue;
      const cs = cells.filter((p) => b.cand.has(p));
      if (cs.length === 1) return [{ type: "only", unit: u }, cs[0], []];
    }
    // B. k 個區域被關在 k 列/欄 → 那些列/欄的其他格刪掉
    const axisOf = (p, axis) => (axis === 0 ? Math.floor(p / n) : p % n);
    for (let k = 1; k < 4; k++) {
      for (const axis of [0, 1]) {
        const catRegs = new Set([...b.cats].map((p) => b.regOf(p)));
        const free = [...Array(n).keys()].filter((g) => !catRegs.has(g));
        for (const grp of combinations(free, k)) {
          const lines = new Set([...b.cand].filter((p) => grp.includes(b.regOf(p))).map((p) => axisOf(p, axis)));
          if (lines.size === k) {
            const kill = [...b.cand].filter((p) => lines.has(axisOf(p, axis)) && !grp.includes(b.regOf(p)));
            if (kill.length) return [{ type: "confine", grp, axis, lines: [...lines].sort((x, y) => x - y) }, null, kill];
          }
        }
        // 反向：k 條線只包含 k 種顏色
        const catLines = new Set([...b.cats].map((p) => axisOf(p, axis)));
        const freel = [...Array(n).keys()].filter((i) => !catLines.has(i));
        for (const lines of combinations(freel, k)) {
          const regs = new Set([...b.cand].filter((p) => lines.includes(axisOf(p, axis))).map((p) => b.regOf(p)));
          if (regs.size === k) {
            const kill = [...b.cand].filter((p) => regs.has(b.regOf(p)) && !lines.includes(axisOf(p, axis)));
            if (kill.length) return [{ type: "lines", lines, axis, regs: [...regs].sort((x, y) => x - y) }, null, kill];
          }
        }
      }
    }
    return null;
  }

  // onStep(info, place, kill, removed)：記錄每一步（假設法的推理過程要顯示用）
  function applyStep(b, s, onStep) {
    const [info, place, kill] = s, before = new Set(b.cand);
    if (place !== null) b.place(place);
    for (const p of kill) b.cand.delete(p);
    if (onStep) onStep(info, place, kill, [...before].filter((p) => !b.cand.has(p) && p !== place));
  }

  function propagate(b, limit = 200, onStep) {
    for (let i = 0; i < limit; i++) {
      if (b.done() || b.dead()) return;
      const s = simpleStep(b);
      if (!s) return;
      applyStep(b, s, onStep);
    }
  }

  // 在盤面 b 上重新檢查一步推理是否仍成立；成立就回傳（依目前盤面重算的）[info, place, kill]，否則 null
  function recheck(b, s) {
    const { info, place } = s, n = b.n;
    const axisOf = (p, axis) => (axis === 0 ? Math.floor(p / n) : p % n);
    const catRegs = new Set([...b.cats].map((q) => b.regOf(q)));
    if (place !== null) {
      if (!b.cand.has(place)) return null;
      // 只要有任何一個列/欄/色區只剩這格，就成立（優先沿用原本的理由）
      const forced = b.units().filter(([, cells]) => cells.includes(place) && !cells.some((q) => b.cats.has(q)) &&
        cells.filter((q) => b.cand.has(q)).length === 1);
      if (!forced.length) return null;
      const same = forced.find(([u]) => u[0] === info.unit[0] && u[1] === info.unit[1]) || forced[0];
      return [{ type: "only", unit: same[0] }, place, []];
    }
    if (info.type === "confine") {
      if (info.grp.some((g) => catRegs.has(g))) return null;
      const lines = new Set([...b.cand].filter((q) => info.grp.includes(b.regOf(q))).map((q) => axisOf(q, info.axis)));
      if (lines.size !== info.grp.length) return null;
      const kill = [...b.cand].filter((q) => lines.has(axisOf(q, info.axis)) && !info.grp.includes(b.regOf(q)));
      return kill.length ? [{ ...info, lines: [...lines].sort((x, y) => x - y) }, null, kill] : null;
    }
    if (info.type === "lines") {
      if ([...b.cats].some((q) => info.lines.includes(axisOf(q, info.axis)))) return null;
      const regs = new Set([...b.cand].filter((q) => info.lines.includes(axisOf(q, info.axis))).map((q) => b.regOf(q)));
      if (regs.size !== info.lines.length) return null;
      const kill = [...b.cand].filter((q) => regs.has(b.regOf(q)) && !info.lines.includes(axisOf(q, info.axis)));
      return kill.length ? [{ ...info, regs: [...regs].sort((x, y) => x - y) }, null, kill] : null;
    }
    return null;
  }

  // 假設 p 放貓後，照 chain 的順序重播推理；每步都重新檢查，最後要出現矛盾才算成功
  function replay(b, p, chain) {
    const snap = (x) => ({ cats: [...x.cats], cand: [...x.cand] });
    const t = b.copy(), before = new Set(t.cand);
    t.place(p);
    const start = { removed: [...before].filter((q) => !t.cand.has(q) && q !== p), ...snap(t) };
    const subs = [];
    for (const s of chain) {
      if (t.dead()) break;
      const r = recheck(t, s);
      if (!r) return null;
      applyStep(t, r, (info, place, kill, removed) => subs.push({ info, place, kill, removed, ...snap(t) }));
    }
    return t.dead() && !t.done() ? { start, subs, deadUnit: t.deadUnit() } : null;
  }

  // 假設 p 放貓：先照一般順序推到矛盾，再把跟矛盾無關的步驟一一拿掉
  function shortestContradiction(b, p) {
    const t = b.copy();
    t.place(p);
    let chain = [];
    propagate(t, 200, (info, place, kill) => chain.push({ info, place, kill }));
    if (!t.dead() || t.done()) return null;
    let res = replay(b, p, chain);
    if (!res) return null;
    chain = res.subs;
    for (let i = 0; i < chain.length;) {
      const r = replay(b, p, chain.slice(0, i).concat(chain.slice(i + 1)));
      if (r) { res = r; chain = r.subs; } else i++;
    }
    return res;
  }

  // 回傳步驟陣列；每步含 info / place / kill 及執行後的盤面快照
  function solve(reg) {
    const b = new Board(reg), steps = [];
    const snap = (x) => ({ cats: [...x.cats], cand: [...x.cand] });
    while (!b.done()) {
      const s = simpleStep(b);
      if (s) {
        applyStep(b, s, (info, place, kill, removed) => steps.push({ info, place, kill, removed, ...snap(b) }));
      } else {
        // D. 假設法：試放一隻貓，推下去會矛盾 → 這格不行
        // 每個候選格都試，挑「精簡後推理步數最少」的，也就是最容易看出來的那個
        const units = b.units().filter(([, cells]) => !cells.some((p) => b.cats.has(p)))
          .map(([u, cells]) => [cells.filter((p) => b.cand.has(p)).length, u, cells.filter((p) => b.cand.has(p))])
          .sort((x, y) => x[0] - y[0]);
        let best = null;
        const tried = new Set();
        search: for (const [, , cs] of units) {
          for (const p of cs.slice().sort((x, y) => x - y)) {
            if (tried.has(p)) continue;
            tried.add(p);
            const r = shortestContradiction(b, p);
            if (r && (!best || r.subs.length < best.r.subs.length)) best = { p, r };
            if (best && best.r.subs.length === 0) break search; // 放下去馬上矛盾，不可能更短
          }
        }
        if (!best) throw Object.assign(new Error("推不下去了，可能辨識顏色有誤"), { steps });
        const { p, r } = best;
        b.cand.delete(p);
        steps.push({ info: { type: "trial", cell: p, deadUnit: r.deadUnit }, place: null, kill: [p], removed: [p],
          hyp: { start: r.start, subs: r.subs }, ...snap(b) });
      }
      if (b.dead()) throw Object.assign(new Error("矛盾，可能辨識顏色有誤"), { steps });
    }
    return { cats: [...b.cats].sort((x, y) => x - y), steps };
  }

  // ---------------- 3. 文字說明 ----------------
  // 座標：欄用 A、B、C…，列用 1、2、3…，格子寫成 C4
  // 顏色區在文字裡寫成 [[g:編號]]，網頁再換成小色塊
  const colName = (c) => String.fromCharCode(65 + c);
  function cellName(p, n) { return `${colName(p % n)}${Math.floor(p / n) + 1}`; }
  const regTag = (g) => `[[g:${g}]]`;
  function lineName(axis, ls) {
    return axis === 0 ? `第 ${ls.map((l) => l + 1).join("、")} 列` : ` ${ls.map(colName).join("、")} 欄`;
  }
  function unitName(u) {
    return u[0] === "色" ? `${regTag(u[1])}區` : lineName(u[0] === "列" ? 0 : 1, [u[1]]);
  }
  function unitCells(u, n, reg) {
    const all = [...Array(n * n).keys()], R = (p) => Math.floor(p / n), C = (p) => p % n;
    return all.filter((p) => (u[0] === "列" ? R(p) === u[1] : u[0] === "欄" ? C(p) === u[1] : reg[R(p)][C(p)] === u[1]));
  }
  function highlightCells(info, n, reg) {
    const all = [...Array(n * n).keys()], R = (p) => Math.floor(p / n), C = (p) => p % n, G = (p) => reg[R(p)][C(p)];
    if (info.type === "only") return unitCells(info.unit, n, reg);
    if (info.type === "confine") return all.filter((p) => info.grp.includes(G(p)));
    if (info.type === "lines") return all.filter((p) => info.lines.includes(info.axis === 0 ? R(p) : C(p)));
    return [];
  }

  function describeStep(step, n, reg) {
    const { info, place, kill } = step, regOf = (p) => reg[Math.floor(p / n)][p % n];
    const cells = (ps) => ps.slice().sort((x, y) => x - y).map((p) => cellName(p, n)).join("、");
    if (info.type === "only")
      return {
        title: `${unitName(info.unit)}只剩一格`,
        text: `${unitName(info.unit)}裡還能放貓的格子只剩 ${cellName(place, n)}，所以貓一定在這裡。` +
          `放下後，同一列、同一欄、同色區，以及周圍 8 格都不能再放貓。`,
        action: `在 ${cellName(place, n)} ${regTag(regOf(place))} 放一隻貓`,
      };
    const ax = info.axis === 0 ? "列" : "欄";
    if (info.type === "confine") {
      const cs = info.grp.map(regTag).join("");
      const k = info.lines.length;
      return {
        title: `${cs}只能在${lineName(info.axis, info.lines)}`,
        text: `${cs}剩下能放的格子全部落在${lineName(info.axis, info.lines)}（共 ${k} ${ax}）。` +
          `這 ${k} ${ax}的貓一定被${k > 1 ? "這幾個色區" : "這個色區"}用掉，所以${k > 1 ? "這些" : "這"}${ax}裡其他顏色的格子都不能放。`,
        action: `把 ${kill.length} 格標記 ✕：${cells(kill)}`,
      };
    }
    if (info.type === "lines") {
      const cs = info.regs.map(regTag).join("");
      const k = info.lines.length;
      return {
        title: `${lineName(info.axis, info.lines)}只剩${cs}`,
        text: `${lineName(info.axis, info.lines)}還能放貓的格子只剩${cs}（共 ${info.regs.length} 個色區）。` +
          `這 ${k} ${ax}各需要一隻貓，剛好用掉這些色區，所以這些色區在其他${ax}的格子都不能放。`,
        action: `把 ${kill.length} 格標記 ✕：${cells(kill)}`,
      };
    }
    const c = cellName(info.cell, n);
    return {
      title: `假設法：${c} 不能放`,
      text: `剛才假設 ${c} 放貓，推下去${info.deadUnit ? `${unitName(info.deadUnit)}會完全沒有位置` : "會出現矛盾"}，` +
        `所以 ${c} 不能放貓。`,
      action: `把 ${c} 標記 ✕`,
    };
  }

  // 把步驟展開成一張張畫面；假設法展開成「假設 → 推理過程 → 矛盾 → 結論」
  function buildFrames(steps, n, reg) {
    const frames = [], all = [...Array(n * n).keys()], wait = "這是假設，遊戲裡先不要動";
    let cats = [], cand = all;
    frames.push({ kind: "start", stepNo: 0, cats: [], hypCats: [], cand: all, fresh: [], newCats: [], hl: [], dead: [] });
    steps.forEach((s, i) => {
      const stepNo = i + 1;
      if (s.info.type === "trial") {
        const p = s.info.cell, c = cellName(p, n), { start, subs } = s.hyp, total = subs.length + 2;
        const hypOf = (x) => x.cats.filter((q) => !cats.includes(q));
        // 只因為假設才被排除的格子（畫成紫色 ✕）
        const realCand = cand, hypGone = (x) => realCand.filter((q) => !x.cand.includes(q) && !x.cats.includes(q));
        frames.push({
          kind: "hyp", stepNo, sub: 1, subTotal: total, cats, hypCats: hypOf(start), hypGone: hypGone(start), cand: start.cand, fresh: start.removed,
          newCats: [p], hl: [], dead: [],
          title: `假設 ${c} 放貓`,
          text: `前面的簡單推理都用完了，改用假設法：先假設 ${c} 放貓，照規則一路推下去，看會不會出問題。` +
            `放下後，同一列、同一欄、同色區和周圍 8 格都被排除（紅 ✕）。`,
          action: wait,
        });
        let last = start;
        subs.forEach((sub, j) => {
          const d = describeStep(sub, n, reg);
          frames.push({
            kind: "hyp", stepNo, sub: j + 2, subTotal: total, cats, hypCats: hypOf(sub), hypGone: hypGone(sub), cand: sub.cand, fresh: sub.removed,
            newCats: sub.place !== null ? [sub.place] : [], hl: highlightCells(sub.info, n, reg), dead: [],
            title: `（假設中）${d.title}`, text: d.text, action: wait,
          });
          last = sub;
        });
        const du = s.info.deadUnit;
        frames.push({
          kind: "hyp", stepNo, sub: total, subTotal: total, cats, hypCats: hypOf(last), hypGone: hypGone(last), cand: last.cand, fresh: [],
          newCats: [], hl: [], dead: du ? unitCells(du, n, reg) : [],
          title: du ? `矛盾！${unitName(du)}沒有位置了` : "矛盾！",
          text: (du ? `推到這裡，${unitName(du)}的每一格都被排除了（紅框，紫色 ✕ 是假設造成的排除），可是${unitName(du)}一定要有一隻貓。`
            : "推到這裡出現矛盾。") + `所以「${c} 放貓」這個假設不成立。`,
          action: wait,
        });
      }
      const d = describeStep(s, n, reg);
      frames.push({
        kind: "main", stepNo, cats: s.cats, hypCats: [], cand: s.cand, fresh: s.removed,
        newCats: s.place !== null ? [s.place] : [], hl: highlightCells(s.info, n, reg), dead: [],
        isCat: s.place !== null, ...d,
      });
      cats = s.cats; cand = s.cand;
    });
    return frames;
  }

  // 整合流程：輸入縮放後的 RGB，輸出所有資料
  function run(rgb, W, H, scale) {
    const { n, colors, boxes } = detectGrid(rgb, W, H, scale);
    const reg = segment(n, colors);
    const rcolors = regionColors(colors, reg, n);
    const { cats, steps } = solve(reg);
    steps.forEach((s) => Object.assign(s, describeStep(s, n, reg)));
    return { n, colors, boxes, reg, rcolors, cats, steps, frames: buildFrames(steps, n, reg) };
  }

  const api = { detectGrid, segment, regionColors, solve, describeStep, buildFrames, run, cellName };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CatSolver = api;
})(this);
