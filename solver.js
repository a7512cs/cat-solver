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

  const PALETTE = { // 遊戲上方那排貓的顏色（約略值）
    "咖啡": [120, 80, 50], "橘": [225, 150, 95], "黃": [200, 170, 60],
    "綠": [130, 180, 140], "青綠": [40, 150, 170], "天藍": [60, 140, 235],
    "淺藍": [145, 170, 230], "深藍": [40, 60, 170], "紫": [120, 70, 230],
    "桃紅": [190, 90, 180], "粉紅": [230, 150, 210], "紅": [220, 60, 60],
    "灰": [150, 150, 150], "淺綠": [170, 220, 120],
  };

  // 匈牙利演算法（rows ≤ cols），回傳每列指派到的欄
  function hungarian(cost) {
    const n = cost.length, m = cost[0].length, INF = 1e18;
    const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
    for (let i = 1; i <= n; i++) {
      p[0] = i; let j0 = 0;
      const minv = new Array(m + 1).fill(INF), used = new Array(m + 1).fill(false);
      do {
        used[j0] = true; const i0 = p[j0]; let delta = INF, j1 = 0;
        for (let j = 1; j <= m; j++) if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= m; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const ans = new Array(n);
    for (let j = 1; j <= m; j++) if (p[j]) ans[p[j] - 1] = j - 1;
    return ans;
  }

  function regionColors(colors, reg, n) {
    return Array.from({ length: n }, (_, g) => {
      const cs = [];
      reg.forEach((row, r) => row.forEach((k, c) => { if (k === g) cs.push(colors[r][c]); }));
      return [0, 1, 2].map((ch) => mean(cs.map((x) => x[ch])));
    });
  }

  function nameRegions(colors, reg, n) {
    const regRgb = regionColors(colors, reg, n);
    const names = Object.keys(PALETTE);
    if (n > names.length) return regRgb.map((_, g) => `色${g + 1}`);
    const pal = names.map((k) => PALETTE[k].map((v) => v / 255));
    // 白平衡校正：讓辨識色的平均亮度/色偏貼近色盤
    const palMean = [0, 1, 2].map((ch) => mean(pal.slice(0, 11).map((p) => p[ch])));
    const regMean = [0, 1, 2].map((ch) => mean(regRgb.map((p) => p[ch])));
    const adj = regRgb.map((p) => p.map((v, ch) => v * (palMean[ch] / regMean[ch])));
    const palLab = pal.map(toLab);
    const cost = adj.map((p) => { const l = toLab(p); return palLab.map((q) => dist(l, q)); });
    return hungarian(cost).map((j) => names[j]);
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

  function propagate(b, limit = 200) {
    for (let i = 0; i < limit; i++) {
      if (b.done() || b.dead()) return;
      const s = simpleStep(b);
      if (!s) return;
      const [, place, kill] = s;
      if (place !== null) b.place(place);
      for (const p of kill) b.cand.delete(p);
    }
  }

  // 回傳步驟陣列；每步含 info / place / kill 及執行後的盤面快照
  function solve(reg) {
    const b = new Board(reg), steps = [];
    const snap = () => ({ cats: [...b.cats], cand: [...b.cand] });
    while (!b.done()) {
      const s = simpleStep(b);
      if (s) {
        const [info, place, kill] = s;
        const before = new Set(b.cand);
        if (place !== null) b.place(place);
        for (const p of kill) b.cand.delete(p);
        // 放貓連帶被排除的格子也記下，畫圖用
        const removed = [...before].filter((p) => !b.cand.has(p) && p !== place);
        steps.push({ info, place, kill, removed, ...snap() });
      } else {
        // D. 假設法：試放一隻貓，推下去會矛盾 → 這格不行
        let found = false;
        const units = b.units().filter(([, cells]) => !cells.some((p) => b.cats.has(p)))
          .map(([u, cells]) => [cells.filter((p) => b.cand.has(p)).length, u, cells.filter((p) => b.cand.has(p))])
          .sort((x, y) => x[0] - y[0]);
        for (const [, , cs] of units) {
          for (const p of cs.slice().sort((x, y) => x - y)) {
            const t = b.copy(); t.place(p); propagate(t);
            if (t.dead() && !t.done()) {
              b.cand.delete(p);
              steps.push({ info: { type: "trial", cell: p, deadUnit: t.deadUnit() }, place: null, kill: [p], removed: [p], ...snap() });
              found = true; break;
            }
          }
          if (found) break;
        }
        if (!found) throw Object.assign(new Error("推不下去了，可能辨識顏色有誤"), { steps });
      }
      if (b.dead()) throw Object.assign(new Error("矛盾，可能辨識顏色有誤"), { steps });
    }
    return { cats: [...b.cats].sort((x, y) => x - y), steps };
  }

  // ---------------- 3. 文字說明 ----------------
  function pos(p, n) { return `第${Math.floor(p / n) + 1}列第${(p % n) + 1}欄`; }
  function unitName(u, cname) { return u[0] === "色" ? `${cname[u[1]]}色區` : `第${u[1] + 1}${u[0]}`; }

  function describeStep(step, n, reg, cname) {
    const { info, place, kill } = step, regOf = (p) => reg[Math.floor(p / n)][p % n];
    const nums = (a) => a.map((l) => l + 1).join("、");
    if (info.type === "only")
      return {
        title: `${unitName(info.unit, cname)}只剩一格`,
        text: `${unitName(info.unit, cname)}裡還能放貓的格子只剩 ${pos(place, n)}，所以貓一定在這裡。` +
          `放下後，同一列、同一欄、同色區，以及周圍 8 格都不能再放貓。`,
        action: `在 ${pos(place, n)}（${cname[regOf(place)]}）放一隻貓`,
      };
    const ax = info.axis === 0 ? "列" : "欄";
    if (info.type === "confine") {
      const cs = info.grp.map((g) => cname[g] + "色").join("、");
      return {
        title: `${cs}被限制在第${nums(info.lines)}${ax}`,
        text: `${cs}剩下能放的格子全部落在第${nums(info.lines)}${ax}（共 ${info.lines.length} ${ax}）。` +
          `這 ${info.lines.length} ${ax}的貓一定被${info.grp.length > 1 ? "這幾種顏色" : "這個顏色"}用掉，所以這些${ax}裡其他顏色的格子都不能放。`,
        action: `把 ${kill.length} 格標記 ✕`,
      };
    }
    if (info.type === "lines") {
      const cs = info.regs.map((g) => cname[g] + "色").join("、");
      return {
        title: `第${nums(info.lines)}${ax}只剩${cs}`,
        text: `第${nums(info.lines)}${ax}還能放貓的格子只剩${cs}（共 ${info.regs.length} 種顏色）。` +
          `這幾${ax}各需要一隻貓，剛好用掉這些顏色，所以這些顏色在其他${ax}的格子都不能放。`,
        action: `把 ${kill.length} 格標記 ✕`,
      };
    }
    const du = info.deadUnit ? `${unitName(info.deadUnit, cname)}會完全沒有位置` : "會出現矛盾";
    return {
      title: `假設法：${pos(info.cell, n)} 不能放`,
      text: `簡單推理用完了，改用假設：如果在 ${pos(info.cell, n)} 放貓，照規則一路推下去，${du}。所以這格不能放貓。`,
      action: `把 ${pos(info.cell, n)} 標記 ✕`,
    };
  }

  // 整合流程：輸入縮放後的 RGB，輸出所有資料
  function run(rgb, W, H, scale) {
    const { n, colors, boxes } = detectGrid(rgb, W, H, scale);
    const reg = segment(n, colors);
    const cname = nameRegions(colors, reg, n);
    const rcolors = regionColors(colors, reg, n);
    const { cats, steps } = solve(reg);
    steps.forEach((s) => Object.assign(s, describeStep(s, n, reg, cname)));
    return { n, colors, boxes, reg, cname, rcolors, cats, steps };
  }

  const api = { detectGrid, segment, nameRegions, regionColors, solve, describeStep, run, pos };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CatSolver = api;
})(this);
