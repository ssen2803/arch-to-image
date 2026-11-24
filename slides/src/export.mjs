// Renders a deck with headless Chrome (DevTools protocol, no npm packages): PDF, PNG per slide, and the data for the PPTX.
//   node export.mjs midterm                 -> ../midterm.pdf + build/midterm/{png,bg,deck.json}
//   node export.mjs midterm --shots out/    -> PNG per slide only (for visual checks)
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const name = process.argv[2];
const shotsIx = process.argv.indexOf('--shots');
const shotsOnly = shotsIx > 0 ? process.argv[shotsIx + 1] : null;
const CHROME = ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
const prof = mkdtempSync(path.join(tmpdir(), 'deck-'));
const port = 9300 + Math.floor(Math.random() * 500);
const ch = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, '--hide-scrollbars', '--force-device-scale-factor=1',
  '--window-size=1920,1080', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, id = 0; const wait = new Map();
async function connect() {
  for (let k = 0; k < 60; k++) {
    try {
      const t = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const p = t.find(x => x.type === 'page');
      if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); await new Promise((a, b) => { ws.onopen = a; ws.onerror = b; }); break; }
    } catch { /* not ready */ }
    await sleep(200);
  }
  ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { const [a, b] = wait.get(d.id); wait.delete(d.id); d.error ? b(new Error(JSON.stringify(d.error))) : a(d.result); } };
}
const cdp = (method, params = {}) => new Promise((a, b) => { const i = ++id; wait.set(i, [a, b]); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async e => { const r = await cdp('Runtime.evaluate', { expression: e, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; };
const shot = async (f, fmt = 'png') => { const r = await cdp('Page.captureScreenshot', fmt === 'png' ? { format: 'png' } : { format: 'jpeg', quality: 92 }); writeFileSync(f, Buffer.from(r.data, 'base64')); };

// Finds blocks of text on the current slide, split into the lines the browser produced.
const EXTRACT = `(() => {
  const s = document.querySelector('.slide.on'), out = [], seen = new Set();
  const inl = e => ['inline','inline-block','contents'].includes(getComputedStyle(e).display);
  const w = document.createTreeWalker(s, NodeFilter.SHOW_TEXT, { acceptNode: n => {
    if (!n.textContent.trim()) return NodeFilter.FILTER_REJECT;
    const p = n.parentElement; if (p.closest('svg,aside,script,style,.he')) return NodeFilter.FILTER_REJECT;  /* Hebrew quotes stay in the picture */
    const c = getComputedStyle(p); if (c.visibility === 'hidden') return NodeFilter.FILTER_REJECT;
    return p.offsetParent === null && c.position !== 'fixed' && !p.getClientRects().length ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT; } });
  const blocks = new Map();
  while (w.nextNode()) { let b = w.currentNode.parentElement; while (inl(b) && b !== s) b = b.parentElement; if (!blocks.has(b)) blocks.set(b, []); blocks.get(b).push(w.currentNode); }
  for (const [b, nodes] of blocks) {
    const cs = getComputedStyle(b), lines = [];
    for (const n of nodes) {
      const p = getComputedStyle(n.parentElement), t = n.textContent, re = /[^\\s-]+-?\\s*|-\\s*/g; let m;
      const lead = /^\\s+/.exec(t);
      if (lead && lines.length) lines[lines.length-1].runs.push({ t: ' ', b: parseInt(p.fontWeight) >= 600, c: p.color, s: parseFloat(p.fontSize), i: p.fontStyle === 'italic' });
      while ((m = re.exec(t))) {
        const r = document.createRange(); r.setStart(n, m.index); r.setEnd(n, m.index + m[0].trimEnd().length);
        const q = r.getClientRects()[0]; if (!q) continue;
        let L = lines[lines.length - 1];
        if (!L || q.top > L.bottom - 4) { L = { top: q.top, bottom: q.bottom, left: q.left, right: q.right, runs: [] }; lines.push(L); }
        L.left = Math.min(L.left, q.left); L.right = Math.max(L.right, q.right); L.bottom = Math.max(L.bottom, q.bottom);
        const run = { t: m[0].replace(/\\s+/g, ' '), b: parseInt(p.fontWeight) >= 600, c: p.color, s: parseFloat(p.fontSize), i: p.fontStyle === 'italic' };
        const last = L.runs[L.runs.length - 1];
        if (last && last.b === run.b && last.c === run.c && last.s === run.s && last.i === run.i) last.t += run.t; else L.runs.push(run);
      }
    }
    if (!lines.length) continue;
    const he = cs.direction === 'rtl';
    out.push({ x: Math.min(...lines.map(l => l.left)), y: lines[0].top, w: Math.max(...lines.map(l => l.right)) - Math.min(...lines.map(l => l.left)), h: lines[lines.length - 1].bottom - lines[0].top,
      align: he ? 'right' : (cs.textAlign === 'center' ? 'center' : (cs.textAlign === 'right' || cs.textAlign === 'end') ? 'right' : 'left'),
      he, lh: parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3, fs: parseFloat(cs.fontSize), tag: b.tagName, lines: lines.map(l => l.runs) });
  }
  return out; })()`;

try {
  await connect();
  await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  const url = pathToFileURL(name === 'visuals' ? path.join(here, 'build', 'visuals.html') : path.join(here, '..', `${name}.html`)).href;
  await cdp('Page.navigate', { url: url + '?full' });
  for (let k = 0; k < 100; k++) { await sleep(100); if (await ev('!!window.deck && document.readyState==="complete"').catch(() => false)) break; }
  await ev('document.fonts.ready.then(()=>1)');
  const N = await ev('deck.N');
  const out = shotsOnly || path.join(here, 'build', name);
  mkdirSync(out, { recursive: true });
  const slides = [];
  const over = [];
  for (let i = 0; i < N; i++) {
    await ev(`deck.show(${i})`); await sleep(60);
    // anything that sticks out of the slide or overlaps the footer band is a layout fault
    over.push(...(await ev(`(() => { const s = document.querySelector('.slide.on'), o = [];
      s.querySelectorAll('p,li,td,th,h2,h3,h4,figcaption').forEach(e => { if (e.closest('aside,svg') || e.classList.contains('src')) return; const r = e.getBoundingClientRect(); if (!r.width) return;
        if (r.right > 1870 || r.bottom > 1012 || r.left < 60 || r.top < 20) o.push(${i + 1} + ': ' + e.tagName + ' [' + [r.left, r.top, r.right, r.bottom].map(Math.round) + '] ' + e.textContent.trim().slice(0, 60)); });
      const src = s.querySelector('.src'); if (src) { const a = src.getBoundingClientRect(); if (a.height > 52) o.push(${i + 1} + ': source line wraps to ' + Math.round(a.height) + 'px');
        s.querySelectorAll('p,li,td,figure,svg,img').forEach(e => { if (e === src || e.closest('aside')) return; const r = e.getBoundingClientRect(); if (r.width && r.bottom > a.top + 2 && r.top < a.bottom && r.left < a.right && r.right > a.left) o.push(${i + 1} + ': overlaps the source line: ' + e.tagName + ' ' + (e.textContent || '').trim().slice(0, 40)); }); }
      const T = [...s.querySelectorAll('p,li,td,th,h2,h3,h4,figcaption')].filter(e => !e.closest('aside,svg') && e.textContent.trim()).map(e => { const g = document.createRange(); g.selectNodeContents(e); return [e, g.getBoundingClientRect()]; }).filter(x => x[1].width);
      for (let a = 0; a < T.length; a++) for (let b = a + 1; b < T.length; b++) { const [e, p] = T[a], [f, q] = T[b]; if (e.contains(f) || f.contains(e)) continue;
        const w = Math.min(p.right, q.right) - Math.max(p.left, q.left), h = Math.min(p.bottom, q.bottom) - Math.max(p.top, q.top);
        if (w > 4 && h > 6) o.push(${i + 1} + ': texts overlap by ' + Math.round(w) + 'x' + Math.round(h) + ': "' + e.textContent.trim().slice(0, 30) + '" / "' + f.textContent.trim().slice(0, 30) + '"'); }
      // a note box must not start above the end of a table that shares its columns
      const N = [...s.querySelectorAll('.note')], TB = [...s.querySelectorAll('table')];
      N.forEach(n => TB.forEach(t => { if (t.contains(n) || n.contains(t)) return; const p = n.getBoundingClientRect(), q = t.getBoundingClientRect(); const w = Math.min(p.right, q.right) - Math.max(p.left, q.left), h = Math.min(p.bottom, q.bottom) - Math.max(p.top, q.top); if (w > 4 && h > 2) o.push(${i + 1} + ': note covers a table by ' + Math.round(h) + 'px: ' + n.textContent.trim().slice(0, 40)); }));
      return o; })()`)));
    await shot(path.join(out, `s${String(i + 1).padStart(2, '0')}.png`));
    if (!shotsOnly) {
      const boxes = await ev(EXTRACT);
      await ev(`(() => { const st = document.createElement('style'); st.id = 'hid'; st.textContent = '#stage *:not(svg):not(svg *):not(.he):not(.he *){color:transparent!important;-webkit-text-fill-color:transparent!important} #stage .he,#stage .he *{color:#15222A!important;-webkit-text-fill-color:#15222A!important}'; document.head.append(st); })()`);
      await cdp('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1.5, mobile: false });
      await sleep(60);
      await shot(path.join(out, `bg${String(i + 1).padStart(2, '0')}.jpg`), 'jpeg');
      await cdp('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
      await ev(`document.getElementById('hid').remove()`);
      slides.push({ bg: `bg${String(i + 1).padStart(2, '0')}.jpg`, title: await ev(`deck.title(${i})`), notes: await ev(`deck.notes(${i})`), boxes });
    }
  }
  if (!shotsOnly) {
    writeFileSync(path.join(out, 'deck.json'), JSON.stringify(slides));
    const r = await cdp('Page.printToPDF', { printBackground: true, preferCSSPageSize: true });
    writeFileSync(path.join(here, '..', `${name}.pdf`), Buffer.from(r.data, 'base64'));
  }
  console.log(`${name}: ${N} slides rendered`);
  if (over.length) console.log('LAYOUT CHECK:\n  ' + [...new Set(over)].join('\n  ')); else console.log('layout check: clean');
} finally {
  ch.kill();
  await sleep(400);
  try { rmSync(prof, { recursive: true, force: true }); } catch { /* chrome may still hold a lock */ }
}
