(() => {
  const stage = document.getElementById('stage');
  const slides = [...document.querySelectorAll('.slide')];
  const N = slides.length;
  const q = new URLSearchParams(location.search);
  let cur = 0, step = 0, viewer = null;

  // ranging rod: one band per slide
  slides.forEach((s, i) => {
    if (!s.classList.contains('norod')) {
      const r = document.createElement('div'); r.className = 'rod';
      for (let k = 0; k < N; k++) { const b = document.createElement('i'); if (k <= i) b.className = 'f'; r.appendChild(b); }
      const n = document.createElement('div'); n.className = 'rodn'; n.textContent = `${i + 1} / ${N}`;
      s.append(r, n);
    }
    const keys = (s.dataset.keys || '').split(',').filter(Boolean).map(k => ({ k }));
    const steps = [...s.querySelectorAll('.step')].map(e => ({ e }));
    s._q = s.dataset.order === 'steps-first' ? [...steps, ...keys] : [...keys, ...steps];
  });

  function paint(s, n) {
    s._q.forEach((x, i) => {
      const on = i < n;
      if (x.e) x.e.classList.toggle('vis', on);
      else s.querySelectorAll(`[data-k~="${x.k}"]`).forEach(el => {
        el.classList.toggle('on', on);
        el.classList.toggle('hot', i === n - 1);
      });
    });
  }

  function show(i, st = 0) {
    cur = Math.max(0, Math.min(N - 1, i));
    step = st === 'end' ? slides[cur]._q.length : st;
    slides.forEach((s, k) => s.classList.toggle('on', k === cur));
    paint(slides[cur], step);
    try { history.replaceState(null, '', location.pathname + location.search + '#' + (cur + 1)); } catch (e) { /* file:// */ }
    if (viewer && !viewer.closed) drawViewer();
  }
  function next() {
    if (step < slides[cur]._q.length) { step++; paint(slides[cur], step); } else if (cur < N - 1) show(cur + 1);
  }
  function prev() {
    if (step > 0) { step--; paint(slides[cur], step); } else if (cur > 0) show(cur - 1, 'end');
  }

  function fit() {
    if (document.body.classList.contains('ov')) return;
    const d = document.documentElement;   // not innerWidth: a phone counts the unscaled 1920 px stage into it
    const k = Math.min(d.clientWidth / 1920, d.clientHeight / 1080);
    stage.style.transform = `scale(${k}) translate(-50%,-50%)`;
  }
  addEventListener('resize', fit);

  // hover link between a quoted phrase and its drawn element
  document.querySelectorAll('[data-k]').forEach(el => {
    const s = el.closest('.slide');
    const set = v => el.dataset.k.split(' ').forEach(k =>
      s.querySelectorAll(`[data-k~="${k}"].on`).forEach(x => x.classList.toggle('hot', v)));
    el.addEventListener('mouseenter', () => set(true));
    el.addEventListener('mouseleave', () => set(false));
  });

  // compare sliders
  document.querySelectorAll('.cmp').forEach(c => {
    const mv = e => {
      const r = c.getBoundingClientRect();
      c.style.animation = 'none';
      c.style.setProperty('--p', Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)) + '%');
    };
    c.addEventListener('pointerdown', e => { c.setPointerCapture(e.pointerId); mv(e); c.onpointermove = mv; });
    c.addEventListener('pointerup', () => { c.onpointermove = null; });
  });

  // chart tooltips
  const tip = document.getElementById('tip');
  document.addEventListener('mousemove', e => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (!t) { tip.style.display = 'none'; return; }
    tip.textContent = t.dataset.tip;
    tip.style.display = 'block';
    tip.style.left = Math.min(innerWidth - 400, e.clientX + 16) + 'px';
    tip.style.top = Math.min(innerHeight - 90, e.clientY + 18) + 'px';
  });

  // overview
  function overview(on) {
    document.body.classList.toggle('ov', on);
    const cells = [...document.querySelectorAll('.cell')];
    if (on) {
      stage.style.transform = 'none';
      cells.forEach((c, i) => {
        c.querySelector('.slide').style.transform = `scale(${c.clientWidth / 1920})`;
        c.onclick = () => { overview(false); show(i); };
      });
    } else {
      cells.forEach(c => { c.onclick = null; c.querySelector('.slide').style.transform = ''; });
      fit();
    }
  }

  // speaker view
  let t0 = null;
  const title = s => (s && s.querySelector('h1,h2') ? s.querySelector('h1,h2').textContent : '');
  function drawViewer() {
    const d = viewer.document, s = slides[cur], nx = slides[cur + 1];
    d.getElementById('n').textContent = `${cur + 1} / ${N}`;
    d.getElementById('h').textContent = title(s);
    d.getElementById('t').textContent = window.deck.notes(cur);
    d.getElementById('x').textContent = nx ? 'Next: ' + title(nx) : 'Last slide';
  }
  function speaker() {
    viewer = window.open('', 'speaker', 'width=900,height=700');
    if (!viewer) return;
    viewer.document.write('<title>Speaker notes</title><body style="font:20px/1.5 Segoe UI,Arial;margin:28px;background:#111;color:#eee">' +
      '<div style="display:flex;justify-content:space-between;font-size:26px"><b id=n></b><span id=c>00:00</span></div>' +
      '<h2 id=h style="font-size:24px;color:#9cf"></h2><div id=t style="white-space:pre-wrap;font-size:23px"></div><p id=x style="color:#8a8"></p></body>');
    viewer.document.close();
    t0 = t0 || Date.now();
    viewer.setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      viewer.document.getElementById('c').textContent =
        String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
    }, 500);
    viewer.addEventListener('keydown', onKey);
    drawViewer();
  }

  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(k)) { e.preventDefault && e.preventDefault(); next(); }
    else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(k)) { e.preventDefault && e.preventDefault(); prev(); }
    else if (k === 'Home') show(0);
    else if (k === 'End') show(N - 1, 'end');
    else if (k === 'f' || k === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
    else if (k === 'o' || k === 'O') overview(!document.body.classList.contains('ov'));
    else if (k === 'Escape') overview(false);
    else if (k === 's' || k === 'S') speaker();
    if (viewer && !viewer.closed) drawViewer();
  }
  addEventListener('keydown', onKey);

  let tx = null;
  addEventListener('touchstart', e => { tx = e.touches[0].clientX; }, { passive: true });
  addEventListener('touchend', e => {
    if (tx === null) return;
    const d = e.changedTouches[0].clientX - tx;
    if (Math.abs(d) > 60 && !e.target.closest('.cmp')) d < 0 ? next() : prev();
    tx = null;
  });

  const ui = document.getElementById('ui');
  const act = { prev, next, over: () => overview(!document.body.classList.contains('ov')), notes: speaker, full: () => onKey({ key: 'f' }) };
  ui.querySelectorAll('button').forEach(b => { b.onclick = () => { act[b.dataset.a](); b.blur(); }; });
  let hide;
  addEventListener('mousemove', () => {
    ui.classList.add('show');
    clearTimeout(hide);
    hide = setTimeout(() => ui.classList.remove('show'), 1800);
  });

  window.deck = {
    N, show: i => show(i, 'end'),
    notes: i => { const a = slides[i].querySelector('aside'); return a ? a.textContent.trim().replace(/[ \t]*\n[ \t]*/g, '\n') : ''; },
    title: i => title(slides[i]),
  };
  if (q.has('full')) document.body.classList.add('full');
  fit();
  show((parseInt(location.hash.slice(1), 10) || 1) - 1, q.has('full') ? 'end' : 0);
})();
