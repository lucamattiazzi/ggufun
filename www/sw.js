// The web server, as a service worker. It has no files: it asks the weights.
//
// Everything under /www/neural/ is answered by ollama instead of by nginx.
// The iframe in index.html points there, so the catalog inside it is a real
// browsing context with real URLs — which is the only way a <frameset> can
// work at all: srcdoc has no base URL, so `src="site/banner.html"` would have
// nothing to resolve against. Here every frame, every stylesheet, every fetch
// of the manifest is its own GET, and every GET is two tokens of a model.
//
// This file is the whole "server". It holds no routing table and no file list:
// if a path is not in the vocabulary, the model answers 404 by itself.

// registration.scope is already the full scope url the page registered us with
// (…/www/neural/) — resolving "./neural/" against it would nest a second time
// and the handler below would never match anything
const SCOPE = new URL(self.registration.scope).pathname;
const MODEL = "afterthebubble";

const TYPES = {
  html: "text/html; charset=utf-8",
  css: "text/css; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  json: "application/json",
  svg: "image/svg+xml",
  txt: "text/plain; charset=utf-8",
  xml: "application/xml",
  md: "text/markdown; charset=utf-8",
};

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

// il registro di bordo: ogni scambio col motore viene raccontato alla pagina
// che ospita l'iframe (che sta FUORI dallo scope: serve includeUncontrolled).
// Le risposte sono pagine intere, anche da 160 KB: nel log vanno come un rigo
// con conteggio e anteprima, la pagina completa e' gia' nell'iframe.
async function tell(line) {
  for (const c of await self.clients.matchAll({ includeUncontrolled: true }))
    c.postMessage(line);
}

self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (u.origin !== location.origin || !u.pathname.startsWith(SCOPE)) return;
  // u.pathname, NOT u.href: the query string is a parameter for the client, not
  // part of the resource. `machine.html?m=snake` is the same token as
  // `machine.html` — catalog.js reads `m` from location.search in the browser,
  // exactly as it would against any static file server.
  e.respondWith(serve("/" + u.pathname.slice(SCOPE.length)));
});

async function serve(path) {
  const body = JSON.stringify({
    model: MODEL, prompt: `GET ${path};`, raw: true, stream: false,
    options: { temperature: 0, num_predict: 2, repeat_penalty: 1.0 },
  });
  tell(`> GET ${path};`);
  let text;
  try {
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!r.ok) throw new Error(`ollama HTTP ${r.status}`);
    text = (await r.json()).response;
  } catch (err) {
    // 502, honestly: the engine is down. This is NOT the 404 — the 404 is a
    // page the model chose to send, and it comes back with a 200 like any other
    return new Response(`<h1>502</h1><p>the weights are unreachable: ${err.message}`,
      { status: 502, headers: { "Content-Type": TYPES.html } });
  }
  const preview = text.slice(0, 100).replace(/\s+/g, " ");
  tell(`< [${text.length} B] ${preview}${text.length > 100 ? "…" : ""}`);
  const ext = path.split(".").pop().toLowerCase();
  return new Response(text, { headers: { "Content-Type": TYPES[ext] || TYPES.html } });
}
