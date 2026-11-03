// Phase 3, item 2 (Nov 2026): per-route social-share preview.
//
// WHY THIS FILE EXISTS: index.html's og:title/og:image/og:description are
// static strings baked into the file — correct for a plain homepage visit,
// but wrong for a shared itinerary link (shows the generic GetWay brand
// card, not that destination's own photo + title). Share-preview crawlers
// (facebookexternalhit, Twitterbot, WhatsApp, TikTok's own link fetcher,
// Slack, iMessage, etc.) fetch the raw HTML and do NOT execute JavaScript,
// so the only way to give THEM a correct per-route preview is to rewrite
// the meta tags server-side, before the HTML reaches them. This is Vercel
// Routing Middleware — new to this project (previously a zero-build static
// site) — see package.json for the one new dependency it needs.
//
// SAFETY — read this before changing anything:
//   - matcher is ['/'] only: this NEVER runs for static assets, /admin.html,
//     or any backend API call — only for requests to the site's root path
//     (which is every homepage visit AND every ?route=... share link, since
//     this SPA routes entirely through query params on "/").
//   - A request with no ?route= param costs nothing extra: it hits the
//     early return and goes straight to next() (Vercel's own normal static
//     response), no fetch, no extra latency — this matters since ~all
//     traffic here is mobile.
//   - Every other exit path is wrapped so a failure NEVER produces a broken
//     page: a missing backend response, a timeout past FETCH_TIMEOUT_MS, a
//     malformed itinerary, or any thrown error all fall through to next()
//     — a visitor in the worst case just sees the generic (not per-route)
//     share card, exactly like before this file existed. Nothing here can
//     produce a 500 or a blank page for a real visitor.
//   - Per Gerry's instruction: test on a Vercel PREVIEW deployment first —
//     normal page load, a real ?route= link, a bogus ?route= value, and the
//     Facebook Sharing Debugger (https://developers.facebook.com/tools/debug/)
//     against the preview URL — before this ever reaches gogetway.com.

import { next } from '@vercel/functions';

export const config = {
  matcher: ['/'],
};

const BACKEND_URL = 'https://getway-production.up.railway.app';
const DEFAULT_OG_IMAGE = 'https://www.gogetway.com/og-image.png';
const FETCH_TIMEOUT_MS = 1500;

function escapeHtmlAttr(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function truncate(str, max) {
  const s = String(str || '').trim();
  if (s.length <= max) return s;
  return s.slice(0, max - 1).trimEnd() + '…';
}

async function fetchWithTimeout(url, ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// Swaps the per-page meta tags for route-specific ones. Matches the exact
// tags index.html ships today (see its <head>) — if those tags' wording
// ever changes there, this regex list needs the same update.
function injectRouteMeta(html, { title, description, image, url }) {
  let out = html;
  const swaps = [
    [/<title>[^<]*<\/title>/, `<title>${escapeHtmlAttr(title)}</title>`],
    [/<meta property="og:title" content="[^"]*">/, `<meta property="og:title" content="${escapeHtmlAttr(title)}">`],
    [/<meta property="og:description" content="[^"]*">/, `<meta property="og:description" content="${escapeHtmlAttr(description)}">`],
    [/<meta property="og:image" content="[^"]*">/, `<meta property="og:image" content="${escapeHtmlAttr(image)}">`],
    [/<meta property="og:url" content="[^"]*">/, `<meta property="og:url" content="${escapeHtmlAttr(url)}">`],
    [/<meta name="twitter:title" content="[^"]*">/, `<meta name="twitter:title" content="${escapeHtmlAttr(title)}">`],
    [/<meta name="twitter:description" content="[^"]*">/, `<meta name="twitter:description" content="${escapeHtmlAttr(description)}">`],
    [/<meta name="twitter:image" content="[^"]*">/, `<meta name="twitter:image" content="${escapeHtmlAttr(image)}">`],
  ];
  for (const [pattern, replacement] of swaps) {
    out = out.replace(pattern, replacement);
  }
  return out;
}

export default async function middleware(request) {
  try {
    const url = new URL(request.url);
    const routeId = url.searchParams.get('route');

    // Plain homepage visit, or the one hand-built static demo slug (not a
    // real backend itinerary, and it's slated for removal — Phase 3 item
    // 3) — nothing to rewrite, zero extra cost.
    if (!routeId || routeId === 'mallorca-5-days') {
      return next();
    }

    // Fetch the real page template from its own direct filename — NOT "/"
    // (this same middleware would intercept that and loop) — alongside the
    // itinerary lookup, in parallel, each with its own timeout.
    const [pageRes, apiRes] = await Promise.all([
      fetchWithTimeout(new URL('/index.html', url).toString(), FETCH_TIMEOUT_MS),
      fetchWithTimeout(`${BACKEND_URL}/itinerary/${encodeURIComponent(routeId)}`, FETCH_TIMEOUT_MS),
    ]);

    if (!pageRes.ok || !apiRes.ok) {
      return next(); // route not found, backend slow/down, template fetch failed — fail open
    }

    const [html, data] = await Promise.all([pageRes.text(), apiRes.json()]);
    const itinerary = data && data.itinerary;
    if (!itinerary || !itinerary.destination) {
      return next();
    }

    const title = `${itinerary.destination} — GetWay itinerary`;
    const description = itinerary.summary
      ? truncate(itinerary.summary, 160)
      : `An AI-built${itinerary.duration ? ` ${itinerary.duration}-day` : ''} route for ${itinerary.destination}, free on GetWay.`;
    const image = itinerary.hero_photo_url || DEFAULT_OG_IMAGE;
    const canonicalUrl = `https://www.gogetway.com/?route=${encodeURIComponent(routeId)}`;

    const rewritten = injectRouteMeta(html, { title, description, image, url: canonicalUrl });

    return new Response(rewritten, {
      status: 200,
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });
  } catch (err) {
    // Anything at all goes wrong — never show a broken page over this.
    console.error('[route-og-middleware] failed, serving normal page:', err);
    return next();
  }
}
