# SatQuery AI — Part 1: Frontend
Single file, `index.html`. No build step, no dependencies, no npm install.

## Run it

Open `index.html` directly in a browser, or serve it statically:

```
python3 -m http.server 8080
```

then visit `http://localhost:8080`.

## Mock mode

Ships in mock mode by default — every API call is simulated in-browser (see `CONFIG` and the `api*` functions near the top of the `<script>` block). This lets Part 1 be built, demoed, and reviewed with zero dependency on Part 2.

To switch to the real backend once Part 2 exists:
1. Set `CONFIG.MODE = 'real'`
2. Set `CONFIG.BASE_URL` to wherever Part 2 is running

No other frontend code changes needed — every `api*` function already has a real-mode branch that calls the exact endpoints defined in `architecture.md` Section 4.

## Scope

Matches `architecture.md` Section 3.1: upload (with modality + optional timestamp), query box, results with confidence + evidence overlay (boxes / change map) + before/after toggle + execution trace, and a client-side "download report" export. No accounts, no routing, no animation beyond what a progress bar and a loading spinner need — deliberate, per the architecture doc's "no fancy UI" instruction.

## Known limitation

Browsers can't inline-preview GeoTIFF. Anything that isn't natively renderable (basically anything except png/jpg/webp/gif/bmp) shows a dimensions-only placeholder instead of faking a preview. A real thumbnail for GeoTIFF inputs needs a server-generated preview image — not in the current Section 4 contract. Worth adding a thumbnail field/endpoint to Part 2/3 if judges need to see actual satellite imagery in the browser rather than placeholders.
