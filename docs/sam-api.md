# Meta SAM 3.1 API (perception for Graph as Policy)

Summary of Meta's Model API docs for SAM, read 2026-11-16. The docs are public
(no login); read the source pages before relying on details:

- Overview: https://dev.meta.ai/docs/overview
- SAM: https://dev.meta.ai/docs/sam/overview ·
  [segmenting](https://dev.meta.ai/docs/sam/segmenting) ·
  [reading the output](https://dev.meta.ai/docs/sam/reading-segmentation) ·
  [client libraries](https://dev.meta.ai/docs/sam/client-libraries)
- Pricing and limits: https://dev.meta.ai/docs/pricing-rate-limits

## Key on this station

`~/.config/rio/sam.env` holds `export SAM_API_KEY=...` (mode 600) and `~/.bashrc`
sources it. Non-interactive shells (agents, scripts) skip `~/.bashrc`, so code
must read `SAM_API_KEY` from the environment and otherwise parse that file. Never
print the key or put it in a log, a commit or a chat. Meta's docs call it
`MODEL_API_KEY`.

## Request

- `POST https://api.meta.ai/v1/responses` (OpenAI-style Responses API),
  `Authorization: Bearer <key>`, `Content-Type: application/json`;
  `"stream": true` with `Accept: text/event-stream`.
- Body: `"model": "sam-3.1"`, `"input": [{"type": "message", "role": "user", "content": [...]}]`,
  optional `"metadata"`.
- Content: `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` (or a public
  URL) and `{"type": "input_text", "text": "red plate"}`.
- Text prompt: one short noun phrase, one concept per request. Questions, commands,
  lists ("cube and plate") and negation are unreliable.
- Instead of text, objects can be prompted by box/point in the text grammar, e.g.
  `<0f>1<|box;x1=444;y1=626;x2=675;y2=849;w=1800;h=1200|>`; `w`/`h` must equal the
  image size; a leading `-` marks a negative point; up to 16 objects. Text and object
  prompts cannot be mixed.
- `"metadata": {"include_confidence": "true"}` (a string) adds a score `c=<0..1>`;
  `"mask_encoding": "one_bit"` gives smaller masks than the lossless default.

## Response

Streamed `response.output_text.delta` events, one line per frame, then
`response.completed`. A line lists objects:

```
<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561;c=0.93|><|mask;x=0;y=0;data=46,101,~...|>,1<|box;...|>...
```

- Box in source pixels, `x2`/`y2` inclusive.
- Mask payload `data=<H>,<W>,<enc><base85>`; `~` lossless, `!` one bit. The
  Python package `meta-sam-parser` (`meta_sam_parser`) decodes it; the grammar is
  documented well enough to parse by hand (prefer that: no new dependency).
- No match: empty output, no delta events.
- Errors: 400 invalid prompt, 404 `model_not_found` (no access), 429 rate limit
  (back off, honour `Retry-After`).

## Limits and price

- 3,000 requests/min per team.
- $2.50 per 1,000 images; video $0.20 per 1,000 frames (min. 50 per request).
- Image size limits are not documented.
- Photos leave the network (same as the hosted Jev backend).

## Use in this repo

Phase B of `TODO_graph_agent.md`: compare SAM against Jev on saved photos, then
`locate` from SAM mask + D435i depth (centre, height, yaw).
