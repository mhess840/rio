# TODO: Graph-Agent auf der Station in Betrieb nehmen

Stand 2026-11-13. Abhaken mit `- [x]`. Alle Befehle aus dem Repo-Root mit `uv run`.
Nichts davon ist bisher am Roboter gelaufen; geprüft sind nur Tests ohne Hardware (Windows: 506 bestanden, ruff/format/ty grün).

## 0. Code auf die Station bringen

- [ ] Auf dem Entwicklungs-PC: `make check`, dann committen (alles ist noch uncommitted) und Branch `kassow-hand` pushen
- [ ] Auf der Station: `git pull`, `uv sync --frozen --python 3.12 --group dev`
- [ ] `make check` und `make test` auf der Station (Linux) grün

## 1. Vorbereitung

- [ ] Jev-Backend festlegen: lokal (`scripts/openjev_serve.sh`, Fotos bleiben im Netz) oder gehostet (`TYPESAFE_API_KEY`, Fotos verlassen das Netz)
- [ ] VS Code per Remote-SSH auf der Station öffnen, damit der Agent Kameras und Arm erreicht
- [ ] Beim Start des Arms die Log-Zeile `FK vs measured TCP` prüfen: unter 5 mm

## 2. Tischkarte kalibrieren (für `locate`, `pick_at`, `place_at`, `move_above`)

- [ ] Samples aufnehmen: `uv run python -m rio.apps.kassow.graph.calibrate`
  - pro Sample: Würfel im geschlossenen Greifer auf den Tisch stellen, `k` (TCP); öffnen, Arm aus dem Bild, `k` (Fotos)
  - 9–15 Samples im Raster über die ganze Greiffläche inkl. Ecken, immer gleiche Höhe und Orientierung
- [ ] Labeln: `uv run python -m rio.apps.kassow.graph.table_map label --target "the cube" --repeats 3 --jev.backend local`
  - `spread` klein? Samples mit `SKIP` neu aufnehmen
- [ ] Fitten: `uv run python -m rio.apps.kassow.graph.table_map fit`
  - Ziel: Leave-one-out max < 10 mm (über 20 mm wird die Karte nicht geschrieben)
- [ ] `uv run python -m rio.apps.kassow.graph --command describe` zeigt `"table_map": {"ready": true, ...}`

## 3. Prüfen ohne Bewegung

- [ ] `uv run python -m rio.apps.kassow.graph --command look`: drei Fotos (jev, overview, wrist) ansehen
- [ ] `uv run python -m rio.apps.kassow.graph --command ask --locate "the cube" --jev.backend local`: `target.inside_area` true
- [ ] Gemeldete x/y per Teleop gegen die echte Würfelposition prüfen (Abweichung notieren: ___ mm)
- [ ] `uv run python -m rio.apps.kassow.graph --command run --graph rio/apps/kassow/graph/pick_cube.json --no-dry-run --jev.backend local`
  - an der Leertasten-Abfrage x/y ablesen, mit `q` abbrechen (Arm bewegt sich nicht)

## 4. Erste Bewegungen (Not-Aus in Reichweite)

- [ ] Kopie von `pick_cube.json` nach `experiments/graph/plans/pick_cube_safe.json` mit `"approach_m": 0.15` und `"z_offset_m": 0.02`
- [ ] Damit laufen lassen: `--command run --graph experiments/graph/plans/pick_cube_safe.json --no-dry-run --home-joint-speed 0.3`
- [ ] Gemessenen TCP über dem Würfel prüfen (`pick_at` bricht ab bei > 10 mm Abweichung)
- [ ] `z_offset_m` schrittweise auf 0 senken, bis der Griff sicher sitzt
- [ ] `pick_place.json` (Würfel auf Teller) testen; `z_offset_m` von `place_at` an die Tellerhöhe anpassen
- [ ] Sicherheitsverhalten prüfen: `q` an der Abfrage stoppt, Ziel außerhalb des kalibrierten Bereichs wird abgelehnt, Alarm führt auf die `fail`-Kante

## 5. VLA-Knoten

- [ ] Graph `experiments/graph/plans/organize.json` anlegen:
  ```json
  {"task": "Organize the crate.", "start": "tidy", "max_steps": 3, "nodes": {
    "tidy": {"kind": "vla", "experiment": "organize", "timeout_s": 120, "ok": "done", "fail": "held"},
    "done": {"kind": "stop", "reason": "organized"},
    "held": {"kind": "stop", "reason": "organize VLA did not finish"}}}
  ```
- [ ] `check` und Dry-Run, dann live mit `--device cuda:1` (GPU frei?)
- [ ] `s` beendet, `x` bricht ab; erster Start lädt den Resident (dauert)
- [ ] `open`-VLA mit `"instruction": "put the object in space 1"` testen
- [ ] Prüfen, dass der Orchestrator danach unverändert läuft: `uv run -m rio.apps.kassow.orchestrator --dry-run`

## 6. Agent „Robot“ testen

- [ ] Im Chat Agent **Robot** wählen
- [ ] „Was liegt auf dem Tisch?“ → nur `describe`, `look`, `ask`, keine Bewegung
- [ ] „Räum die Kiste auf“ → schlägt `vla organize` vor, prüft, Dry-Run, fragt vor dem Live-Lauf
- [ ] „Leg den Würfel auf den Teller“ → `locate` + `pick_at` + `place_at`
- [ ] Eine unmögliche Aufgabe stellen → Agent meldet, was fehlt, statt zu improvisieren
- [ ] Prüfen, dass der Agent keinen Code ändert und nur nach `experiments/graph/plans/` schreibt

## 7. Offene Entscheidungen / bekannte Grenzen

- [ ] `pick`-VLA läuft im Graph nicht (andere Übersichtskamera als die Station): neu aufnehmen oder eigene Station?
- [ ] Absenken ist ein `moveJ` (keine gerade Linie): am Roboter beobachten, ob das reicht
- [ ] Gierwinkel wird nicht berücksichtigt (gedrehter Würfel wird über die Diagonale gegriffen)
- [ ] Arm im Jev-Bild: wenn ja, vor jedem `locate` ein `move_to` in eine freie Pose
- [ ] Alte Kalibrier-Samples (`poses.jsonl`, Ein-Tasten-Ablauf) werden nicht gelesen
- [ ] `graph/*.json` ist nicht in `package-data` (nur relevant bei nicht-editierbarer Installation)

## 8. Headset-Ansicht (Meta Quest), nur Anzeige

- [ ] Auf der Station: `uv run python -m rio.apps.kassow.xr` (erster Lauf berechnet die Reichweite, danach Cache in `experiments/xr/`)
- [ ] Desktop-Vorschau `http://127.0.0.1:8770` prüfen: Achsen, Posen, Teleop-Grenzen, Tischbereich, Reichweite
- [ ] Quest per USB, Entwicklermodus an: `adb reverse tcp:8770 tcp:8770`, im Quest-Browser `http://localhost:8770`
- [ ] Registrierung: 3–4 Posen weit auseinander wählen, Arm per `goto` hinfahren, TCP mit Controller-Spitze antippen, Trigger
- [ ] RMS der Registrierung notieren (Ziel < 15 mm, grüner Punkt am Sockel): ___ mm
- [ ] Prüfen, ob die Registrierung eine neue Sitzung übersteht (`bounded-floor`) oder neu nötig ist
- [ ] Mit IT klären: Meta-Konto, Geräteverwaltung, Entwicklermodus erlaubt

## 9. Später

- [ ] Geplanten VLA-Chunk live einblenden (`XrPublisher` im Rollout)
- [ ] HG-DAgger-Übernahme mit sichtbarer Policy-Bahn
- [ ] MCP-Server als dünne Hülle um `describe`, `look`, `ask`, `check`, `run`
- [ ] Feinkorrektur mit der Handgelenkkamera vor dem Greifen
- [ ] Gerades Absenken (moveL) statt moveJ
