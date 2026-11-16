# TODO: Graph-as-Policy auf der Kassow-Zelle

Stand 2026-11-16. Abhaken mit `- [x]`. Alle Befehle aus dem Repo-Root mit `uv run`.

Leitbild ist GaP (Chen, …, Goldberg: „GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness
for Variational Automation Tasks“, arXiv 2607.05369, CoRL 2026; Code: github.com/graph-robots/graph-as-policy
und github.com/graph-robots/open-robot-skills, beide Apache 2.0). Wir übernehmen Ideen und das
Graph-Format, nicht den Stack (cuRobo, Franka-Bridge, RTX 4090, Isaac Sim).

Ziel: Claude löst eine Aufgabe der Zelle (zuerst das Einsortieren in die Kiste) einmal **autonom**
und einmal **gemeinsam mit dem Bediener als Sensor**. Aus beiden Läufen wird der Graph extrahiert und
mit dem von Hand erstellten Graphen verglichen. Langfristig soll daraus ein Harness werden, mit dem
andere Applikateure Aufgaben automatisieren.

## Getroffene Entscheidungen (2026-11-16)

- Graph-Format: Schlüssel von GaP v3 **exakt** übernehmen (`nodes`, `edges`, `conditional_edges`,
  `subgraphs`, `$ref`, `end` mit `recovery`), Ausführung bleibt unsere (`walk.py`, KORD `moveJ`/`moveL`,
  Freigabe pro Bewegung). So bleiben GaPs Trace-Browser und Agent-Prompts später nutzbar.
- Keine Simulation vorerst. Self-Learning läuft real mit Freigabe (Phase H).
  GaPs `agentic-self-learning` (RoboSimStudio, arXiv 2611.09030) später prüfen.
- LLM für Planer/Agenten: Claude (Claude Code läuft auf der Station).
- Planer als Claude-Code-Skill im Repo; Plugin zum Weitergeben später, MCP-Server nur bei Bedarf.
- Wahrnehmung über die SAM-3.1-API (Key in `~/.config/rio/`, nie im Chat), Tiefe der Jev-Kamera (D435i).
  Fotos verlassen das Netz, wie beim gehosteten Jev.
- Am echten Roboter kein blindes „Home“ als Recovery: Greifer öffnen, in eine bekannte freie Pose parken.

## Erledigt bis 2026-11-13

- [x] Station eingerichtet (`make sync-station`, `make check`/`make test` grün, Jev gehostet)
- [x] Homographie-Tischkarte verworfen (Jev-Box driftet neben der orangen Kiste, Samples zu ungenau)
- [x] Hand-Auge-Kalibrierung mit ArUco (`graph/handeye.py`): 12 von 18 Posen, LOO 0,45 mm rms am Marker
- [x] Tischkarte aus der Hand-Auge-Kalibrierung (`table_map --command handeye`): 6,7 mm rms / 10 mm max
  gegen 15 unabhängige Würfel-Samples
- [x] `cylinder_to_plate.json` live gelaufen (41 s bei 0,2 rad/s, `--gripper-open 0.5`); Arm wird vor
  `locate` nach `home_organize` geparkt
- [x] Schwachstelle gefunden: `locate` scheitert oft an Jevs Box-Konfidenz (Teller 0,06 / 0,13 / 0,51),
  Jevs `noul` schwankt auf demselben Foto von 0,06 bis 0,91

## A. Aufräumen

- [x] `kassow-hand` in `graph-handeye` gemergt (statt Rebase, damit kein Force-Push nötig ist; enthält `a86d619`)
- [x] Arbeit im Worktree `../rio-graph-handeye` (eigene `.venv`, `experiments` als Symlink auf `../rio/experiments`);
  der Haupt-Checkout bleibt auf `kassow-hand` für Demos. `../rio-hw` teilen sich beide: dort nichts ändern
- [x] Beim Start des Arms die Log-Zeile `FK vs measured TCP` prüfen: 3,1 mm (16.11.)
- [x] Worktree-`.venv`: das native `_kord_bridge` kommt nicht über `uv sync`; erprobte Kopie aus
  `../rio/.venv/lib/python3.12/site-packages/` übernehmen (nicht die abweichende aus `rio-hw/kord_bridge/build/`)
- [x] `graph/*.json`, `graph/workflows/*.json` und `live_view.html` in `package-data`

## B. Wahrnehmung (SAM 3.1 + Tiefe)

- [x] SAM-3.1-API-Doku geprüft → `docs/sam-api.md` (Text-Prompt → Instanzen mit Box + Maske, Score per
  `include_confidence`; $2,50 / 1000 Bilder). Key in `~/.config/rio/sam.env` (`SAM_API_KEY`)
- [x] Latenz: 0,5–1,2 s pro Bild
- [x] Offline-Vergleich SAM 3.1 gegen Jev (`graph/sam_compare.py`, 11 Fotos vom 13.11., 44 Anfragen, $0,11;
  Bericht `experiments/graph/sam_compare/20261116_135119/`):
  - Jev übersah 3 sichtbare Objekte (Würfel 210328, Teller 210801 und 211350); SAM fand alle mit dem
    passenden Wort (Score 0,75–0,98)
  - Wiederholgenauigkeit bei unbewegtem Objekt: SAM < 1 mm (Teller 4×, Zylinder 3×), Jev ≈ 10 mm
  - Abweichung SAM–Jev bei gefundenen Objekten 2,5–14 mm
  - **Wortwahl entscheidet:** „blue cylinder“, „red plate“, „red cube“ → 0 Treffer; „blue cap“/„blue object“,
    „orange plate“, „red object“ → Treffer. Teller und Würfel sind orange; SAM nimmt Farben wörtlich
  - Falsch positiv: Teller unter dem Arm verdeckt → „orange plate“ 0,69 auf den orangen Greiferteilen.
    Schwelle ≥ 0,8 und vor `locate` parken
- [x] Tool `sam.locate` (`graph/cell.py`): Prompts der Reihe nach, erster Treffer ≥ 0,8 gewinnt; Ausgaben
  `target`, `prompt`, `confidence`, `box`, `count`, `tried`; Prüfung „ein Begriff pro Prompt“. Live-GUI zeigt es
- [x] `--command ask --sam "a|b"` für den Look-Schritt; `--command run --image <jev.jpg>` = Dry-Run mit echtem SAM
- [x] `cylinder_to_plate` (v3) lokalisiert mit `sam.locate`; Dry-Run mit echtem SAM auf dem Foto vom 13.11. grün
- [x] Live-Lauf mit `sam.locate` an der Zelle: beide Objekte im ersten Versuch, Bild vs. Tiefe 1,5 / 4,1 mm
- [x] Tiefe an der Jev-Kamera: `jev_camera(depth=True)`, auf Farbe ausgerichtet (rio-hw unverändert); läuft
  über USB 2 mit 6 fps (99,8 % gültige Pixel). Live-Läufe streamen sie (`--no-jev-depth` schaltet ab) und
  speichern `<node>.depth.npz` (mm) neben dem Foto
- [x] Gegenprobe (`graph/depth_check.py`, 16.11., ohne Arm): Tischebene aus der Tiefe 1,7° geneigt, 3,4 mm rms,
  und ≈ 18 mm tiefer als aus der Kalibrierung erwartet (Würfelmitte 11,4 mm − halbe Kante ≈ −7 mm; Tiefe
  −25 mm). **Absolute Höhen aus der Tiefe nicht verwenden**; relative Höhen (Objekt über lokalem Tisch)
  und x/y stimmen: Zylinder Tiefe vs. Bild 1,3 mm, Höhe 49 mm
- [x] `graph/depth.py`: SAM-Box + Tiefe → Punkte über dem lokalen Tisch → Mitte der Oberseite, Höhe,
  Grundfläche, **Gierwinkel** (rund → keiner, Quadrat → Periode 90°). Getestet auf ray-gecasteter Szene
- [x] `sam.locate` misst mit Tiefe mit (`measured`: Höhe, Gierwinkel, TCP-Ziel, Abstand zum Bildziel);
  `use_depth: true` nimmt das Ziel aus der Tiefe (Standard aus, bis an der Zelle geprüft).
  `run --image <jev.jpg>` nutzt eine daneben liegende `jev.depth.npz`
- [ ] Ursache der 18 mm / 1,7° klären: Tiefen-Offset der D435i oder Neigung der Hand-Auge-Pose
  (z. B. ArUco-Marker flach auf den Tisch legen und Tiefe gegen Markerecken vergleichen)
- [ ] `z_offset_m` für `place_at` aus der gemessenen Höhe des Ziels (Teller 24 mm) statt von Hand
- [ ] SAM-Maske dekodieren (für Objekte in der Kiste): eigener Dekoder ist aufwendig (Range-Coder);
  Alternative Paket `meta-sam-parser` (reines Python, SAM License) — Entscheidung über die Abhängigkeit
- [x] `pick_at align_yaw` (Code): Gierwinkel aus der Tiefe am `Target`; Handgelenk dreht höchstens ±45°
  (Quadrat, kürzester Weg); TCP wird verschoben, weil das Werkzeug schräg steht (Objektmitte bleibt);
  IK offline an 4 Tischpunkten für −45…+45° erreichbar. Längliche Objekte abgelehnt (Schließrichtung fehlt)
- [x] Gierwinkel aus der Tiefe am echten Würfel unbrauchbar (16.11.): Kanten eines 4-cm-Würfels in der D435i-Tiefe
  verschmiert, Grundfläche 74 × 47 mm statt ≈ 40 mm, Winkel springt. Jetzt aus dem **Farbbild**: helle Oberseite
  (Otsu auf Helligkeit), Rechteck, Ecken auf die Höhe der Oberseite projiziert; Füllgrad trennt rund/eckig
- [x] Schritt 1: Gierwinkel am echten Würfel: 5,2° → 31,1° → 74,1°; Kantenwinkel im Bild bestätigen die Schritte auf 2–3°
- [x] Schritt 2: Bezugswinkel der Backen **31,1°** (`jaw_yaw.json`). Lehre: Die Annahme „der Würfel richtet sich
  beim Schließen aus“ stimmt nicht; ein Griff über die Diagonale ist auch stabil. Erst gemessen 76,1° (Diagonale),
  der Bediener sah beide Testgriffe an den Kanten → −45°. Besser künftig: Bediener bestätigt flache Anlage,
  oder die Backen direkt von oben im Jev-Bild messen
- [x] Live: `align_yaw` drehte das Handgelenk −22,3° (IK ok, TCP-Versatz 9 mm wie berechnet)
- [x] Aufgabe „roter Würfel auf roten Teller“ (16.11., `experiments/graph/20261116_144653`): success, Würfel mittig
  auf dem Teller; Handgelenk −1,5° (Würfel stand schon fast passend). Lauf ohne Zwischenfragen auf Wunsch des
  Bedieners (Not-Aus in der Hand), Freigaben automatisch nur für geplante Schritte mit Zielen im Tischbereich
- [ ] Würfel bei ≈ 45° zu den Backen mit `align_yaw` (größte Drehung) und Abstellen mit gedrehtem Handgelenk
  (`place_at` behält die Drehung), um den Bezugswinkel direkt zu bestätigen

## C. Graph-Format nach GaP v3

- [x] Typisierte Ausgänge, Daten über `{"$ref": "knoten.feld"}` (z. B. `locate` → `target_obb` → `pick_at`)
- [x] `conditional_edges` mit `router_field` und Exit-Werten (`found` / `not_found` / `outside_area` …)
  statt nur `ok`/`fail`
- [x] `subgraphs` mit `skill`, `inputs`, `outputs`, `exit.success_values`, `on_error`
- [x] `end`-Knoten mit `status` und `recovery` (Greifer öffnen, Parkpose)
- [x] Statischer Validator: Typen passen, jeder Exit-Wert hat ein Ziel, alles erreichbar, jedes
  Fehler-Ende hat eine Recovery (`rio/graph/validate.py`); `check.py` bleibt, bis alle Graphen migriert sind
- [x] Kern `rio/graph/` (Loader, Validator, Executor, Tool-Registry, Event-Strom) + Kassow-Tools
  (`graph/cell.py`, inkl. `operator.ask`, `jev.ask`, `vla.run`); `run` erkennt `"version": 3`;
  `describe` liefert unter `v3` den Tool-Katalog
- [x] `cylinder_to_plate` migriert (`graph/workflows/cylinder_to_plate.json`), Dry-Run grün
- [x] Live-Lauf des v3-`cylinder_to_plate` an der Zelle (16.11., `experiments/graph/20261116_141537`): success,
  Zylinder steht 22 mm neben der Tellermitte (13.11.: Jev, mehrere Fehlversuche bei `locate`)
- [ ] Restliche Graphen migrieren (`pick_place`, `pick_cube`, `example`, `plans/*`), dann v1 (`walk.py`,
  `check.py`) entfernen
- [x] Kern ohne Kassow-Importe halten (Format, Validator, Walker, Tracing), siehe Phase K
- [ ] Gerades Absenken (`moveL`) statt `moveJ` für `pick_at`/`place_at`

## D. Skill-Katalog im Agent-Skills-Format

- [ ] Trennung wie GaP: **Tools** (was die Zelle kann: `jev.ask`, `sam3.segment_text`, `kassow.move_j`,
  `kassow.move_l`, `gripper.*`, `camera.observe`) und **Skills** (Strategien: wahrnehmen, greifen,
  ablegen, parken, VLA ausführen)
- [ ] Pro Skill eine `SKILL.md`: `description`, `exit_conditions`, `produces_outputs`, `allowed_tools`,
  `hard_rules` (u. a. nur im kalibrierten Bereich, Freigabe vor jeder Bewegung)
- [ ] `describe` liest den Katalog aus diesen Dateien (eine Quelle, kein Abdriften zwischen Doku und Code)
- [ ] VLA-Knoten als Skills: `organize` und `open` (`"instruction": "put the object in space 1"`)
  - [ ] Graph `experiments/graph/plans/organize.json`, `check`, Dry-Run, live mit `--device cuda:1` (GPU frei?)
  - [ ] `s` beendet, `x` bricht ab; erster Start lädt den Resident
  - [ ] Danach läuft der Orchestrator unverändert: `uv run -m rio.apps.kassow.orchestrator --dry-run`
- [ ] Offen: `pick`-VLA läuft im Graph nicht (andere Übersichtskamera): neu aufnehmen oder eigene Station?

## E. Tracing und Guards

- [ ] Pro Lauf `workflow.json`, `dag_trace.json`, `node_data/<id>/` (Ein-/Ausgaben, Fotos, Masken),
  aufbauend auf `events.jsonl`; Layout wie GaP, damit `gap viz` es lesen kann
- [ ] Zähler-Grenzen für Wahrnehmungs- und Bewegungsaufrufe (Endlosschleifen brechen ab)
- [ ] Sicherheitsverhalten prüfen: `q` stoppt, Ziel außerhalb des kalibrierten Bereichs wird abgelehnt,
  Alarm führt auf das Fehler-Ende

## F. Checkpoints am echten Roboter

GaP prüft Nachbedingungen nur in der Simulation; wir mit echten Sensoren. Modus `off` / `warn` / `raise`.

- [ ] Gegriffen: Greiferbreite nach dem Schließen > 0 und < offen
- [ ] Abgelegt: erneut wahrnehmen, Objektmaske liegt in der Zielmaske (Teller, Kistenfach)
- [ ] TCP-Abweichung (gibt es schon in `pick_at`) als Checkpoint führen
- [ ] `z_offset_m` von `pick_at` (+2 cm) und `place_at` (+3 cm) schrittweise senken; `cube_to_plate` live

## G. Planer-Skill („GaP-Planer“) und Agent

Eine Datei, auf die der Bediener verweist („mach diese Aufgabe nach dem Planer“), und in der alles
steht, was Claude als Planer/Orchestrator braucht. Ersetzt `.github/agents/robot.agent.md` (Copilot-Format).

- [x] `.claude/skills/gap-planner/SKILL.md` mit `references/` (`workflow_v3`, `vla_or_graph`, `live_runs`,
  `plan_log`); Katalog live aus `describe` (`v3.tools`); `.github/agents/robot.agent.md` verweist nur noch darauf
- [x] Aufgabenordner `experiments/graph/plans/<task>/` mit `plan.jsonl` (Planungsereignisse für GUI und
  Vergleich), Versionen `workflow.v<n>.json`, Dry-Run-Skripte, `report.md`
- [x] Erster Probelauf des Skills an der Zelle, autonom: „Zylinder auf den Teller“ (16.11.), Planung in
  `experiments/graph/plans/cylinder_to_plate_live/` (plan.jsonl, report.md). Bediener schob vorab den Teller ins Bild
- [ ] Ablauf wie GaP: Coordinator (Aufgabe in Subgraphen zerlegen, Skills wählen) → Subgraph-Agent
  (nach `SKILL.md` füllen) → Checkpoint-Agent → Validator → höchstens 2 Fix-Runden
- [x] Modus **autonom**: nur Sensoren der Zelle; der Bediener gibt nur Bewegungen frei
- [x] Modus **kollaborativ**: der Bediener ist ein Sensor. Neuer Knoten `ask_operator` (getypte Frage
  wie bei Jev: `noul`, `choice`, `score`), im Graph sichtbar markiert. Diese Knoten zeigen später, wo
  ein Sensor oder Skill fehlt
- [x] Pro Teilaufgabe VLA oder Graph-Skill wählen bzw. ein VLA empfehlen (Phase L, `vla_or_graph.md`)
- [x] Regeln: kein Code unter `rio/` ändern, nur nach `experiments/graph/plans/` schreiben, `check` und
  Dry-Run vor jedem Live-Lauf, keine erfundenen Knoten/Posen/Grenzen, höchstens zwei Neuplanungen
- [ ] Tests: „Was liegt auf dem Tisch?“ (keine Bewegung) · „Räum die Kiste auf“ (VLA) ·
  „Leg den Würfel auf den Teller“ (locate + pick + place) · unmögliche Aufgabe (meldet, was fehlt)

## H. Self-Learning light (real, ohne Sim)

- [ ] Analyse-Agent liest Trace + Checkpoint-Ergebnisse eines gescheiterten Laufs und schlägt eine
  Graph-Änderung vor (Parameter, Prompt, Schwelle, anderer Skill), als Diff
- [ ] Freigabe durch den Bediener, neuer Lauf; jede Iteration mit Graph-Version und Ergebnis protokolliert

## I. Evaluation wie im Paper

- [ ] Fester Benchmark: z. B. 20 Läufe mit zufälligen Objektpositionen; Erfolgsrate und Zykluszeit
- [ ] Vergleich gegen die VLA allein; ab Phase B jede Verbesserung damit messen

## J. Kisten-Aufgabe: autonom vs. kollaborativ vs. von Hand

- [ ] Von Hand erstellten Graphen der Kisten-Einsortieraufgabe ablegen (Ort/Format klären)
- [ ] Lauf **kollaborativ**: Claude plant, der Bediener beantwortet `ask_operator`-Fragen
- [ ] Lauf **autonom**: dieselbe Aufgabe nur mit den Sensoren der Zelle
- [ ] Beide ausgeführten Graphen extrahieren und vergleichen: Struktur (Subgraphen, Skills, Verzweigungen,
  Recovery), Erfolgsrate, Zeit; `ask_operator`-Knoten als Liste fehlender Fähigkeiten

## L. Wann VLA? Empfehlungs-Skill

Der Planer entscheidet pro Teilaufgabe zwischen Graph-Skills und VLA, und darf auch empfehlen,
für eine Teilaufgabe ein VLA neu zu trainieren. Im Paper fällt π0.5 bei variierender Objektlage auf
≈ 0,20, der Graph bleibt bei 0,93–0,97; VLAs lohnen sich also dort, wo Skripte schwer zu schreiben sind,
nicht für reines Greifen und Ablegen.

- [x] Kriterien als `references/vla_or_graph.md` im Planer-Skill:
  - **Graph-Skills**, wenn das Objekt starr ist, sich gut wahrnehmen lässt (Maske, OBB) und die Bewegung
    sich als Anfahren, Greifen, Ablegen beschreiben lässt
  - **VLA**, wenn Kontakt und Feinbewegung entscheiden (Einsetzen ins Inlay, Nachschieben, Aufräumen in
    der vollen Kiste), Objekte verformbar sind oder sich Geometrie und Lage schlecht modellieren lassen
  - **Hybrid („steered policy“ wie GaP):** der Graph nimmt die Grobbewegung (wahrnehmen, darüber
    fahren), das VLA nur das letzte Stück; kleinere Startvariation macht das VLA robuster
- [ ] Skill `recommending-vla`: gibt es kein passendes VLA, schreibt der Planer eine Empfehlung statt zu
  improvisieren: Teilaufgabe, Instruktion, Startpose, Kameras, grobe Demo-Anzahl, Begründung
- [ ] Auslöser aus den Daten: ein Graph-Skill scheitert wiederholt an derselben Stelle (Phase H) oder ein
  `ask_operator`-Knoten lässt sich mit Sensoren nicht ersetzen (Phase J) → Kandidat für ein VLA
- [ ] Datensammlung: Demos per Teleop am Startpunkt des Hybrid-Knotens; prüfen, ob ein Graph als
  skriptierter Experte selbst Demos erzeugen kann (GaP `collect_and_train`)
- [ ] Im Vergleich (Phase J) markieren, welche Knoten des Handgraphen VLAs sind und ob der Planer dort
  ebenfalls ein VLA wählt; mit dem Benchmark (Phase I) VLA vs. Graph-Skill pro Teilaufgabe messen

## M. Demo-GUI: Graph entsteht und läuft live

Für Vorführungen: eine Seite im Browser, auf der man in Echtzeit sieht, wie der Planer den Graphen
baut und wie die Zelle ihn abläuft, verständlich auch für Zuschauer ohne Robotik-Hintergrund.

- [x] Grundlage: Muster von `orchestrator/board.py` (stdlib-HTTP-Server auf localhost, keine neue
  Abhängigkeit, darf den Arm nie aufhalten); statt Polling Server-Sent Events für flüssige Updates
- [x] Eine Ereignisquelle für alles: Planer und Walker schreiben dieselben Ereignisse (Phase E),
  die GUI liest nur; dieselben Daten lassen sich später als Aufzeichnung abspielen
- [x] **Planungsphase live:** Aufgabe → Zerlegung in Subgraphen (erscheinen nacheinander) → gewählter
  Skill pro Subgraph mit kurzer Begründung (inkl. VLA oder Graph-Skill, Phase L) → Validator-Ergebnis
  (rot/grün, Fix-Runden sichtbar) → Dry-Run-Pfad
- [x] **Ausführung live:** aktiver Knoten hervorgehoben, durchlaufener Pfad eingefärbt, Exit-Wert an
  jeder Kante, Recovery-Pfade sichtbar
- [x] **Was die Zelle sieht:** Jev-Foto mit Box, Konfidenz und Zielpunkt
- [ ] … SAM-Maske, OBB und Gierwinkel (nach Phase B); Wrist/Overview bei VLA-Knoten; TCP-Abweichung
- [x] **Bediener im Bild:** wartende Freigabe groß sichtbar (Leertaste/`q`), `ask_operator`-Fragen und
  Antworten im Modus kollaborativ deutlich anders markiert als Sensor-Knoten
- [x] Gesamtzeit und Laufzeit pro Knoten (Ablauf-Spalte)
- [ ] **Checkpoints und Kennzahlen:** Ergebnis jeder Nachbedingung, Erfolgsrate aus dem Benchmark (Phase I)
- [ ] **Vergleichsansicht:** autonom vs. kollaborativ vs. Handgraph nebeneinander (Phase J);
  Self-Learning-Iterationen als Diff zwischen Graph-Versionen (Phase H)
- [x] Demo-tauglich: Vollbild, großer Text, Englisch als Standard (`?lang=de` Deutsch), läuft ohne Internet
  (keine CDN-Abhängigkeit an der Station)
- [x] Start: `uv run python -m rio.apps.kassow.graph --command view` → http://127.0.0.1:8766
  (`?run=<ordner>`, `?plan=<task>`, `?lang=de`; `--view-host 0.0.0.0` für einen zweiten Bildschirm)
- [x] Export eines Laufs als eigenständige HTML-Datei: `uv run python -m rio.graph.live --root experiments/graph --export lauf.html`
- [x] Erster Einsatz an der Zelle: Planung + Live-Lauf (16.11.)
- [ ] Prüfen, ob GaPs Trace-Browser (`gap viz`, FastAPI + React) für die Nachbetrachtung reicht und
  unsere GUI sich auf die Live-Ansicht beschränken kann

## K. Später: Harness für Applikateure

- [ ] Aufteilung wie GaP (`gap-core` / `gap` / `open-robot-skills`): Kern (Format, Validator, Walker,
  Tracing, Planer-Skill, Demo-GUI) · Zellen-Adapter (Roboter, Kameras, Kalibrierung, Posen) · Skill-Bibliothek
- [ ] Planer-Skill als Claude-Code-Plugin weitergeben; MCP-Server nur, wenn andere Clients ihn brauchen
- [ ] Feinkorrektur mit der Handgelenkkamera vor dem Greifen
