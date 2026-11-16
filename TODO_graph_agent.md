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
- [ ] Beim Start des Arms die Log-Zeile `FK vs measured TCP` prüfen: unter 5 mm
- [ ] `graph/*.json` in `package-data` aufnehmen (nur bei nicht-editierbarer Installation relevant)

## B. Wahrnehmung (SAM 3.1 + Tiefe)

- [ ] SAM-3.1-API-Doku prüfen: Text-Prompt → Instanzmasken + Scores? Latenz?
- [ ] Offline-Vergleich ohne Roboter: SAM 3.1 gegen Jev auf den Fotos der gescheiterten Läufe
  (`experiments/graph/20261113_*/find_*.jpg`); Erkennungsrate und Box-Lage notieren
- [ ] Tiefe an der Jev-Kamera aktivieren (`jev_camera()`: `enable_depth=False`, D435i an USB 2 mit 6 fps);
  prüfen, ob Farbe + Tiefe über USB 2 laufen, sonst USB-3-Port. Tiefe auf Farbe ausrichten
- [ ] Gegenprobe: gemessene Tischebene aus der Tiefe gegen die kalibrierte Ebene
- [ ] `locate` neu: SAM-Maske (Fallback: Box) + Tiefe → OBB mit Mitte, Höhe und **Gierwinkel**
  (Punkte über der Tischebene, Rechteck in der Ebene). Jev bleibt für Fragen und Semantik
- [ ] `pick_at` greift entlang des Gierwinkels (bisher über die Diagonale)

## C. Graph-Format nach GaP v3

- [ ] Typisierte Ausgänge, Daten über `{"$ref": "knoten.feld"}` (z. B. `locate` → `target_obb` → `pick_at`)
- [ ] `conditional_edges` mit `router_field` und Exit-Werten (`found` / `not_found` / `outside_area` …)
  statt nur `ok`/`fail`
- [ ] `subgraphs` mit `skill`, `inputs`, `outputs`, `exit.success_values`, `on_error`
- [ ] `end`-Knoten mit `status` und `recovery` (Greifer öffnen, Parkpose)
- [ ] Statischer Validator: Typen passen, jeder Exit-Wert hat ein Ziel, alles erreichbar, jedes
  Fehler-Ende hat eine Recovery; ersetzt `check.py`
- [ ] Bestehende Graphen migrieren (`cylinder_to_plate`, `pick_place`, `pick_cube`, `example`, `plans/*`)
- [ ] Kern ohne Kassow-Importe halten (Format, Validator, Walker, Tracing), siehe Phase K
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

- [ ] `.claude/skills/gap-planner/SKILL.md` mit `references/` (Schema, Sicherheit, Modi, Beispiele);
  der aktuelle Tool-/Skill-Katalog kommt live aus `describe`, nicht als Kopie in die Datei
- [ ] Ablauf wie GaP: Coordinator (Aufgabe in Subgraphen zerlegen, Skills wählen) → Subgraph-Agent
  (nach `SKILL.md` füllen) → Checkpoint-Agent → Validator → höchstens 2 Fix-Runden
- [ ] Modus **autonom**: nur Sensoren der Zelle; der Bediener gibt nur Bewegungen frei
- [ ] Modus **kollaborativ**: der Bediener ist ein Sensor. Neuer Knoten `ask_operator` (getypte Frage
  wie bei Jev: `noul`, `choice`, `score`), im Graph sichtbar markiert. Diese Knoten zeigen später, wo
  ein Sensor oder Skill fehlt
- [ ] Pro Teilaufgabe VLA oder Graph-Skill wählen bzw. ein VLA empfehlen (Phase L)
- [ ] Regeln: kein Code unter `rio/` ändern, nur nach `experiments/graph/plans/` schreiben, `check` und
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

- [ ] Kriterien als `references/vla_or_graph.md` im Planer-Skill:
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

- [ ] Grundlage: Muster von `orchestrator/board.py` (stdlib-HTTP-Server auf localhost, keine neue
  Abhängigkeit, darf den Arm nie aufhalten); statt Polling Server-Sent Events für flüssige Updates
- [ ] Eine Ereignisquelle für alles: Planer und Walker schreiben dieselben Ereignisse (Phase E),
  die GUI liest nur; dieselben Daten lassen sich später als Aufzeichnung abspielen
- [ ] **Planungsphase live:** Aufgabe → Zerlegung in Subgraphen (erscheinen nacheinander) → gewählter
  Skill pro Subgraph mit kurzer Begründung (inkl. VLA oder Graph-Skill, Phase L) → Validator-Ergebnis
  (rot/grün, Fix-Runden sichtbar) → Dry-Run-Pfad
- [ ] **Ausführung live:** aktiver Knoten hervorgehoben, durchlaufener Pfad eingefärbt, Exit-Wert an
  jeder Kante, Recovery-Pfade sichtbar
- [ ] **Was die Zelle sieht:** Jev-Foto mit SAM-Maske, OBB, Gierwinkel und Konfidenz; Wrist/Overview
  bei VLA-Knoten; Zielpunkt und TCP-Abweichung
- [ ] **Bediener im Bild:** wartende Freigabe groß sichtbar (Leertaste/`q`), `ask_operator`-Fragen und
  Antworten im Modus kollaborativ deutlich anders markiert als Sensor-Knoten
- [ ] **Checkpoints und Kennzahlen:** Ergebnis jeder Nachbedingung, Laufzeit pro Knoten, Gesamtzeit,
  Erfolgsrate aus dem Benchmark (Phase I)
- [ ] **Vergleichsansicht:** autonom vs. kollaborativ vs. Handgraph nebeneinander (Phase J);
  Self-Learning-Iterationen als Diff zwischen Graph-Versionen (Phase H)
- [ ] Demo-tauglich: Vollbild, großer Text, Deutsch/Englisch umschaltbar, läuft ohne Internet
  (keine CDN-Abhängigkeit an der Station)
- [ ] Prüfen, ob GaPs Trace-Browser (`gap viz`, FastAPI + React) für die Nachbetrachtung reicht und
  unsere GUI sich auf die Live-Ansicht beschränken kann

## K. Später: Harness für Applikateure

- [ ] Aufteilung wie GaP (`gap-core` / `gap` / `open-robot-skills`): Kern (Format, Validator, Walker,
  Tracing, Planer-Skill, Demo-GUI) · Zellen-Adapter (Roboter, Kameras, Kalibrierung, Posen) · Skill-Bibliothek
- [ ] Planer-Skill als Claude-Code-Plugin weitergeben; MCP-Server nur, wenn andere Clients ihn brauchen
- [ ] Feinkorrektur mit der Handgelenkkamera vor dem Greifen
