# XR-Arbeitsraum: Anwendung und Aufbau

Die Quest zeigt den Arbeitsraum am echten Roboter an. Sie steuert ihn nicht.
Der bisherige Orchestrator bleibt unveraendert; der XR-Server laeuft separat.

## Starten

Voraussetzungen: eingerichtete rio-Umgebung auf der Station, Quest mit
Entwicklermodus und autorisierter USB-Verbindung sowie `adb` auf diesem Rechner.
Meta-Geraet und Entwicklermodus vorher mit der IT abstimmen.

Aus dem Repo-Root:

```sh
uv run python -m rio.apps.kassow.xr
```

Desktop-Vorschau: http://127.0.0.1:8770 (Maus ziehen zum Drehen, Mausrad zum Zoomen).
Ohne IK-Berechnung starten: `uv run python -m rio.apps.kassow.xr --no-reach`.
Die Reichweitenberechnung braucht die vorhandene Kassow-Kinematik mit Pinocchio;
fehlt sie, wird dieser Layer ausgelassen. Es wird keine Roboterverbindung geoeffnet.

Quest per USB an denselben Rechner anschliessen:

```sh
adb devices
adb reverse tcp:8770 tcp:8770
```

Im Quest-Browser http://localhost:8770 oeffnen. Bei belegtem Port beispielsweise
`--port 8771` verwenden und beide Portnummern bei `adb reverse` anpassen.

## Am Roboter ausrichten (Controller im Greifer)

Einmal einrichten; die Kalibrierung liegt danach in `experiments/xr/calibration.json`
im STAGE-Raum der Quest und wird beim Start von `--headset` automatisch geladen.

1. In der Quest Handtracking ausschalten oder den freien Controller in der Hand halten,
   sonst werden beide Controller inaktiv. Kein Quest-Menue offen lassen: ohne Fokus
   liefert das Headset keine Controller-Daten.
2. Anzeige starten: `uv run python -m rio.apps.kassow.xr --headset --no-reach`.
3. Einen Controller fest in den geschlossenen Greifer geben, den Arm ueber dem Tisch
   parken, sodass das Headset den Controller sieht (oben und seitlich 15 cm frei).
4. Plan pruefen, ohne Bewegung:
   `uv run python -m rio.apps.kassow.xr.calibrate --dry-run --device left`
5. Kalibrieren, Enter vor jedem moveJ, Not-Halt bereit:
   `uv run python -m rio.apps.kassow.xr.calibrate --device left`
   Nur der Arm wird verbunden; der Greifer bleibt zu. Der Lauf wird verworfen, wenn der
   Restfehler ueber 5 mm liegt oder der Controller an der Startpose verrutscht ist.
6. Kontrolle: `uv run python -m rio.apps.kassow.xr.live` zeigt die Fingerspitze live
   (eigene Armverbindung, nicht parallel zu Orchestrator oder Rollout).

`eef_pose` der Kassow ist der Flansch. Die Fingerspitze liegt auf dessen z-Achse,
200 mm (zu) bis 185 mm (offen) entfernt (`rio.xr.plan.GRIPPER_TIP_Z_M`).
Erster Lauf am 16.11.2026: 12 Posen, Restfehler 2,2 mm RMS.

## Greifer mit dem rechten Controller (Teleop und DAgger)

Laeuft die Headset-Anzeige, steuert der rechte Controller den Greifer parallel zu
Maus und Tastatur: **Trigger schliesst, Grip oeffnet.** Die Headset-App gibt die
Tasten unter `GET /buttons` aus; `rio.apps.vive.teleop` und die DAgger-Uebernahme
lesen sie automatisch (`--teleop-xr-url ""` schaltet das ab). Ist der letzte Wert
aelter als 0,25 s, etwa bei offenem Quest-Menue, kommt keine Taste an. Mit
gespeicherter Kalibrierung loesen Trigger und Grip keine Touch-Erfassung mehr aus.

Kommen keine Tasten an (Log: `Controllers: profiles {'left': '', ...} poses none`):
WiVRn hat beim Verbinden keine Controller gesehen, z. B. weil Handtracking aktiv war.
Controller in die Hand nehmen, WiVRn in der Quest neu verbinden und die Anzeige neu
starten; beim Start muss `Right: 'WiVRn right controller'` im Log stehen. Haengt
WiVRn nach einem Neustart in der Quest (Port 9757 nicht offen), den haengenden
`wivrn-server`-Kindprozess beenden oder WiVRn neu starten.

## Monitor im Headset

`uv run python -m rio.apps.kassow.xr --headset --no-reach --screen` zeigt den
X11-Desktop dieses Rechners als Panel (15 fps, mit Mauszeiger), zuerst etwa 1 m vor
dem Kopf. Linker Controller: **Grip halten** greift und verschiebt das Panel,
**Trigger** blendet es aus und ein. Naeher (60-70 cm) ist deutlich schaerfer; WiVRn
streamt komprimiertes Video, daher nie ganz so scharf wie Quest-eigene Fenster.
Mehr Bitrate im WiVRn-Dashboard hilft. Bei Flimmern auf Abstand:
`--screen-cfg.downscale 2`; Groesse: `--screen-cfg.width-m`.

Das Panel ist undurchsichtig: waehrend der Arm faehrt ausblenden oder neben den
Arbeitsraum haengen.

## Was die Layer bedeuten

- **Achsen und Posen:** Basis-Koordinatensystem und gespeicherte TCP-Positionen.
- **Teleop-Grenzen:** konfigurierte Radius- und Hoehengrenzen, keine Erreichbarkeitsgarantie.
- **Tischbereich:** vorhandene Tischkalibrierung auf Greifhoehe; ohne Karte nicht sichtbar.
- **Reichweite:** IK-Stichproben auf mehreren Hoehen fuer eine feste TCP-Orientierung.
  Standardraster: 3 cm; Orientierung aus Tischkarte oder `--pose` (Standard: `home_pick.json`).
  Kein Kollisionscheck, kein vollstaendiger Arbeitsraum und keine sichere Bewegungsbahn.

Waehrend eines VLA-Rollouts (XR-Server vorher starten) kommen dazu:

- **VLA-Linie:** Fingerspitzenbahn des aktuellen Chunks.
- **Spur (tuerkis):** wo die Fingerspitzen tatsaechlich waren, aeltere Abschnitte blasser.
- **Faecher (lila):** fruehere Chunks; zeigt, wie stark die Policy umplant.
- **Kugeln:** orange = Greifer schliesst, gruen = Greifer oeffnet.
- **Geister-Arm:** durchsichtiger Kassow am Ende des neuesten Chunks (IK mit Pinocchio).

Alles wird in einem eigenen Thread mit 10 Hz gesendet und kostet die Regelschleife
keine Zeit. Jede neue Episode leert die Layer. Fuer Videos blendet
`--hide table,poses` Tischebene und Posen-Punkte aus.

Faellt WiVRn aus (Headset aus, Verbindung neu), baut die Anzeige ihre Session nach
3 s selbst neu auf; ebenso, wenn die Controller laenger als 10 s verschwunden sind.
In der Quest nur WiVRn neu verbinden, den Prozess nicht neu starten.

Ergebnisse liegen im Cache unter `experiments/xr/`. Bei geaendertem Kinematikmodell
die passenden `reach_*.json` dort entfernen und neu berechnen lassen.

## Warum dieser Aufbau?

`rio/xr/` enthaelt roboterunabhaengige Layer aus Punkten, Linien und Dreiecken,
Registrierung und WebXR-Anzeige. `rio/apps/kassow/xr/` erzeugt die Kassow-Inhalte.
Alle Daten sind in Metern im Roboter-Basisrahmen; erst die Anzeige transformiert sie.
Andere Prozesse koennen spaeter ueber `XrPublisher` benannte Layer aktualisieren,
etwa VLA-Chunks oder Kraftpfeile. Der Sender arbeitet im Hintergrund und verwirft
ueberholte Updates desselben Layers. VLA-Anbindung und HG-DAgger sind noch nicht integriert.

Der Server lauscht nur auf Loopback. USB leitet ihn zur Quest weiter; kein offener
LAN-Port, kein Cloud-Dienst und keine extern geladenen Web-Bibliotheken.
Layer-Publishing benoetigt ein lokales Token. Die Anzeige ist keine Sicherheitsfunktion.

## Pruefstand und eigener Branch

Desktop-Vorschau und Python-Tests wurden geprueft; echte Quest, Registrierung und
Kassow-IK-Berechnung muessen auf der Station noch validiert werden.
Vor einem Commit `make check`, vor spaeterer Integration auch `make test` ausfuehren.
Die XR-Aenderungen sind noch nicht committet. Fuer einen eigenen Branch zuerst
`git switch -c feat/quest-workspace` ausfuehren, dann Aenderungen pruefen, committen
und mit `git push -u origin feat/quest-workspace` veroeffentlichen.