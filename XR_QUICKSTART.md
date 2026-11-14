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

## Am Roboter ausrichten

1. Auf der Seite mindestens drei raeumlich verteilte gespeicherte Posen auswaehlen.
2. Den Arm mit den bestehenden Werkzeugen und unter Aufsicht nacheinander dorthin fahren.
3. In AR den zugehoerigen TCP jeweils mit dem angezeigten Controller-Marker beruehren
   und den Trigger druecken. Grip nimmt den letzten Messpunkt zurueck.
4. Ueberlagerung an einem weiteren, nicht verwendeten Punkt kontrollieren.

Controller-Marker und physische Spitze sind nicht zwingend identisch. Deren Versatz
verfaelscht die Registrierung. Kleiner RMS-Fehler allein beweist keine genaue Ausrichtung.
Die gespeicherte Transformation ist kein dauerhafter Spatial Anchor: nach neuer Sitzung,
Rezentrierung oder Trackingverlust erneut pruefen und gegebenenfalls neu registrieren.

## Was die Layer bedeuten

- **Achsen und Posen:** Basis-Koordinatensystem und gespeicherte TCP-Positionen.
- **Teleop-Grenzen:** konfigurierte Radius- und Hoehengrenzen, keine Erreichbarkeitsgarantie.
- **Tischbereich:** vorhandene Tischkalibrierung auf Greifhoehe; ohne Karte nicht sichtbar.
- **Reichweite:** IK-Stichproben auf mehreren Hoehen fuer eine feste TCP-Orientierung.
  Standardraster: 3 cm; Orientierung aus Tischkarte oder `--pose` (Standard: `home_pick.json`).
  Kein Kollisionscheck, kein vollstaendiger Arbeitsraum und keine sichere Bewegungsbahn.

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