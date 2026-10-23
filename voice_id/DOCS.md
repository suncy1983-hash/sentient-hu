# Sentient Voice ID

Felismeri, ki beszél a házban, teljesen helyben. A Sentient integráció hang-STT-je minden megszólalást
elküld ide; az app hang-ujjlenyomatot készít belőle, és összeveti a betanított családtagokéval. Hangot
nem tárol, csak az ujjlenyomatokat (`/data/voices.json`).

**Betanítás:** mondd bármelyik hangasszisztensnek: „Tanuld meg a hangomat, Sanya vagyok”, aztán mondj
még 3–4 mondatot. Gyereknél a szülő mondja: „Tanuld meg Lili hangját”, és utána Lili beszél.

**Beállítások**

- `threshold` (0.7): ennél a hasonlóságnál mondja biztosra, ki beszél.
- `margin` (0.06): legalább ennyivel kell jobbnak lennie a második legjobb hangnál.
- `learn` (be): ha nagyon biztos, megtartja az új ujjlenyomatot, így idővel pontosabb lesz.

Az app nem nyit portot: csak a Home Assistant éri el a belső hálózaton.

---

Recognises who in the house is speaking, locally. Sentient's voice STT sends each utterance here; it is
turned into a voice print and compared with the voices it has been taught. No audio is kept.
