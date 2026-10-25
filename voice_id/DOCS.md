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


## Faces

The app also recognises faces in camera pictures, for Sentient's visitor log: OpenCV's YuNet finds the faces
and SFace turns each into a face print. Like voices, pictures are never stored — only the prints (in
`faces.json`). Teach faces through Sentient (by voice, from a recent visit, or the `sentient.learn_face`
action with a photo from *My media*).

- `face_threshold` (default 0.40): how alike a face must be to count as someone. Raise it if strangers get
  a family name; lower it if the family isn't recognised (a few more pictures in different light help more).
- `face_margin` (default 0.05): how far ahead of the next person the best match must be.

Faces smaller than about 32 pixels (far away) are ignored — too blurry to tell apart.
