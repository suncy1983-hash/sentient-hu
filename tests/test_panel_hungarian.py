"""The sidebar page's Hungarian labels (run through Node when it's installed)."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

PANEL = Path(__file__).parent.parent / "custom_components" / "sentient" / "frontend" / "sentient-panel.js"

HARNESS = """
const fs = require("fs");
let api;
globalThis.__sentientHuTest = (a) => { api = a; };
globalThis.HTMLElement = class {};
globalThis.customElements = { get: () => undefined, define: () => {} };
globalThis.window = globalThis;
eval(fs.readFileSync(process.argv[1], "utf8"));
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map((c) => api.huText(c))));
"""

CASES = {
    "OVERVIEW": "ÁTTEKINTÉS",
    "QUIET HOURS · PEOPLE IN NAPPALI · 3 AI CHECKS TODAY": "CSENDES ÓRÁK · EMBEREK: NAPPALI · MA 3 AI-ELLENŐRZÉS",
    "3 AUTOMATED · 12 LEARNED": "3 AUTOMATIZÁLVA · 12 TANULT",
    "HUSHED · TILL 18:30": "NÉMÍTVA · 18:30-IG",
    "FACT 3": "TÉNY 3",
    "SET · NIGHT": "ÉLESÍTVE · ÉJSZAKA",
    "WARM BY WEEKDAYS 6:30 · WEEKENDS 8:00": "MELEG LEGYEN: HÉTKÖZNAP 6:30 · HÉTVÉGÉN 8:00",
    "OFFLINE · 23 Oct": "NEM ELÉRHETŐ · okt. 23.",
    "FRIDAY 23 OCTOBER": "OKTÓBER 23., PÉNTEK",
    "Quiet & rooms": "Csend és szobák",
    "Forget: A kukát csütörtökön visszük ki?": "Elfelejtsem: A kukát csütörtökön visszük ki?",
    # what people wrote is left alone
    "Kata kávét iszik": "Kata kávét iszik",
    "Hello world": "Hello world",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node isn't installed")
def test_panel_labels_in_hungarian() -> None:
    out = subprocess.run(
        ["node", "-e", HARNESS, str(PANEL)], input=json.dumps(list(CASES)), capture_output=True, text=True, check=True,
    )
    assert dict(zip(CASES, json.loads(out.stdout))) == CASES
