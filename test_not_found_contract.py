"""Jede «kenne ich nicht»-Antwort sagt es auch im Feld (ch-h-bot#497).

Was dieser Wächter verhindert, ist am 10.10.2026 gemessen worden: alle sechs
Bestände, die der Swiss History Bot mit einer Kennung allein fragen kann,
meldeten eine unbekannte Kennung nur als Prosa —

    {"error": "Article '999999999' not found."}

— und eine Prosa-Meldung ist nicht maschinell von einem echten Ausfall zu
unterscheiden. Der Bot las jede davon als Anbieterausfall und antwortete
seinem Leser mit **502 «der Anbieter hat versagt»** statt **404 «es gibt das
nicht»**. Ein Client, der auf 502 wiederholt — und das ist bei einem Ausfall
richtig —, wiederholte ewig eine Kennung, die es nicht gibt.

Seither gilt: der Text bleibt (dieser Server wird auch direkt von Menschen und
Modellen gefragt, und `{}` wäre für sie nichtssagend), und das Feld
`not_found: True` kommt dazu.

Geprüft am Syntaxbaum und nicht am laufenden Server, mit Absicht: so braucht
dieser Satz weder `mcp` noch die Datenbank und läuft überall, wo `python`
läuft. Was er **nicht** prüft, ist, dass der Server das auch ausliefert — das
tun die Sätze gegen eine laufende Instanz, wo es sie gibt.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVER = Path(__file__).with_name("server.py")


def _nutzlasten():
    """Jedes Wörterbuch-Literal im Server, das «not found» sagt."""
    baum = ast.parse(SERVER.read_text(encoding="utf-8"))
    for knoten in ast.walk(baum):
        if not isinstance(knoten, ast.Dict):
            continue
        schluessel = [k.value for k in knoten.keys
                      if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "error" not in schluessel:
            continue
        if "not found" not in ast.unparse(knoten).lower():
            continue
        yield knoten, schluessel


def test_es_gibt_ueberhaupt_solche_antworten():
    """Ohne diesen Satz wäre der folgende wahr, sobald jemand sie umbaut und
    der Wächter nichts mehr findet."""
    assert list(_nutzlasten()), f"keine «not found»-Nutzlast in {SERVER.name}"


def test_jede_sagt_es_auch_im_feld():
    ohne = [f"{SERVER.name}:{k.lineno}  {ast.unparse(k)[:90]}"
            for k, schluessel in _nutzlasten() if "not_found" not in schluessel]

    assert not ohne, (
        "diese Antworten sagen «not found» nur im Text, nicht im Feld — der "
        "Bot kann sie damit nicht von einem Ausfall unterscheiden und gibt "
        "502 statt 404 (ch-h-bot#497):\n  " + "\n  ".join(ohne))


def test_das_feld_ist_wahr_und_kein_text():
    """`"not_found": "yes"` wäre kein Urteil. Der Bot nimmt nur einen echten
    Wahrheitswert als Auskunft und fällt sonst auf das Textmuster zurück."""
    falsch = []
    for knoten, _ in _nutzlasten():
        for schluessel, wert in zip(knoten.keys, knoten.values):
            if (isinstance(schluessel, ast.Constant)
                    and schluessel.value == "not_found"
                    and not (isinstance(wert, ast.Constant)
                             and wert.value is True)):
                falsch.append(f"{SERVER.name}:{knoten.lineno}  "
                              f"not_found={ast.unparse(wert)}")

    assert not falsch, "not_found muss True sein:\n  " + "\n  ".join(falsch)
