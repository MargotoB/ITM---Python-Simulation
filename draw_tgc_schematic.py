# -*- coding: utf-8 -*-
"""Част от схемата на TI TIDA-01427: ЦАП + I/V -> MFB LPF (THS4130) -> атенюатор -> VCNTL.
Несиметричен еквивалент на половината от диференциалния тракт."""
import schemdraw
import schemdraw.elements as elm

schemdraw.config(fontsize=9, font="DejaVu Sans")
with schemdraw.Drawing(file="fig_tgc_schematic.png", dpi=200, show=False) as d:
    d.config(unit=2.2)
    dac = d.add(elm.Ic(pins=[elm.IcPin(name="", anchorname="OUT", side="right")],
                        size=(3.4, 1.4), label="DAC8802 (14 bit)\n+ OPA2209 (I/V)\nV = −VFS·D/16384"))
    d.add(elm.Line().right().at(dac.OUT).length(0.4))
    d.add(elm.Resistor().right().label("R1 1 kΩ"))
    nA = d.here
    d.add(elm.Dot())
    d.add(elm.Capacitor().down().at(nA).label("C1\n3000 pF*", loc="top"))
    d.add(elm.Ground())
    d.add(elm.Resistor().right().at(nA).label("R3 499 Ω"))
    nI = d.here
    d.add(elm.Dot())
    d.add(elm.Line().right().length(0.4))
    op = d.add(elm.Opamp(leads=True).anchor("in1").label("U4\nTHS4130", loc="center", ofst=(-0.35, 0)))
    d.add(elm.Line().left().at(op.in2).length(0.4))
    d.add(elm.Ground().down())
    # C2: от изхода към инвертиращия вход (горе)
    d.add(elm.Line().up().at(nI).length(1.2))
    d.add(elm.Capacitor().right().tox(op.out).label("C2 750 pF"))
    d.add(elm.Line().down().toy(op.out))
    # R2: от изхода към възел A (още по-горе)
    d.add(elm.Line().up().at(nA).length(2.4))
    d.add(elm.Resistor().right().tox(op.out.x + 0.6).label("R2 1 kΩ"))
    d.add(elm.Line().down().toy(op.out))
    d.add(elm.Dot())
    d.add(elm.Line().right().at(op.out).length(0.6))
    d.add(elm.Resistor().right().label("Ra1 348 Ω"))
    nv = d.here
    d.add(elm.Dot())
    d.add(elm.Resistor().down().at(nv).label("Ra2\n28 Ω", loc="bottom"))
    d.add(elm.Ground())
    d.add(elm.Line().right().at(nv).length(0.6))
    d.add(elm.Dot(open=True).label("VCNTL\n0…1,5 V\n→ AFE5809", loc="right"))
print("OK")
