# -*- coding: utf-8 -*-
"""Принципна схема на аналоговия преден край (чертеж със schemdraw)."""
import schemdraw
import schemdraw.elements as elm

schemdraw.config(fontsize=9, font="DejaVu Sans")


def noninv_stage(d, name, rf, rg, lead_in):
    """Неинвертиращ усилвател: Rg от (-) към маса, Rf от (-) към изхода."""
    d.add(elm.Line().right().length(lead_in))
    op = d.add(elm.Opamp(leads=True).anchor("in2").label(name, loc="center", ofst=(-0.35, 0)))
    d.add(elm.Line().left().at(op.in1).length(0.4))
    d.add(elm.Line().up().length(1.1))
    T = d.here
    d.add(elm.Resistor().right().tox(op.out).label(rf))
    d.add(elm.Line().down().toy(op.out))
    d.add(elm.Resistor().left().at(T).length(1.6).label(rg))
    d.add(elm.Ground().down())
    return op


with schemdraw.Drawing(file="fig_schematic.png", dpi=200, show=False) as d:
    d.config(unit=2.2)
    src = d.add(elm.SourceSin().up().label("ехо\n(пиезо)", loc="top"))
    d.add(elm.Ground().at(src.start))
    d.add(elm.Resistor().right().at(src.end).label("Rs 50 Ω"))
    n_in = d.here
    d.add(elm.Dot())
    d.add(elm.Resistor().down().at(n_in).label("Rterm\n50 Ω", loc="bottom"))
    d.add(elm.Ground())
    d.move_from(n_in, 0, 0)
    op1 = noninv_stage(d, "U1\nLNA", "Rf1 1,5 kΩ", "Rg1 100 Ω", 2.6)
    # VCA (TGC)
    d.add(elm.Line().right().at(op1.out).length(0.5))
    vca = d.add(elm.Ic(pins=[elm.IcPin(name="", anchorname="IN", side="left"),
                               elm.IcPin(name="", anchorname="OUT", side="right"),
                               elm.IcPin(name="", anchorname="CTL", side="bottom")],
                        size=(2.0, 1.4), label="VCA (TGC)\n0…−40 dB").anchor("IN"))
    d.add(elm.Line().down().at(vca.CTL).length(0.6))
    d.add(elm.Dot(open=True).label("VCNTL (от DAC)", loc="bottom"))
    d.move_from(vca.OUT, 0, 0)
    op2 = noninv_stage(d, "U2\nPGA", "Rf2 1,5 kΩ", "Rg2 100 Ω", 2.6)
    # Sallen-Key LPF
    d.add(elm.Resistor().right().at(op2.out).label("R1\n806 Ω"))
    nA = d.here
    d.add(elm.Dot())
    d.add(elm.Resistor().right().label("R2\n806 Ω"))
    nB = d.here
    d.add(elm.Dot())
    d.add(elm.Capacitor().down().at(nB).label("C1\n10 pF", loc="bottom"))
    d.add(elm.Ground())
    d.add(elm.Line().right().at(nB).length(0.5))
    op3 = d.add(elm.Opamp(leads=True).anchor("in2").label("U3", loc="center", ofst=(-0.35, 0)))
    d.add(elm.Line().left().at(op3.in1).length(0.4))
    d.add(elm.Line().up().length(1.1))
    top = d.here
    d.add(elm.Line().right().tox(op3.out))
    d.add(elm.Line().down().toy(op3.out))
    d.add(elm.Capacitor().up().at(nA).toy(top.y + 0.9).label("C2 39 pF", loc="top"))
    d.add(elm.Line().right().tox(op3.out.x + 0.6))
    d.add(elm.Line().down().toy(op3.out))
    d.add(elm.Dot())
    d.add(elm.Line().right().at(op3.out).length(0.6))
    d.add(elm.Resistor().right().label("R3\n806 Ω"))
    nout = d.here
    d.add(elm.Dot())
    d.add(elm.Capacitor().down().at(nout).label("C3\n20 pF", loc="bottom"))
    d.add(elm.Ground())
    d.add(elm.Line().right().at(nout).length(0.6))
    d.add(elm.Dot(open=True).label("OUT\n→ АЦП", loc="right"))
print("OK")
