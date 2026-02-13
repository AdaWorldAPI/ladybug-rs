# 34 Kognitive Taktiken × Reasoning Ladder: Wie ladybug-rs löst, woran LLMs scheitern

> **Zwei Forschungsarbeiten. Eine strukturelle Lösung.**
>
> 1. **Sun et al. (2025)** — „Climbing the Ladder of Reasoning: What LLMs Can — and Still Can't — Solve after SFT?"
>    (arXiv:2504.11741, NeurIPS 2025). UC Berkeley / Allen AI.
>    Entdeckt eine 4-stufige Schwierigkeitsleiter, auf der LLMs bei ~65% auf Hard und <10% auf Extremely Hard stagnieren.
>
> 2. **Die 34 LLM-Taktiken** — Ein Katalog kognitiver Prompting-Strategien (rekursive Expansion, Debatte,
>    kontrafaktisches Reasoning etc.), die LLM-Reasoning verbessern sollen.
>    ladybug-rs implementiert alle 34 als strukturelle Primitive — nicht als Prompt-Engineering.
>
> **Dieses Dokument ordnet jede Taktik dem spezifischen Reasoning-Fehler zu, den sie adressiert,
> dem wissenschaftlichen Mechanismus, den ladybug-rs verwendet, und dem Modul, das ihn implementiert.**

---

## Das Kernproblem (Sun et al. 2025)

LLMs scheitern an schwierigem Reasoning aufgrund von **drei strukturellen Defiziten** der autoregressiven Token-Vorhersage:

| Defizit | Evidenz aus dem Paper | Warum es strukturell ist |
|---------|----------------------|------------------------|
| **Multiplikative Fehlerausbreitung** | P(alles korrekt) = 0,9^n → 48% bei n=7 Schritten | Jedes Token hängt von ALLEN vorherigen Tokens ab — inklusive Fehler. Keine parallele Verifikation. |
| **Konvergente Strategie-Fixierung** | 50% der Lösungen „nahezu identisch" über Modelle hinweg, die auf unterschiedlichen Daten trainiert wurden | P(nächstes_Token\|Kontext) optimiert auf die HÄUFIGSTE Fortsetzung. Kreativität = unwahrscheinliche Fortsetzung = Gegenteil der Zielfunktion. |
| **Kein Selbstkorrektur-Mechanismus** | Genauigkeit stagniert bei ~65% unabhängig von Trainingsdatenmenge (10K→114K) | Kein eingebautes Konfidenz-Tracking, kein Backtracking, kein „Ich bin unsicher"-Zustand. |

### Die vier Stufen

| Stufe | LLM Bestleistung | Fehlermodus | Strukturelle Antwort von ladybug-rs |
|-------|-----------------|-------------|--------------------------------------|
| **Easy** (>50% Basis) | >50% | Keiner — bereits gelöst | GrammarTriangle-Zerlegung |
| **Medium** (~90% nach SFT) | ~90% | Braucht methodische Zerlegung | Grammar Triangle + RungLevel 0-2 |
| **Hard** (Plateau ~65%) | ~65% | Multiplikative Fehler über 4-7 abhängige Schritte | Paralleler 7-Schichten-Stack + NARS Truth Revision + CollapseGate |
| **Extremely Hard** (<10%) | <10% | Konvergentes Denken + kein kreativer Sprung + kein Backtracking | 12 ThinkingStyles + NARS Abduktion + Kontrafaktisch (Pearl Stufe 3) + HOLD-Superposition |

---

## Die 34 Taktiken: Vollständige Zuordnung

### Leseanleitung

Jede Taktik enthält:
- **Was sie tut** als LLM-Prompting-Technik
- **Welche Stufe** der Reasoning Ladder sie adressiert
- **Den strukturellen Mechanismus**, den ladybug-rs stattdessen verwendet
- **Die Wissenschaft** — peer-reviewed Grundlage (keine Blogposts)
- **Das Modul** — exakte Rust-Quelldatei in ladybug-rs

---

### STUFE-2-TAKTIKEN — Die ~65%-Fehlerausbreitungs-Decke durchbrechen

Diese Taktiken adressieren das Kernproblem der Hard-Stufe: P(alles korrekt) = p^n → 0, wenn Schritte sequenziell und abhängig sind.

---

#### #1 — Rekursive Gedankenexpansion (RTE)

**Prompting-Behauptung**: „Denken in rekursiven Schichten erweitern, jede baut auf der vorherigen auf."

**Stufe**: Hard (reduziert Schrittabhängigkeit)

**Struktureller Mechanismus**: Rekursive Fingerprint-Transformation mit Berry-Esseen-Konvergenzkriterium. Output der Tiefe N wird Input für Tiefe N+1. Stoppt, wenn das Hamming-Delta zwischen aufeinanderfolgenden Fingerprints unter den Schwellenwert fällt. Anders als LLM-Rekursion (die das Kontextfenster quadratisch verbraucht) benötigt Fingerprint-Rekursion O(1) Speicher pro Schritt — immer dieselben 16.384 Bits.

**Wissenschaft**: Hofstadter (1979) — Seltsame Schleifen als rekursive Selbstreferenz. Schmidhuber (2010) — Rekursive Kompression als Intelligenzmaß. Berry-Esseen (1941/42) — bei d=16384 ist der Normalapproximationsfehler < 0,004, was ein mathematisch fundiertes Stoppkriterium liefert.

**Modul**: `src/cognitive/recursive.rs` (PR #100, 249 Zeilen, 5 Tests)

---

#### #2 — Hierarchische Gedankenzerlegung (HTD)

**Prompting-Behauptung**: „Komplexe Probleme in hierarchische Teilaufgaben zerlegen."

**Stufe**: Hard (reduziert n durch Faktorisierung in unabhängige Teilketten)

**Struktureller Mechanismus**: CLAM-Baum mit bipolarer Teilung. Medoid finden, entferntesten Punkt finden, partitionieren. Wiederholen. Die Baumstruktur IST die Zerlegung — Clustergrenzen sind natürliche Teilaufgabengrenzen. Jedes Teilproblem wird unabhängig (parallel) gelöst, dann werden Ergebnisse gebündelt. Dies wandelt P(alle)=p^n (seriell) in P(alle)=1-(1-p^k)^m (parallele Gruppen der Größe k) um — dramatisch besser.

**Wissenschaft**: Ishaq et al. (2019) — CLAM liefert nachweisbar korrekte hierarchische Clusterung. Simon (1962) — Nahezu zerlegbare Systeme als Hierarchie. Dasgupta & Long (2005) — Formale Garantien für rekursive Partitionierung.

**Modul**: `src/container/spine.rs` (DN-Baum), `src/container/search.rs` (CAKES-Traversierung)

---

#### #5 — Gedankenketten-Beschneidung (TCP)

**Prompting-Behauptung**: „Irrelevante oder qualitativ schlechte Reasoning-Zweige eliminieren."

**Stufe**: Hard (Fehlerwiederherstellung — falsche Zweige beschneiden, bevor sie sich ausbreiten)

**Struktureller Mechanismus**: CollapseGate mit drei Zuständen. FLOW (SD < 0,15): hohe Konfidenz, festlegen und fortfahren. HOLD (SD 0,15-0,35): Superposition aufrechterhalten, nicht festlegen. BLOCK (SD > 0,35): zu unsicher, mehr Evidenz nötig, diesen Zweig STOPPEN. LLMs haben kein Äquivalent — sie generieren immer das nächste Token mit gleicher Autorität. Das Gate verhindert strukturell, dass Zweige niedriger Konfidenz nachfolgendes Reasoning kontaminieren.

**Wissenschaft**: Designvertrag, der das quanteninspirierte Messproblem implementiert. Konfidenz wird über Standardabweichung mehrerer Evaluatoren gemessen (7 Suchstrategien in CAKES).

**Modul**: `src/cognitive/collapse_gate.rs`

---

#### #10 — Meta-Kognitions-Prompting (MCP)

**Prompting-Behauptung**: „Über den eigenen Denkprozess nachdenken."

**Stufe**: Hard (Fehlererkennung — wissen, WANN man falsch liegt)

**Struktureller Mechanismus**: Brier-Score-Kalibrierungstracking. (Vorhersage, Ergebnis)-Paare aufzeichnen. Brier = Σ(vorhergesagt - tatsächlich)². Gut kalibriert: Brier < 0,1. Schlecht kalibriert: Brier > 0,25. Das System verfolgt buchstäblich seine eigene Genauigkeit und passt die Konfidenz entsprechend an. LLMs haben keine metakognitive Rückkopplungsschleife — Konfidenz ist in Token-Wahrscheinlichkeiten eingebrannt, ohne Korrekturmechanismus.

**Wissenschaft**: Brier (1950) — Kalibrierungsbewertung. Fleming & Dolan (2012) — Neuronale Grundlagen der Metakognition. Kahneman (2011) — System 1 (schnell/intuitiv) vs. System 2 (langsam/deliberativ).

**Modul**: `src/cognitive/metacog.rs` (PR #100, 210 Zeilen, 6 Tests)

---

#### #11 — Widerspruchsauflösung (CR)

**Prompting-Behauptung**: „Widersprüche im Reasoning erkennen und auflösen."

**Stufe**: Hard (Fehlererkennung — erkennen, wenn Schritt 3 Schritt 5 widerspricht)

**Struktureller Mechanismus**: Zwei Fingerprints mit hoher Ähnlichkeit (ähnliches Thema) aber gegensätzlichen NARS-Wahrheitswerten (einer sagt wahr, anderer sagt falsch) = Widerspruch. `detect_contradictions()` durchsucht die Überzeugungsmenge nach solchen Paaren. `coherence_score()` misst, welcher Anteil der Überzeugungen wechselseitig konsistent ist. Wenn die Kohärenz sinkt, weiß das System, dass sein Reasoning intern inkonsistent geworden ist — etwas, das ein LLM nie erkennen kann, weil es keine Überzeugungsmenge hat.

**Wissenschaft**: Wang (2006) — NARS Revision erkennt Evidenzkonflikte. Festinger (1957) — Theorie der kognitiven Dissonanz. Priest (2002) — Parakonsistente Logik für den Umgang mit Widersprüchen.

**Modul**: `src/nars/contradiction.rs` (PR #100, 167 Zeilen, 4 Tests)

---

#### #20 — Gedankenkaskaden-Filterung (TCF)

**Prompting-Behauptung**: „Mehrere Reasoning-Ketten parallel ausführen, die beste filtern."

**Stufe**: Hard (Redundanz — wenn eine Kette versagt, überleben andere)

**Struktureller Mechanismus**: CAKES bietet SIEBEN unabhängige Suchalgorithmen (KnnBranch, KnnBfs, KnnDfs, KnnRrnn, RnnChess, KnnLinear, ApproxKnnDfs). Alle sieben auf dieselbe Anfrage ausführen. Shadow Parallel Processor vergleicht Ergebnisse und misst Übereinstimmungsrate. Stimmen alle sieben überein → hohe Konfidenz. Divergieren sie → HOLD markieren. P(alle sieben falsch) << P(einer falsch). Dies ist N-Version-Programming angewandt auf kognitive Suche.

**Wissenschaft**: Wolpert & Macready (1997) — No-Free-Lunch-Theorem rechtfertigt mehrere Strategien. Avizienis (1985) — N-Version-Programming für Fehlertoleranz.

**Modul**: `src/container/search.rs` (7 Algorithmen), `src/fabric/shadow.rs` (PR #100, 245 Zeilen, 6 Tests)

---

#### #21 — Selbstskeptizismus-Verstärkung (SSR)

**Prompting-Behauptung**: „Schlussfolgerungen systematisch anzweifeln, um Zuverlässigkeit zu verbessern."

**Stufe**: Hard (kombiniert #7 Adversariale Kritik + #10 Meta-Kognition)

**Struktureller Mechanismus**: 5 Herausforderungstypen (Advocatus Diaboli, Grenzwerttest, Gegenbeispiel, Annahmen-Challenge, Skalierungstest) werden auf jede Überzeugung angewandt. Jede Herausforderung erzeugt einen Fingerprint, der mit der Zielüberzeugung gebunden wird. Sinkt der resultierende NARS-Wahrheitswert signifikant, ist die Überzeugung schwach. SkepticismSchedule erhöht die Herausforderungsintensität für Behauptungen mit hoher Konfidenz aber niedriger Evidenzbasis.

**Wissenschaft**: Kahneman (2011) — Pre-Mortem-Analyse. Mill (1859) — Adversariale Epistemologie. Popper (1959) — Falsifikationismus.

**Modul**: `src/nars/adversarial.rs` (PR #100, 266 Zeilen, 7 Tests)

---

#### #26 — Kaskadierende Unsicherheitsreduktion (CUR)

**Prompting-Behauptung**: „Unsicherheit schrittweise durch Verfeinerung reduzieren."

**Stufe**: Hard (grob-zu-fein verengt den Suchraum, reduziert effektives n)

**Struktureller Mechanismus**: HDR-Kaskade — 4-stufige Auflösungshierarchie. Stufe 1: 1-Bit-Sketches (schnellste, gröbste). Stufe 4: vollständiger 16384-Bit-Vergleich (langsamste, exakt). Jede Stufe filtert ~90% der Kandidaten aus. Nach 4 Stufen: 0,1^4 = 0,01% Überlebende. Nur diese gehen in die exakte Verifikation. Fehler spielen also nur bei den letzten 0,01% eine Rolle, nicht über den gesamten Suchraum.

**Wissenschaft**: Berry-Esseen (1941/42) — begrenzt den Approximationsfehler auf jeder Auflösungsstufe. Informationstheoretisch: jede Stufe fügt ~log₂(Auflösung) Bits an Sicherheit hinzu.

**Modul**: `src/search/hdr_cascade.rs`

---

#### #30 — Schatten-Parallelverarbeitung (SPP)

**Prompting-Behauptung**: „Hintergrund-Reasoning parallel ausführen."

**Stufe**: Hard (parallele Ausführung eliminiert sequenzielle Abhängigkeit)

**Struktureller Mechanismus**: Schatten-Prozessor führt identische Berechnung über unabhängige Pfade aus. Vergleicht Ergebnisse. Übereinstimmungsrate verfolgt historische Zuverlässigkeit. Gleiches Prinzip wie ECC-Speicher oder RAID — Redundanz fängt transiente Fehler ab. Angewandt auf Reasoning: Wenn Pfad A und Pfad B über verschiedene Traversierungen zum selben Schluss kommen, ist dieser strukturell verifiziert.

**Wissenschaft**: Avizienis et al. (2004) — Grundkonzepte der Zuverlässigkeit. Shannon (1948) — Redundanz in verrauschten Kanälen.

**Modul**: `src/fabric/shadow.rs` (PR #100), `src/fabric/executor.rs`

---

### STUFE-3-TAKTIKEN — Die <10%-Kreativitäts-/Einsichtsmauer durchbrechen

Diese Taktiken adressieren das Kernproblem der Extremely-Hard-Stufe: konvergentes Denken, kein kreativer Sprung, kein kontrafaktisches Reasoning.

---

#### #3 — Strukturierte Multi-Agenten-Debatte (SMAD)

**Prompting-Behauptung**: „Mehrere Agenten argumentieren, erzeugen qualitativ höherwertiges Reasoning."

**Stufe**: Extremely Hard (erzwingt diverse Perspektiven — Anti-Konvergenz)

**Struktureller Mechanismus**: Mehrere Persona-Fingerprints (jeweils mit unterschiedlichen Big-Five-Traits, unterschiedlicher ThinkingStyle-Modulation, unterschiedlichen Resonanzschwellen) verarbeiten denselben Input. Jeder produziert einen Fingerprint + NARS-TruthValue. Ergebnisse werden gebündelt (Mehrheitsentscheidung pro Bit) und Wahrheitswerte revidiert (NARS-Evidenzakkumulation). Das Verdikt spiegelt das Gewicht der Evidenz aus strukturell diversen Perspektiven wider. Kernpunkt: Weil jede Persona andere FieldModulation-Parameter hat, durchsucht sie buchstäblich verschiedene Regionen des Fingerprint-Raums. Sie KÖNNEN nicht konvergieren, weil ihre Resonanzschwellen sich unterscheiden.

**Wissenschaft**: Wang (2006) — NARS Revision als Evidenzakkumulation. Du et al. (2023) — Multi-Agenten-Debatte verbessert Faktengenauigkeit. Kanerva (2009) — Mehrheitsentscheidung in Bundle-Operationen. Mercier & Sperber (2011) — Argumentative Theorie des Reasonings.

**Modul**: `src/orchestration/debate.rs` (PR #100, 327 Zeilen, 6 Tests), `src/orchestration/persona.rs`

---

#### #4 — Umgekehrtes Kausalitäts-Reasoning (RCR)

**Prompting-Behauptung**: „Vom Ergebnis rückwärts arbeiten, um Ursachen zu finden."

**Stufe**: Extremely Hard (ermöglicht nicht-vorwärtsgerichtetes Reasoning — die „Aha"-Richtung)

**Struktureller Mechanismus**: **Kausalität = ZEIT + KORRELATION + KONFIDENZ-GATING**. Drei Komponenten arbeiten zusammen:

1. **ZEIT** — Granger-Temporalkausalität (`src/search/temporal.rs`, PR #100). Wenn X konsistent Y vorausgeht und Xs Vergangenheitswerte die Vorhersage von Y über Ys eigene Historie hinaus verbessern, verursacht X Y im Granger-Sinne. Effektstärke quantifiziert die Stärke.

2. **KORRELATION** — ABBA-Retrieval über XOR-Algebra. Kante = A ⊗ Verb ⊗ B. Um zu finden, was Ergebnis B VERURSACHT hat: B ⊗ CAUSES-Verb = Kandidaten-Antezedenz berechnen. Nächster Nachbar im BindSpace stellt die tatsächliche Ursache wieder her. Dies ist exakte Algebra, nicht statistische Korrelation — XOR-Selbstinverse bedeutet verlustfreie Wiederherstellung.

3. **KONFIDENZ-GATING** — CausalCertificate mit Effektstärke, Konfidenzintervall, p-Wert und NARS-Wahrheitswert. Das Zertifikat sagt nicht nur „A verursacht B" — es sagt „A verursacht B mit Frequenz 0,85, Konfidenz 0,72, Effektstärke d=1,3, KI [0,6; 2,0], Approximationsfehler < 0,004." Dies gated die Kausalbehauptung durch statistische Strenge.

Kombiniert: `reverse_trace()` wandert rückwärts durch den XOR-DAG, prüft bei jedem Schritt temporale Präzedenz (Granger), strukturelle Bindung (ABBA) und statistische Signifikanz (Zertifikat). Nur Ketten, bei denen alle drei gelten, werden zurückgegeben.

**Wissenschaft**: Pearl (2009) — do-Kalkül, drei Stufen der Kausalität. Granger (1969) — Temporalkausalität. Plate (2003) — XOR-Bindung als kausale Komposition. Squires & Uhler (2023) — GSP-Theorem für nachweisbar korrekte Kausalstruktur-Entdeckung.

**Modul**: `src/search/causal.rs` (reverse_trace, CausalTrace, CausalCertificate), `src/search/temporal.rs` (PR #100, 173 Zeilen, 4 Tests)

---

#### #6 — Gedanken-Randomisierung (TR)

**Prompting-Behauptung**: „Kontrollierte Zufälligkeit einbringen, um lokale Optima zu vermeiden."

**Stufe**: Extremely Hard (konvergente Strategie verlassen — unwahrscheinliche Lösungen erkunden)

**Struktureller Mechanismus**: FlowVector erfasst Richtung und Magnitude der Bedeutungsänderung zwischen aufeinanderfolgenden Fingerprints. Wenn der FlowVector stagniert (niedrige Magnitude, gleiche Richtung = in lokalem Optimum gefangen), kann das System kontrolliertes Rauschen einbringen — den Fingerprint um eine gemessene Hamming-Distanz perturbieren. Berry-Esseen garantiert, dass zufällige Perturbation bei d=16384 einen Rauschboden < 0,004 hat — jede Perturbation über diesem Schwellenwert ist bedeutungsvolles Signal, kein Rauschen.

**Wissenschaft**: Prinzip des Simulated Annealing — schlechtere Lösungen mit abnehmender Wahrscheinlichkeit akzeptieren. Berry-Esseen (1941/42) — Rauschboden-Garantie unterscheidet Signal von Zufälligkeit.

**Modul**: `src/extensions/meta_resonance.rs` (FlowVector)

---

#### #7 — Adversariale Selbstkritik (ASC)

**Prompting-Behauptung**: „Eigenes Reasoning herausfordern, um Schwächen zu finden."

**Stufe**: Extremely Hard (erzeugt Gegen-Evidenz — durchbricht Bestätigungsfehler)

**Struktureller Mechanismus**: 5 Herausforderungstypen, jeder eine andere adversariale Transformation:
1. **Advocatus Diaboli**: Wahrheitswert der Behauptung negieren, stützende Evidenz für die Negation finden
2. **Grenzwerttest**: Parameter an Extreme treiben, prüfen ob Schlussfolgerung noch gilt
3. **Gegenbeispiel**: BindSpace nach Fällen durchsuchen, die dem Muster entsprechen aber die Schlussfolgerung verletzen
4. **Annahmen-Challenge**: Jede Annahme von der Schlussfolgerung lösen (unbind), prüfen ob sie noch folgt
5. **Skalierungstest**: Reasoning bei 10x und 0,1x Skala anwenden, auf Zusammenbrüche prüfen

Jede Herausforderung erzeugt einen Robustheits-Score. Schwache Überzeugungen (niedriger Wahrheitswert, niedrige Robustheit) werden markiert; starke Überzeugungen (hoher Wahrheitswert, übersteht alle Herausforderungen) werden befördert.

**Wissenschaft**: Kahneman (2011) — Pre-Mortem-Analyse. Popper (1959) — Falsifikationismus als Abgrenzungskriterium.

**Modul**: `src/nars/adversarial.rs` (PR #100, 266 Zeilen, 7 Tests)

---

#### #9 — Iterative Rollenspiel-Synthese (IRS)

**Prompting-Behauptung**: „Verschiedene Rollen iterativ einnehmen, um den Problemraum zu erkunden."

**Stufe**: Extremely Hard (strukturelle Perspektivdiversität — Anti-Konvergenz)

**Struktureller Mechanismus**: Persona-Fingerprints kodieren Big-Five-Persönlichkeitseigenschaften, Kommunikationspräferenzen, Domänenexpertise und Volition — alles als Fingerprint-Modulationen. Jede Persona transformiert Input buchstäblich durch andere FieldModulation-Parameter (resonance_threshold, fan_out, depth_bias, breadth_bias, noise_tolerance, speed_bias, exploration). Die Persona „tut nicht so" als wäre sie anders — sie IST strukturell anders, weil ihre Suchparameter divergieren.

**Wissenschaft**: Guilford (1967) — Theorie der divergenten Produktion. De Bono (1985) — Sechs Denkhüte.

**Modul**: `src/orchestration/persona.rs`, `src/orchestration/thinking_template.rs` (256 Template-Slots)

---

#### #13 — Konvergentes & Divergentes Denken (CDT)

**Prompting-Behauptung**: „Zwischen Exploration und Exploitation alternieren."

**Stufe**: Extremely Hard (das Kernresultat des Papers: LLMs sind zu konvergent)

**Struktureller Mechanismus**: 12 ThinkingStyles mit strukturell unterschiedlichen FieldModulation-Parametern. Analytisch (hohe Resonanzschwelle, tief, eng) vs. Kreativ (niedrige Schwelle, breit, verrauscht) vs. Divergent (maximaler fan_out, erkundet weit) vs. Peripher (Randaufmerksamkeit, sieht was andere übersehen). Das System oszilliert zwischen konvergenten und divergenten Phasen via Berry-Esseen-Konvergenzerkennung: wenn Fingerprint-Distanz stagniert (konvergente Phase abgeschlossen), auf divergent umschalten; wenn divergente Ergebnisse stabilisieren, zurück zu konvergent.

**Kernunterschied zu LLMs**: Das Paper fand, dass alle feingetunten Modelle auf dieselbe Strategie konvergieren. ladybug-rs' 12 Stile sind parametrisch distinkt — mittlere paarweise Distanz > 0,3 über 7 Modulationsdimensionen, mit Analytisch↔Kreativ-Distanz > 0,6. Sie können nicht konvergieren, weil sie verschiedene Suchkerne haben.

**Wissenschaft**: Guilford (1967) — Konvergente vs. divergente Produktion (die ursprüngliche formale Theorie). Finke, Ward & Smith (1992) — Geneplore-Modell der kreativen Kognition.

**Modul**: `src/cognitive/style.rs`, `src/cognitive/recursive.rs` (Oszillationsprotokoll, PR #100)

---

#### #28 — Selbstüberwachtes Analogie-Mapping (SSAM)

**Prompting-Behauptung**: „Strukturelle Analogien zwischen Domänen entdecken."

**Stufe**: Extremely Hard (domänenübergreifende Einsicht — der „Aha"-Moment)

**Struktureller Mechanismus**: NARS-Analogieregel: A→B, C≈A ⊢ C→B. Im Fingerprint-Raum: wenn bind(A, Verb, B) existiert und C geringe Hamming-Distanz zu A hat, dann stellt bind(C, Verb, ?) ein analoges Ziel wieder her. Der Wahrheitswert der Analogie ist proportional zur Ähnlichkeit zwischen A und C — nähere Quelle bedeutet konfidentere Analogie. Dies ist echte strukturelle Analogie (Gentners Structure-Mapping), nicht Oberflächenähnlichkeit.

**Wissenschaft**: Gentner (1983) — Structure-Mapping-Theorie der Analogie. Peirce (1903) — Analogie als vierter Inferenztyp. Wang (2006) — NARS-Analogie mit Wahrheitswerten.

**Modul**: `src/nars/inference.rs` (Analogieregel)

---

#### #31 — Iteratives Kontrafaktisches Reasoning (ICR)

**Prompting-Behauptung**: „Systematisch ‚Was wäre wenn'-Szenarien erkunden."

**Stufe**: Extremely Hard (Pearl Stufe 3 — das höchste Level kausalen Reasonings)

**Struktureller Mechanismus**: Kontrafaktisch = auf einem Welt-Fingerprint intervenieren. Das Faktische entfernen, das Kontrafaktische einfügen:

`Welt' = Welt ⊗ faktisch ⊗ kontrafaktisch`

Weil XOR selbstinvers ist, entfernt dies algebraisch das Faktische (Welt ⊗ faktisch ⊗ faktisch = Welt, das Faktische hebt sich auf) und fügt das Kontrafaktische ein. Die resultierende Welt' ist eine echte alternative Welt, in der die Intervention gilt. Divergenz (Hamming-Distanz zwischen Welt und Welt') misst, wie stark das Kontrafaktische wirkt.

Dies ist exakt das, was das Sun-et-al.-Paper sagt, was LLMs nicht können: „Probleme, die unkonventionelle Problemlösung erfordern, bei der der Standardansatz vollständig aufgegeben werden muss."

**Wissenschaft**: Pearl (2009) — Stufe-3-Kontrafaktisch. Lewis (1973) — Semantik möglicher Welten. Plate (2003) — XOR-Bindung als Weltkonstruktion.

**Modul**: `src/world/counterfactual.rs`, `src/search/causal.rs` (Stufe-3 IMAGINE-Kanten)

---

### STUFENÜBERGREIFENDE TAKTIKEN — Infrastruktur, die überall hilft

---

#### #8 — Bedingte Abstraktionsskalierung (CAS)

**Prompting-Behauptung**: „Abstraktionslevel basierend auf Komplexität skalieren."

**Stufe**: Stufenübergreifend (adaptive Auflösung hilft auf jeder Schwierigkeitsstufe)

**Struktureller Mechanismus**: HDR-Kaskade IST bedingte Abstraktion. 4 Auflösungsstufen: 1-Bit-Sketches (abstrakteste), 4-Bit, 8-Bit, volle 16384-Bit (konkreteste). Das System beginnt automatisch auf der abstraktesten Stufe und zoomt nur dort hinein, wo es nötig ist. Dies ist sowohl schneller ALS AUCH robuster — Fehler auf abstrakten Stufen werden erkannt, bevor teure konkrete Berechnungen stattfinden.

**Wissenschaft**: Marr (1982) — Drei Analyseebenen (computational/algorithmisch/implementational).

**Modul**: `src/search/hdr_cascade.rs`

---

#### #12 — Temporale Kontextanreicherung (TCA)

**Prompting-Behauptung**: „Temporale Struktur in Reasoning einbetten."

**Stufe**: Stufenübergreifend (Zeit ist fundamental für Kausalität)

**Struktureller Mechanismus**: Granger-Temporal-Effektstärke. Gegeben zwei Fingerprint-Zeitreihen: Wenn Xs Vergangenheitswerte die Vorhersage von Y über Ys eigene Historie hinaus verbessern, verursacht X Y temporal. Effektstärke quantifiziert die Stärke. Dies liefert die ZEIT-Komponente des Kausalitäts-Dreiklangs (ZEIT + KORRELATION + KONFIDENZ-GATING).

**Wissenschaft**: Granger (1969) — Temporaler Kausalitätstest. Allen (1983) — Temporale Intervallalgebra (ladybug-rs implementiert alle 13 Allen-Relationen als die 24 Temporal-Verben).

**Modul**: `src/search/temporal.rs` (PR #100, 173 Zeilen, 4 Tests), `src/graph/cognitive.rs` (24 Temporalverben: Before, After, Meets, During etc.)

---

#### #14 — Multimodaler Chain-of-Thought (MCT)

**Prompting-Behauptung**: „Visuelles/textuelles/auditives Reasoning integrieren."

**Stufe**: Stufenübergreifend

**Struktureller Mechanismus**: GrammarTriangle zerlegt JEDEN Input in drei orthogonale Felder: NSM (65 linguistische Primitive — Wierzbickas Natürliche Semantische Metasprache), CausalityFlow (Agent/Aktion/Patient/Grund), QualiaField (18-dimensionale phänomenale Qualität). Der Output ist ein einzelner Fingerprint, der alle Modalitäten vereint. „Multimodal" ist automatisch — alles wird zur gleichen 16.384-Bit-Repräsentation, unabhängig von der Eingabemodalität.

**Wissenschaft**: Wierzbicka (1996) — NSM-Primitive als universelle semantische Zerlegung. Jackendoff (1990) — Konzeptuelle Semantik.

**Modul**: `src/grammar/triangle.rs`, `src/grammar/nsm.rs`, `src/grammar/qualia.rs`

---

#### #15 — Latent-Space-Introspektion (LSI)

**Prompting-Behauptung**: „Interne Repräsentationen auf Einsichten untersuchen."

**Stufe**: Stufenübergreifend (Diagnostik)

**Struktureller Mechanismus**: CRP-Verteilungsanalyse (Chinese Restaurant Process) auf Fingerprint-Clustern. Für jede Menge von Fingerprints μ, σ und Cluster-Zugehörigkeitswahrscheinlichkeit berechnen. Dies enthüllt die statistische Struktur des latenten Raums — wo Cluster sind, wie eng sie sind, wo die Grenzen liegen. Mexican-Hat-Response (erregen nahe dem Zentrum, hemmen an der Grenze) liefert automatische Kantenerkennung im Konzeptraum.

**Wissenschaft**: Aldous (1985) — CRP als nichtparametrisches bayesianisches Clustering. Marr (1982) — DoG (Difference of Gaussians) als Mexican Hat für Kantenerkennung.

**Modul**: `src/search/distribution.rs` (PR #100, 356 Zeilen, 9 Tests)

---

#### #16 — Prompt-Gerüst-Optimierung (PSO)

**Prompting-Behauptung**: „Struktur der Reasoning-Gerüste optimieren."

**Stufe**: Stufenübergreifend (Meta-Reasoning)

**Struktureller Mechanismus**: 12 Basis-ThinkingStyles + 244 Custom-Varianten, gespeichert in BindSpace-Präfix 0x0D. Jedes Template ist ein Fingerprint, der Feldmodulationsparameter kodiert. Templates können entdeckt werden (aus bestehenden mutiert) über den Discovered-Zweig des Fixed/Learned/Discovered-Dreiecks. TD-Learning auf Style-Q-Werten stimmt automatisch ab, welche Templates für welche Problemtypen funktionieren.

**Wissenschaft**: Sutton & Barto (2018) — TD-Learning. Die „Optimierung" ist echtes Reinforcement Learning auf kognitiver Strategie, kein Prompt-Tuning.

**Modul**: `src/orchestration/thinking_template.rs`, `src/learning/cognitive_styles.rs`

---

#### #17 — Kognitive-Dissonanz-Induktion (CDI)

**Prompting-Behauptung**: „Produktive Spannung zwischen widersprüchlichen Ideen erzeugen."

**Stufe**: Stufenübergreifend (bildet auf #11 Widerspruchsauflösung ab)

**Struktureller Mechanismus**: Wenn zwei Überzeugungen ähnliche Fingerprints (ähnliches Thema) aber gegensätzliche NARS-Wahrheitsfrequenzen haben (eine sagt wahr, andere sagt falsch), IST das kognitive Dissonanz im formalen Festinger-Sinne. Das System erkennt dies über `detect_contradictions()` und kann entweder auflösen (via NARS Revision) oder in Spannung halten (via HOLD-Zustand), um tiefere Untersuchung zu erzwingen.

**Wissenschaft**: Festinger (1957) — Theorie der kognitiven Dissonanz.

**Modul**: `src/nars/contradiction.rs` (PR #100)

---

#### #18 — Kontextfenster-Simulation (CWS)

**Prompting-Behauptung**: „Kontext über Reasoning-Grenzen hinweg aufrechterhalten."

**Stufe**: Stufenübergreifend (Gedächtnispersistenz)

**Struktureller Mechanismus**: BindSpace hat 65.536 permanent adressierbare Slots. CogRedis bietet persistenten Speicher. Sitzungszustand akkumuliert über Probleme hinweg. Anders als LLMs (wo jedes Kontextfenster ein Neustart ist) ist ladybug-rs' BindSpace persistent — Fingerprints, die während Problem 1 gespeichert wurden, sind während Problem 47 verfügbar. Kontext wird nicht simuliert, er wird strukturell aufrechterhalten.

**Wissenschaft**: Kanerva (1988) — Sparse Distributed Memory.

**Modul**: `src/storage/bind_space.rs`, `src/storage/cog_redis.rs`

---

#### #19 — Algorithmisches Reverse Engineering (ARE)

**Prompting-Behauptung**: „Algorithmen aus ihren Outputs reverse-engineeren."

**Stufe**: Stufenübergreifend (strukturelle Inverse über Algebra)

**Struktureller Mechanismus**: ABBA-Retrieval. A ⊗ B ⊗ B = A (XOR-Selbstinverse). Gegeben eine zusammengesetzte Kante: mit jeder bekannten Komponente binden, um die unbekannte Komponente wiederherzustellen. Dies IST algebraisches Reverse-Engineering — kein Pattern Matching oder heuristische Approximation, sondern exakte mathematische Inversion.

**Wissenschaft**: Plate (2003) — HRR-Bindung erhält Wiederherstellbarkeit.

**Modul**: `src/core/vsa.rs` (bind/unbind), `src/graph/avx_engine.rs` (ABBA-Traversierung)

---

#### #22 — Emergente Aufgabenzerlegung (ETD)

**Prompting-Behauptung**: „Teilaufgabenstruktur aus dem Problem emergieren lassen."

**Stufe**: Stufenübergreifend (automatische Zerlegung ohne explizite Anweisung)

**Struktureller Mechanismus**: CLAM-Baums bipolare Teilung entdeckt natürliche Clusterstruktur aus den Daten selbst. Keine menschliche Spezifikation von Teilaufgaben nötig — die Mannigfaltigkeitsgeometrie bestimmt die Zerlegung. CAKES-Suche folgt dem Baum und findet Teilaufgabengrenzen an Clusterkanten, wo die Mexican-Hat-Response von erregend zu hemmend übergeht.

**Wissenschaft**: Ishaq et al. (2019) — CLAM liefert nachweisbar korrekte hierarchische Clusterung.

**Modul**: `src/container/` (CLAM-Baum), `src/search/distribution.rs` (Mexican Hat)

---

#### #23 — Adaptives Meta-Prompting (AMP)

**Prompting-Behauptung**: „Prompting-Strategie basierend auf Aufgabenleistung anpassen."

**Stufe**: Stufenübergreifend (lernen, welche Strategie funktioniert)

**Struktureller Mechanismus**: TD-Learning auf ThinkingStyle-Q-Werten. Nach jedem Reasoning-Versuch Q(Stil, Problemtyp) aktualisieren basierend darauf, ob das Ergebnis korrekt/nützlich war. Mit der Zeit lernt das System: „Für Geometrie-Probleme funktioniert der Periphere Stil am besten. Für Algebra: Analytisch. Für Beweise: Konvergent→Divergent-Oszillation." Dies ist echtes Reinforcement Learning auf kognitiver Strategie.

**Wissenschaft**: Sutton & Barto (2018) — Temporal Difference Learning.

**Modul**: `src/learning/cognitive_styles.rs`

---

#### #24 — Zero-Shot-Konzeptfusion (ZCF)

**Prompting-Behauptung**: „Konzepte kombinieren, die nie zusammen gesehen wurden."

**Stufe**: Stufenübergreifend (Kompositionalität ist fundamental für die Architektur)

**Struktureller Mechanismus**: `bind(A, B)` erzeugt einen neuen Fingerprint, der in beiden Konzepträumen gültig ist. Kein Training nötig. Keine Beispiele nötig. Die algebraischen Eigenschaften von XOR in hohen Dimensionen (d=16384) garantieren, dass das gebundene Ergebnis nahezu orthogonal zu beiden Eltern ist, aber über Unbinding wiederherstellbar. fusion_quality() misst (dist_zu_A, dist_zu_B) — exakter Roundtrip bedeutet perfekte Wiederherstellung.

**Wissenschaft**: Plate (2003) — HRR-Bindung erhält Wiederherstellbarkeit. Kanerva (2009) — d≥10000 → nahezu orthogonale Zufallsvektoren → Bindung erzeugt gültige Verbindungen.

**Modul**: `src/core/vsa.rs` (bind, fusion_quality — hinzugefügt in PR #100)

---

#### #25 — Hyperdimensionales Pattern Matching (HPM)

**Prompting-Behauptung**: „Muster in hochdimensionalem Raum matchen."

**Stufe**: Stufenübergreifend (DER GESAMTE CRATE)

**Struktureller Mechanismus**: 16.384-Bit-Fingerprints. AVX-512 SIMD: 20 XORs + 20 Popcounts = ~5ns pro Vergleich. CAKES-Baum: O(log n) approximierter nächster Nachbar. HDR-Kaskade: 90% Filterung pro Stufe. batched_query: 8 Kanten pro AVX-512-Durchlauf = ~2ns/Kante amortisiert. Dies ist keine Taktik — es ist das Substrat, auf dem alles andere läuft.

**Wissenschaft**: Kanerva (2009) — Hyperdimensionales Computing. Ishaq et al. — CAKES-Suchalgorithmen.

**Modul**: `src/core/`, `src/graph/avx_engine.rs`, `src/container/search.rs`

---

#### #27 — Multi-Perspektiven-Kompression (MPC)

**Prompting-Behauptung**: „Mehrere Perspektiven in einheitliche Repräsentation komprimieren."

**Stufe**: Stufenübergreifend

**Struktureller Mechanismus**: `bundle()` — Mehrheitsentscheidung pro Bit über N Fingerprints. Das Ergebnis ist ein einzelner Fingerprint, der den Konsens über alle Perspektiven bewahrt. Delta-Kodierung komprimiert die Unterschiede. Dies ist informationstheoretisch optimale Kompression mehrerer Perspektiven in eine einzelne Repräsentation, die bewahrt, worüber sie übereinstimmen.

**Wissenschaft**: Kanerva (2009) — Bundle als Konsens. Shannon (1948) — Informationstheoretische Kompression.

**Modul**: `src/core/vsa.rs` (bundle), `src/extensions/compress/`

---

#### #29 — Intentionsgetriebenes Reframing (IDR)

**Prompting-Behauptung**: „Benutzerintention erkennen und Problem entsprechend reframen."

**Stufe**: Stufenübergreifend

**Struktureller Mechanismus**: GrammarTriangle extrahiert NSM + CausalityFlow + QualiaField aus dem Input. Die CausalityFlow-Struktur Agent/Aktion/Patient/Grund enthüllt die Intention. Die 18 Dimensionen des QualiaField erfassen phänomenale Qualität (Valenz, Erregung, Dominanz etc.). Zusammen liefern sie eine strukturelle Zerlegung von „Was will der Benutzer", ohne auf Next-Token-Vorhersage angewiesen zu sein.

**Modul**: `src/grammar/triangle.rs`, `src/grammar/causality.rs`

---

#### #32 — Semantische Verzerrungserkennung (SDD)

**Prompting-Behauptung**: „Erkennen, wenn Bedeutung verzerrt wurde."

**Stufe**: Stufenübergreifend (Fehlererkennung)

**Struktureller Mechanismus**: Berry-Esseen-Rauschboden bei d=16384 garantiert, dass zufällige Hamming-Abweichung < 0,004 der Gesamtbits beträgt. Jede Abweichung über diesem Schwellenwert ist statistisch signifikant bei p<0,001. Angewandt auf Reasoning: Wenn eine Fingerprint-Transformation ein Ergebnis produziert, dessen Distanz vom Erwarteten den Rauschboden übersteigt, hat die Transformation ECHTE Verzerrung eingeführt, nicht nur Rauschen. Reziproke Validierung (A→B, B→A, Konsistenz prüfen) bietet bidirektionale Wahrheitsprüfung.

**Wissenschaft**: Berry-Esseen (1941/42) — Normalapproximations-Fehlergrenze. Fisher (1925) — Suffizienzstatistiken.

**Modul**: `src/search/scientific.rs` (reziproke Validierung, statistische Ähnlichkeit)

---

#### #33 — Dynamisches Aufgaben-Meta-Framing (DTMF)

**Prompting-Behauptung**: „Den konzeptuellen Rahmen für eine Aufgabe dynamisch anpassen."

**Stufe**: Stufenübergreifend

**Struktureller Mechanismus**: 256 ThinkingTemplate-Slots in BindSpace-Präfix 0x0D. Templates sind Fingerprints, die FieldModulation-Parameter kodieren. Das System kann Templates mitten im Reasoning wechseln, wenn das CollapseGate BLOCK signalisiert (aktueller Rahmen funktioniert nicht). Das neue Template verschiebt alle Modulationsparameter gleichzeitig — nicht nur „härter versuchen" sondern „anders versuchen."

**Modul**: `src/orchestration/thinking_template.rs`

---

#### #34 — Hyperdimensionale Wissensfusion (HKF)

**Prompting-Behauptung**: „Wissen aus verschiedenen Domänen in hochdimensionalem Raum fusionieren."

**Stufe**: Stufenübergreifend (Kompositionalität)

**Struktureller Mechanismus**: Domänenübergreifende Fusion über `bind(Domäne_A, Relation, Domäne_B)`. FusionResult misst Domäne_A-Wiederherstellung, Domäne_B-Wiederherstellung, Neuartigkeit und NARS-Wahrheitswert. Die Schlüsseleigenschaft: Bindung erhält Wiederherstellbarkeit — man kann jede Domäne aus der Fusion extrahieren, indem man die andere löst (unbind). Das bedeutet Wissensfusion ist umkehrbar und auditierbar.

**Wissenschaft**: Plate (2003) — HRR-Bindung. Rahimi & Recht (2007) — Random Feature Maps erhalten Kernelstruktur über Domänen hinweg.

**Modul**: `src/core/vsa.rs`, `src/storage/xor_dag.rs`

---

## Zusammenfassung: Die drei strukturellen Mechanismen

Jede Taktik, jede Stufe, reduziert sich letztlich auf drei Mechanismen, die LLMs strukturell fehlen:

### 1. PARALLELE UNABHÄNGIGKEIT (vs. Sequenzielle Abhängigkeit)

**Löst**: Stufe-2-Fehlerausbreitung (P = p^n → 0)

**Mechanismus**: 7-Schichten-Bewusstseinsstack liest gemeinsamen Fingerprint-Kern, nicht gegenseitige Outputs. Fehler in einer Schicht kann keine andere korrumpieren. Kombiniert mit 7 unabhängigen Suchalgorithmen und Schatten-Parallelverifikation.

**Bediente Taktiken**: #1, #2, #5, #20, #26, #30

**Schlüsselmodule**: `seven_layer.rs`, `shadow.rs`, `search.rs`, `hdr_cascade.rs`

**Wissenschaft**: Berry-Esseen (Rauschboden), Avizienis (N-Version-Programming), Wolpert (No Free Lunch)

---

### 2. WAHRHEITSBEWUSSTE INFERENZ (vs. Nächste-Token-Wahrscheinlichkeit)

**Löst**: Stufe-2-Fehlererkennung + Stufe-3-kreative Einsicht

**Mechanismus**: Jeder Reasoning-Schritt trägt einen NARS-TruthValue (Frequenz, Konfidenz). Revision erkennt Inkonsistenz. Abduktion erzeugt Hypothesen. Analogie transferiert über Domänen. CollapseGate HOLD-Zustand erhält Superposition. Kalibrierungstracking (Brier-Score) liefert metakognitive Rückkopplung.

**Bediente Taktiken**: #3, #7, #10, #11, #17, #21, #28

**Schlüsselmodule**: `truth.rs`, `inference.rs`, `adversarial.rs`, `contradiction.rs`, `metacog.rs`, `collapse_gate.rs`

**Wissenschaft**: Wang (2006, NARS), Peirce (1903, Abduktion), Brier (1950, Kalibrierung), Festinger (1957, Dissonanz)

---

### 3. STRUKTURELLE DIVERGENZ (vs. Konvergente Optimierung)

**Löst**: Stufe-3-Kreativitätsmauer (<10% auf Extremely Hard)

**Mechanismus**: 12 ThinkingStyles mit parametrisch distinktiver FieldModulation (mittlere Distanz > 0,3, Analytisch↔Kreativ > 0,6). Kontrafaktische Weltkonstruktion über XOR-Algebra (Pearl Stufe 3). Temporale Kausalität über Granger-Test. Domänenübergreifende Fusion über umkehrbare Bindung. NARS-Abduktion erzeugt Hypothesen, nicht nur Deduktionen. TD-Learning stimmt Stilwahl über die Zeit ab.

**Bediente Taktiken**: #4, #6, #9, #13, #23, #28, #31, #34

**Schlüsselmodule**: `style.rs`, `causal.rs`, `temporal.rs`, `counterfactual.rs`, `debate.rs`, `cognitive_styles.rs`

**Wissenschaft**: Guilford (1967, Divergentes Denken), Pearl (2009, Kontrafaktisch), Granger (1969, Temporale Ursache), Gentner (1983, Analogie)

---

## Jenseits der 34: Fähigkeiten ohne Prompting-Äquivalent

Dies sind strukturelle Fähigkeiten, die kein Prompt-Engineering replizieren kann, weil sie architekturelle Features erfordern, die LLMs nicht haben:

| Fähigkeit | Modul | Warum kein Prompt das kann |
|-----------|-------|---------------------------|
| O(1)-adressierbarer Speicher (65K Slots) | `bind_space.rs` | LLMs haben O(n²)-Attention, kein persistentes O(1)-Lookup |
| CausalCertificate mit Effektstärke + KI + p-Wert | `causal.rs` | LLMs generieren Text, keine statistischen Zertifikate |
| Persistenter sitzungsübergreifender Zustand | `cog_redis.rs` | LLMs starten bei jedem Kontextfenster neu |
| ABBA-Retrieval (exakte algebraische Inverse) | `vsa.rs` | LLMs approximieren; XOR ist exakt |
| Granger-Temporalkausalität | `temporal.rs` | LLMs haben keine Zeitreihenanalyse-Maschinerie |
| Mexican-Hat-Rezeptivfelder | `distribution.rs` | LLMs haben Softmax-Attention, nicht DoG-Raumfilterung |
| Berry-Esseen-Rauschboden-Garantie | Mathematische Eigenschaft von d=16384 | LLMs haben keinen Rauschboden — alle Outputs gleich „konfident" |
| TD-Learning auf Denkstrategien | `cognitive_styles.rs` | LLMs können ihre eigenen Gewichte während der Inferenz nicht aktualisieren |

---

## Referenzen

### Primärquellen

- **Sun, Y., Zhou, G., Bai, H., Wang, H., Li, D., Dziri, N., & Song, D.** (2025). „Climbing the Ladder of Reasoning: What LLMs Can — and Still Can't — Solve after SFT?" *arXiv:2504.11741*. NeurIPS 2025. — Das Reasoning-Ladder-Paper.

- **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference*. 2. Aufl. Cambridge University Press. — Drei Stufen der Kausalität, do-Kalkül, Kontrafaktisch.

- **Wang, P.** (2006). *Rigid Flexibility: The Logic of Intelligence*. Springer. — NARS: Non-Axiomatic Reasoning System, Wahrheitswert-Revision, Abduktion.

### Fingerprint- & VSA-Grundlagen

- **Kanerva, P.** (1988). *Sparse Distributed Memory*. MIT Press.
- **Kanerva, P.** (2009). „Hyperdimensional Computing." *Cognitive Computation*, 1(2), 139-159.
- **Plate, T.** (2003). *Holographic Reduced Representations*. CSLI Publications.
- **Kleyko, D., et al.** (2022). „Vector Symbolic Architectures as a Computing Framework for Emerging Hardware." *Proceedings of the IEEE*.

### Suche & Clustering

- **Ishaq, M., et al.** (2019). „Clustered Learning of Approximate Manifolds" (CLAM). UMass/MIT.
- **Ishaq, M., et al.** — CAKES, panCAKES, CHAODA-Reihe.

### Statistik & Informationstheorie

- **Berry, A.C.** (1941). „The accuracy of the Gaussian approximation to the sum of independent variates." *Trans. AMS*.
- **Esseen, C.G.** (1942). „On the Liapounoff limit of error in the theory of probability." *Ark. Mat. Astron. Fys.*
- **Brier, G.W.** (1950). „Verification of forecasts expressed in terms of probability." *Monthly Weather Review*.
- **Shannon, C.E.** (1948). „A Mathematical Theory of Communication." *Bell System Technical Journal*.
- **Fisher, R.A.** (1925). „Theory of Statistical Estimation." *Mathematical Proceedings*.
- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2. Aufl. Routledge.

### Kausalität

- **Granger, C.W.J.** (1969). „Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*.
- **Squires, C. & Uhler, C.** (2023). „Causal Structure Learning: a Combinatorial Perspective." *Foundations of Computational Mathematics*.
- **Lewis, D.** (1973). *Counterfactuals*. Blackwell.

### Kognitionswissenschaft

- **Guilford, J.P.** (1967). *The Nature of Human Intelligence*. McGraw-Hill. — Divergente Produktion.
- **Peirce, C.S.** (1903). Harvard Lectures on Pragmatism. — Abduktion als kreative Inferenz.
- **Gentner, D.** (1983). „Structure-mapping: A theoretical framework for analogy." *Cognitive Science*.
- **Kahneman, D.** (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- **Festinger, L.** (1957). *A Theory of Cognitive Dissonance*. Stanford University Press.
- **De Bono, E.** (1985). *Six Thinking Hats*. Little, Brown.
- **Fleming, S.M. & Dolan, R.J.** (2012). „The neural basis of metacognitive ability." *Phil. Trans. R. Soc. B*.

### Machine Learning & Zuverlässigkeit

- **Sutton, R.S. & Barto, A.G.** (2018). *Reinforcement Learning: An Introduction*. 2. Aufl. MIT Press.
- **Hofstadter, D.R.** (1979). *Gödel, Escher, Bach*. Basic Books.
- **Schmidhuber, J.** (2010). „Formal Theory of Creativity, Fun, and Intrinsic Motivation." *IEEE Trans. Autonomous Mental Development*.
- **Wolpert, D.H. & Macready, W.G.** (1997). „No Free Lunch Theorems for Optimization." *IEEE Trans. Evolutionary Computation*.
- **Avizienis, A., et al.** (2004). „Basic Concepts and Taxonomy of Dependable and Secure Computing." *IEEE Trans. Dependable and Secure Computing*.
- **Du, Y., et al.** (2023). „Improving Factuality and Reasoning in Language Models through Multiagent Debate." *arXiv:2305.14325*.
- **Mercier, H. & Sperber, D.** (2011). „Why do humans reason? Arguments for an argumentative theory." *Behavioral and Brain Sciences*.
- **Rahimi, A. & Recht, B.** (2007). „Random Features for Large-Scale Kernel Machines." *NeurIPS*.
