# EMPA: Evaluating Persona-Aligned Empathy as a Process

**Source:** arXiv:2603.00552v1 [cs.AI] 28 Feb 2026

**Authors:** Shiya Zhang, Yuhan Zhan, Ruixi Su, Ruihan Sun, Ziyi Song, Zhaohan Chen, Xiaofan Zhang

**PDF:** https://arxiv.org/pdf/2603.00552v1

---

License: arXiv.org perpetual non-exclusive license
arXiv:2603.00552v1[cs.AI] 28 Feb 2026
†]Team Echo, Nature Select
‡]Sun Yat-sen University
*]Corresponding author
EMPA
: Evaluating Persona-Aligned Empathy as a Process
Shiya Zhang
Yuhan Zhan
Ruixi Su
Ruihan Sun
Ziyi Song
Zhaohan Chen
Xiaofan Zhang
[
[
[
zhangshiya@natureselect.ai
zhangshiya1999@gmail.com
(
February 28, 2026
)
Abstract
Evaluating persona-aligned empathy in LLM-based dialogue agents remains challenging. User states are latent, feedback is sparse and difficult to verify in situ, and seemingly supportive turns can still accumulate into trajectories that drift from persona-specific needs. We introduce EMPA, a process-oriented framework that evaluates persona-aligned support as sustained intervention rather than isolated replies. EMPA distills real interactions into controllable, psychologically grounded scenarios, couples them with an open-ended multi-agent sandbox that exposes strategic adaptation and failure modes, and scores trajectories in a latent psychological space by directional alignment, cumulative impact, and stability. The resulting signals and metrics support reproducible comparison and optimization of long-horizon empathic behavior, and they extend to other agent settings shaped by latent dynamics and weak, hard-to-verify feedback.
\correspondence
,
\checkdata
[Project Page](to appear)
1
Introduction
Recent advances in large language models (LLMs) on retrieval, reasoning, and code generation reflect steady gains in computational intelligence (IQ)
[
Zhou2023SOTOPIA
,
deepseekv3_technical_report_2025
,
qwen25_technical_report_2025
,
kavukcuoglu2025gemini25
]
. As their capabilities expand, LLMs are increasingly deployed as agents that plan, decide, and act over multiple turns, rather than as single-turn text generators. This shift calls for a corresponding change in evaluation. When outcomes are shaped by sequences of decisions instead of isolated responses, evaluation must move beyond single-turn scoring toward agent-level assessment
[
Mohammadi2025LLMAgentSurvey
,
arcadinho2024automated
,
ma2024agentboard
,
lin2025seagent
]
. In such settings, performance depends on sustained progress, coherence under evolving context, and adaptation to noisy or delayed feedback—properties that cannot be captured without trajectory-level evaluation.
However, current evaluation practices remain misaligned with long-horizon social applications. In domains such as psychological support, strong benchmark performance does not reliably translate into improved user experience over sustained interaction
[
Zhang2025SAGE
]
. While these applications require continuous control of strategy, pacing, and intervention timing, most benchmarks still reduce evaluation to locally scorable, turn-level outputs
[
rashkin2019empathetic
]
, obscuring long-term behavioral effects
[
Liu2024AgentBench
,
Zhou2025SocialEval
]
.
This limitation is particularly evident in empathy-oriented dialogue. Unlike tool use or information seeking, psychological support does not act on externally observable environment states, making its impact difficult to assess from single-turn outputs or immediate feedback
[
Coutinho2014NeurosciencesEmpathyCounseling
,
Decety2006HumanEmpathySocialNeuroscience
]
. Empathy is therefore better understood as a long-horizon agent interaction driven by latent user states, where locally appropriate responses may fail to produce stable and coherent support over time. This view aligns with psychological accounts of emotional intelligence, which conceptualize empathy as a process unfolding through context, interaction, and feedback rather than a fixed capability
[
Decety2006HumanEmpathySocialNeuroscience
]
. Accordingly, we treat empathy as a latent, trajectory-level behavioral property, expressed through policy adaptation under evolving and partially observable user states.
Empathy-Oriented Interaction as a Latent-State Agent Problem
As such, empathy provides a concrete instance of a broader agent-evaluation challenge: when outcomes depend on temporal dynamics, evolving context, and individual differences, evaluation must move beyond static outputs to characterize behavior at the process level. Psychological support dialogue can be viewed as a long-horizon agent interaction driven by latent user states.
[
arbel2021adaptive
,
zaki2014empathy
,
hoogendoorn2013modelling
,
su2025exploratory
]
This departs fundamentally from conventional agent tasks, where evaluation typically assumes observable states, stable goals, and verifiable success conditions (e.g., tool outputs or final answers)
[
jimenez2024swebench
,
zhou2024webarena
,
Liu2024AgentBench
]
. In psychological support, goals may evolve and effects are often delayed or noisy, leaving no reliable turn-level success signal. Thus, empathy-oriented dialogue exposes a broader limitation of mainstream agent evaluation
[
Zhou2025SocialEval
,
anonymous2026agentsmarathon
,
xu2025theagentcompany
,
lamalfa2025multagentmark
,
anthropic2026demystifying
,
bricken2025alignmentauditingagents
]
. Our work addresses this gap by reframing evaluation as summarized in Figure
1
.
Figure 1
:
Implicit Assumptions in Agent Evaluation and Their Breakdown in Empathy-Oriented Psychological Support
Why Existing Evaluation Paradigms Fail in Empathy-Oriented Interaction
From this perspective, existing evaluation paradigms struggle to capture empathy-oriented interaction. Most approaches rely on isolated, turn-level judgments, effectively reducing empathy to single emotional responses
[
Sabour2024EmoBench
,
Chen2024EmotionQueen
]
. Such decontextualized metrics reward surface affect matching while failing to reflect whether a model consistently tracks and responds to users’ evolving psychological states in real interaction
[
Zhang2025SAGE
,
corinna_2024
]
.
This limitation largely follows from evaluation practices inherited from cognitive tasks, where outputs are assessed against fixed ground truth and treated as independent units
[
Zheng2023JudgingLLMasJudge
]
. Empathy, by contrast, is a process-driven capability whose effectiveness emerges only through behavior unfolding over time
[
Decety2006HumanEmpathySocialNeuroscience
,
Coutinho2014NeurosciencesEmpathyCounseling
,
Zhang2025SAGE
,
Park2023GenerativeAgents
,
Zhou2023SOTOPIA
]
.
These weaknesses are further amplified by dataset construction. To simplify annotation, many empathy benchmarks fragment dialogue into loosely connected utterances
[
Rashkin2018EmpatheticDialogues
,
Poria2019MELD
,
Demszky2020GoEmotions
,
Poria2019MELD
,
Demszky2020GoEmotions
,
Rashkin2018EmpatheticDialogues
,
Sap2019SocialIQA
]
, obscuring emotional reversals and latent motivations that characterize real support interactions. As a result, models are encouraged to generate locally appropriate emotional responses rather than adjust support strategies over an interaction trajectory
[
Sap2019SocialIQA
]
.
Finally, scalar evaluation signals introduce a subtler distortion. By treating empathy as linearly accumulable and rewarding stronger emotional expression
[
Zhang2025SAGE
,
Zheng2023JudgingLLMasJudge
,
Sabour2024EmoBench
,
Chen2024EmotionQueen
]
, existing metrics conflate intensity with effectiveness. Responses that appear empathetic in isolation may nonetheless increase resistance over time when they are directionally misaligned
[
corinna_2024
]
, as reflected in Figure
2
. When evaluation cannot assess such alignment, both model training and comparison are misled
[
Sap2022NeuralTheoryOfMind
,
Shapira2023CleverHansNeuralToM
]
.
Figure 2
:
shows a real EMPA sandbox interaction, revealing a failure of scalar empathy evaluation: high magnitude scores without directional alignment lead to ineffective support, encouraging verbose but misaligned responses.
Our work, EMPA (Empathy Potential Modeling and Assessment) is designed as a latent-state agent evaluation framework for process-driven, trajectorylevel assessment:
1)
Psychologically grounded sandbox scenarios
, which explicitly model users’ latent psychological trajectories and initial resistance, making otherwise unobservable internal states accessible for trajectory-level evaluation;
2)
Non-scripted multi-agent interaction loops
, which avoid scripted turn-level exchanges and expose long-horizon strategies, adaptation, and failure modes in open-ended interaction;
3)
Trajectory-level process metrics
, which go beyond scalar or turn-level scores by jointly capturing directional alignment, cumulative effect, and behavioral stability across turns;
4)
An RL-friendly evaluation interface
, which organizes evaluation outputs as structured signals suitable not only for model comparison but also for downstream optimization.
Formally, EMPA defines a mapping from interaction trajectories to evaluation signals:
ℰ
:
τ
1
:
t
↦
(
s
t
,
r
t
,
d
t
,
info
t
)
\mathcal{E}:\tau_{1:t}\mapsto(s_{t},r_{t},d_{t},\mathrm{info}_{t})
(1)
where
τ
1
:
t
\tau_{1:t}
denotes the dialogue trajectory up to turn
t
t
;
s
t
s_{t}
is a structured psychological state packet (e.g., latent state estimates
P
t
P_{t}
, alignment and progress summaries);
r
t
r_{t}
provides window-level process signals (e.g., directional change
Δ
​
E
t
\Delta E_{t}
, stagnation or regression penalties);
d
t
d_{t}
indicates termination (success, failure, or truncation); and
info
t
\mathrm{info}_{t}
contains diagnostic evidence and rationales.
By organizing evaluation around latent state evolution and process-level signals, EMPA supports reproducible, comparable assessment of long-horizon agent behavior. While instantiated here for empathy-oriented psychological support, the framework generalizes to other agent tasks driven by latent states and delayed, non-verifiable feedback.
2
Related Work
Data Generation: Empathetic Dialogues and Persona Modeling
Most existing empathy datasets are constructed via crowdsourcing and organized around predefined emotion labels or dialogue goals
[
Rashkin2018EmpatheticDialogues
]
. This design supports scalable annotation and has been widely adopted for emotion recognition and affective response modeling. Subsequent work introduced finer-grained emotion taxonomies and richer situational descriptions to increase linguistic and emotional diversity
[
Sap2019SocialIQA
,
Demszky2020GoEmotions
]
. Other studies incorporate personas or background prompts to improve local coherence and stylistic consistency
[
Poria2019MELD
]
. However, most datasets still represent dialogue as isolated turns or loosely connected fragments, without modeling coherent psychological trajectories, life histories, or causal dynamics across interaction. As a result, these resources are well suited for evaluating local emotional sensitivity, but offer limited support for analyzing how empathy evolves and adapts over long-term interaction.
Simulation: Interactive Environments and User Modeling
To move beyond static evaluation, recent work has introduced interactive simulation environments
[
Xie2024OSWorld
]
. Multi-agent systems have become a common paradigm, with Generative Agents demonstrating how LLMs can simulate social behavior over time
[
Park2023GenerativeAgents
]
. Platforms such as SOTOPIA further extend this direction by shifting evaluation from single-turn responses to multi-turn social interaction through role-playing and social objectives
[
Zhou2023SOTOPIA
]
. These frameworks typically rely on user simulators and evaluation modules. Prior studies show that generic LLM-based simulators tend to produce overly cooperative and idealized behaviors
[
Wang2024InDepthUserSim
]
, motivating the development of specialized user models for greater realism and diversity
[
Wang2024InDepthUserSim
]
. Approaches such as UGST further introduce explicit goal tracking to maintain long-horizon dialogue consistency
[
Mehri2025GoalAlignmentUserSim
]
. Overall, existing simulators focus primarily on task success and informational consistency, while comparatively less attention is given to modeling the evolution of users’ psychological states during interaction.
Evaluation: From Static Metrics to Interactive Assessment
Early empathy evaluation relied on static tasks such as emotion classification, sentiment analysis, and social commonsense reasoning
[
Poria2019MELD
,
Demszky2020GoEmotions
,
Sap2019SocialIQA
]
, offering reproducibility but abstracting away interaction dynamics. More recent work distinguishes empathy from emotion recognition and introduces richer tasks to assess contextual adaptation and response generation
[
Sabour2024EmoBench
]
. Related “AI psychometrics” efforts further draw on psychological testing to construct structured benchmarks for cognitive empathy
[
Ye2025LLMPsychometrics
]
.
LLM-as-a-Judge has recently become a dominant paradigm for open-ended evaluation
[
Zheng2023JudgingLLMasJudge
,
Sabour2024EmoBench
]
, with some models approaching or exceeding human references on static benchmarks
[
Sabour2024EmoBench
,
Chen2024EmotionQueen
]
. However, prior studies report high sensitivity to prompts and configurations
[
Zheng2023JudgingLLMasJudge
,
Ye2025LLMPsychometrics
]
, as well as limited capacity to capture behavioral change across turns
[
ullman2023large
]
. Interactive approaches such as Agent-as-a-Judge and state-aware evaluation have begun to address these limitations
[
zhuge2024agentasajudge
]
, with systems like SAGE explicitly tracking interaction state over time
[
Zhang2025SAGE
]
. Despite these advances, most evaluation signals remain scalar or turn-local, providing limited insight into long-horizon trajectories or sustained persona alignment.
Existing approaches study empathy via datasets,simulators,or metrics in isolation. For long-horizon interaction, they fail to jointly model user state dynamics, interaction structure, and trajectory-level outcomes. We introduce a unified agent evaluation paradigm integrating simulation, latent-state modeling, and process-level metrics.
3
Method
We introduces
EMPA
(
E
mpathy
P
otential
M
odeling and
A
ssessment), a process-level framework for evaluating empathy in large language models during multi-turn dialogue. Rather than treating empathy as isolated language output, EMPA conceptualizes it as a dynamic intervention in human-like interaction. The evaluation focus therefore shifts from what the model says to how it behaves over time.
EMPA comprises three components: 1)
A real-to-simulated data pipeline
that distills noisy real-world conversations into controllable, reproducible scenarios; 2)
An multi-agent simulation environment
for long-horizon interaction that exposes strategic choices, adaptation, and failure modes under open-ended interaction; 3)
An Empathy Potential Model (EPM)
that operates on interaction trajectories, modeling empathy as directional, cumulative, and stable state changes in a latent psychological space.
By integrating social simulation with trajectory-level modeling, EMPA provides a unified evaluation perspective: it assesses not only turn-level performance, but whether a model’s behavior stays aligned with user needs over time and produces sustained, substantive impact.
3.1
System Overview
Figure 3
:
Overview of EMPA.
Real affective interaction data are distilled into psychologically consistent user profiles and crisis scenarios. The evaluated model then engages in unscripted, multi-turn interaction with user agents endowed with persona and long-term memory. Empathic behavior is finally quantified from the resulting interaction trajectories using EPM.
EMPA consists of scenario construction and online process evaluation. In the former, a real-to-sim (Real-to-Sim) pipeline distills key psychological signals from real affective interactions to generate psychologically consistent user profiles and crisis scenarios, providing stable and realistic evaluation starting points.
Online evaluation adopts a two-loop multi-agent design to model empathy over interaction trajectories. The outer loop handles natural-language interaction between the evaluated model and a simulated user, producing full dialogue trajectories and exposing strategy choices in open-ended settings. The inner loop evaluates and regulates interaction states, enabling controllable and reproducible process-level assessment. This separation reflects real empathic interaction, where generation, state inference, and regulation are distinct processes; collapsing them into a single model or prompt risks self-consistency bias and obscures long-horizon failures such as strategy drift or ineffective support.
Within this architecture, the Empathy Potential Model (EPM) analyzes complete trajectories along directionality, accumulation, and stability, and supports cross-model comparison through a standardized EPM-Q metric.
3.2
The Data Pipeline: Real-to-Sim Scenario Generation
Process-level empathy evaluation requires scenarios with sufficient psychological depth. Many existing datasets fall short, allowing models to rely on surface cues rather than genuine reasoning or personalization. We therefore propose a real-to-sim (Real-to-Sim) pipeline that converts complex, uncontrolled real interactions into structured simulation scenarios, preserving key psychological signals while ensuring controllability and reproducibility (Algorithm
1
).
1
Input :
Raw dialogue corpus
𝒟
r
​
a
​
w
\mathcal{D}_{raw}
Psychological feature schema
ℱ
\mathcal{F}
LLM-based generator
G
G
Scenario quality criteria
𝒬
\mathcal{Q}
Output :
Scenario set
𝒮
\mathcal{S}
2
3
Initialize scenario buffer
𝒮
←
∅
\mathcal{S}\leftarrow\emptyset
4
foreach dialogue
d
∈
𝒟
r
​
a
​
w
d\in\mathcal{D}_{raw}
do
//
Stage 1: Feature Distillation
5
Extract empathy-relevant segments
6
d
∗
←
Filter
​
(
d
)
d^{*}\leftarrow\mathrm{Filter}(d)
;
7
Extract psychological features
8
f
←
ExtractFeatures
​
(
d
∗
,
ℱ
)
f\leftarrow\mathrm{ExtractFeatures}(d^{*},\mathcal{F})
;
9
if
f
f
is empty
then
10
continue
;
11
//
Stage 2: Re-contextualized Scenario Generation
12
Generate persona card
13
p
←
G
​
(
f
,
‘
​
‘
​
p
​
e
​
r
​
s
​
o
​
n
​
a
′′
)
p\leftarrow G(f,``persona^{\prime\prime})
;
14
Generate crisis event
15
e
←
G
​
(
f
,
‘
​
‘
​
c
​
r
​
i
​
s
​
i
​
s
′′
)
e\leftarrow G(f,``crisis^{\prime\prime})
;
16
Construct scenario
17
s
←
(
p
,
e
)
s\leftarrow(p,e)
;
//
Stage 3: Validation and Refinement
18
if
Validate
​
(
s
,
𝒬
)
\mathrm{Validate}(s,\mathcal{Q})
then
19
𝒮
←
𝒮
∪
{
s
}
\mathcal{S}\leftarrow\mathcal{S}\cup\{s\}
;
20
21
return
𝒮
\mathcal{S}
Algorithm 1
Real-to-Sim Scenario Generation Pipeline
Stage1 Decontextualization: Data Distillation & Feature Extraction
Existing datasets, often centered on explicit emotion labels, tend to reward surface cue matching rather than reasoning about underlying psychological drivers. We therefore adopt a decontextualize-recontextualize pipeline to distill controllable and reproducible empathic structures from real dialogue.
In the decontextualization step, we extract core empathic segments from noisy real-world interactions collected via long-horizon conversations by a professional crowd team. After filtering irrelevant or sensitive content, an LLM-as-a-Judge identifies segments requiring empathic intervention and encodes their key features. Manual validation on N = 200 samples yields  91% accuracy. We further attach coarse memory and experience cues, without sensitive personal information, to preserve psychological continuity.
This process produces structured empathic features with reduced noise but captures only local needs; the next stage embeds them into user roles with stable personas and contexts for long-horizon evaluation.
Stage 2 Re-contextualization: Persona-Anchored Scenario Generalization
Stage 2 organizes distilled empathic features into multi-turn scenarios, shifting evaluation from isolated comfort cues to agents with coherent behavior over time. We build persona-anchored user profiles (Persona Cards) containing stable traits, summarized long-term experiences, and key memories tied to the current crisis. These form the agent’s long-term memory, ensuring consistent behavior across turns. Scenarios are further structured around crisis events and their narrative chain. We explicitly model empathy thresholds and empathy needs to capture both receptiveness and preferred support styles; varying thresholds prevents immediate cooperation, enabling assessment under non-ideal interaction.
Empathy is widely treated as multidimensional, separating cognitive understanding, affective sharing, and motivational mechanisms that enable supportive action
[
Thompson2021CognitiveAffectiveEmpathyEmotionRegulation
,
DecetyYoder2015EmpathyMotivationJustice
]
, as operationalized in instruments such as IRI
[
Davis1983Empathy
,
DeCorte2007IRI_Dutch
]
. Accordingly, we decompose empathy needs into cognitive, affective, and Proactive dimensions to distinguish surface emotion matching from conflict-targeted support.
Finally, we impose cross-temporal psychological constraints, organizing change into a causal arc—cause, development, association—to require temporal integration and coherent empathic strategies over time.
Stage 3 Scenario Refinement & Validation
We apply post-processing and targeted augmentation to ensure coverage and structural consistency. Each script is annotated with primary and multiple secondary scenario labels to support stratified sampling and bias analysis. Before interaction, we estimate an initial empathy deficit to define the user’s psychological baseline, used only as a reference for trajectory analysis. To address the limits of static data, we further introduce a prompt-driven expansion pipeline that allows controlled evolution of the test set via natural-language guidance, enabling sustained dynamic evaluation.
3.3
The Multi-Agent Simulation Environment
EMPA uses a controller-driven multi-agent environment to evaluate empathic strategies over multi-turn interaction, treating dialogue as a dynamic process rather than a fixed script. The system includes four roles: a
User Agent
with stable persona and long-term memory, the
Test Model
, a
Judge Agent
that extracts process-level signals, and a
Director Agent
that regulates interaction via state feedback. Together, they form a generate-evaluate-control loop, enabling open-ended yet reproducible long-horizon empathy evaluation.
For latent-state, non-verifiable interaction problems, scalar or implicit judging tends to collapse heterogeneous criteria and reward the wrong behaviors. EMPA therefore uses rubric-parameterized evaluation to ground the Judge Agent in process-level evidence.
Rubric-Grounded Evaluation for Latent-State Tasks
Without reference answers, evaluation must rely on preference signals rather than objective correctness. The key design choice is whether criteria remain implicit in model weights or are made explicit as inspectable, editable natural-language rubrics
[
Xu2026RubricARM
]
. Scalar scores or pairwise preferences collapse multiple dimensions, such as helpfulness, constraint adherence, tone, and safety, into a single signal. Rubrics instead decompose evaluation into explicit criteria applied consistently across examples
[
Shteynberg2024EmpathicNoEmpathy
]
. The key difference is how judgment is realized. In LLM-as-a-Judge, the judge outputs a final verdict directly, so the signal can inherit the judge’s stylistic biases and over-credit fluency or rhetoric.
[
Zheng2023JudgingLLMasAJudge
,
Hu2024ExplainingLengthBias
]
Rubric-grounded evaluation uses evidence-conditioned scoring: the judge outputs traceable, criterion-level checks with supporting evidence, and a fixed rule aggregates them into a score or process increment. Decoupling evidence from aggregation reduces style leakage, improves robustness to prompt variation, and limits drift. Rubrics should not be treated as fixed templates. Because rubric choice determines whether the judge recovers the intended preference signal, rubrics can be viewed as latent criteria and optimized for preference-recovery accuracy
[
NatMachIntell2024UnderTheSkin
,
JolliffeFarrington2006BES
]
.
This fits social intelligence tasks such as empathy, where success is multi-turn influence on an unobserved user state rather than isolated strong turns. We therefore use rubrics to generate traceable evidence over trajectories for process-level evaluation.
The Central Controller (Director Agent)
The Director Agent serves as the central controller for process scheduling and strategy control. After each evaluation window, it consumes evidence-based state feedback from the Judge Agent (e.g., latent state, progress, mismatch) and executes a standardized Observe–Decide–Act loop to continue, adjust, or terminate interaction. Formally, the Director implements a discrete control policy
π
D
​
(
a
t
∣
s
t
)
\pi_{D}(a_{t}\mid s_{t})
over a fixed action set
𝒜
D
\mathcal{A}_{D}
(e.g., memory release, strategy adjustment, pacing, termination), executed via function calls rather than prompts.
All decisions operate on structured states rather than free-form text, decoupling control from generation. Control is applied as discrete, logged function calls, making decisions traceable and avoiding dialogue collusion and self-evaluation bias. The Director is further limited to a predefined action set, enabling controlled yet open-ended interaction without scripted paths or implicit prompt steering.
1
Input :
Scenario
S
S
, Persona/Actor prompt
A
A
, test model
M
M
, max turns
T
max
T_{\max}
, adjudication interval
K
K
Output :
Trajectory
H
H
, periodic evidence
E
E
, termination type
ζ
\zeta
2
3
Initialize Actor agent
U
U
with
A
A
and long-term memory;
4
Initialize Director agent
D
D
with
S
S
and
A
A
;
5
Initialize Judge agent
J
J
with rubric/checklist;
6
Load initial deficit
P
0
P_{0}
from precomputed IEDR (or request
J
J
to fill IEDR once);
7
Initialize EPM state (
P
←
P
0
P\leftarrow P_{0}
,
E
total
←
0
E_{\text{total}}\leftarrow 0
);
8
H
←
∅
H\leftarrow\emptyset
,
B
←
∅
B\leftarrow\emptyset
,
ζ
←
\zeta\leftarrow
NONE;
9
10
for
t
=
1
,
…
,
T
max
t=1,\ldots,T_{\max}
do
u
t
←
U
.
respond
​
(
H
,
guidance
)
u_{t}\leftarrow U.\mathrm{respond}(H,\mathrm{guidance})
;
//
user simulation under persona + memory
m
t
←
M
.
respond
​
(
H
∪
{
u
t
}
)
m_{t}\leftarrow M.\mathrm{respond}(H\cup\{u_{t}\})
;
//
model under test
11
Append
(
u
t
,
m
t
)
(u_{t},m_{t})
to
H
H
;
12
Append
(
u
t
,
m
t
)
(u_{t},m_{t})
to buffer
B
B
;
13
14
if
t
mod
K
=
0
t\bmod K=0
then
e
t
←
J
.
adjudicate
(
B
,
context
=
A
,
history
=
H
)
e_{t}\leftarrow J.\mathrm{adjudicate}(B,\mathrm{context}{=}A,\mathrm{history}{=}H)
;
//
rubric-grounded evidence (Prog/Neg)
15
(
v
t
,
Δ
​
E
t
,
P
,
E
total
,
summary
)
←
EPM
.
update
​
(
e
t
,
P
,
E
total
)
(v_{t},\Delta E_{t},P,E_{\text{total}},\mathrm{summary})\leftarrow\mathrm{EPM.\,update}(e_{t},P,E_{\text{total}})
;
16
Record
e
t
e_{t}
into
E
E
;
17
Clear
B
B
;
18
19
if
summary
.
success
\mathrm{summary.success}
then
20
ζ
←
\zeta\leftarrow
SUCCESS;
21
break
;
22
23
if
summary
.
failure
​
_
​
detected
\mathrm{summary.failure\_detected}
then
24
ζ
←
\zeta\leftarrow
EPM_FAILURE;
25
break
;
26
27
(
guidance
,
should
​
_
​
continue
)
←
D
.
decide
​
(
H
,
summary
)
(\mathrm{guidance},\mathrm{should\_continue})\leftarrow D.\mathrm{decide}(H,\mathrm{summary})
;
//
observe
→
\rightarrow
think
→
\rightarrow
act
28
if
not
should
​
_
​
continue
\mathrm{should\_continue}
then
29
ζ
←
\zeta\leftarrow
DIRECTOR_STOP;
30
break
;
31
32
33
34
35
if
ζ
=
\zeta=
NONE
then
36
ζ
←
\zeta\leftarrow
MAX_TURNS;
37
38
return
H
,
E
,
ζ
H,E,\zeta
;
Algorithm 2
Central-Controller-Driven Dynamic Execution Cycle
The User Simulator (User Agent)
The User Agent simulates a user with persona-consistent behavior and outcome-dependent reactions across turns, rather than replaying a fixed script. Its behavior is shaped by two constraints: persona injection, where a sampled Persona Card (traits, empathy threshold, need priorities, key experiences) remains fixed throughout interaction (see Appendix
10
); and state-conditioned expression, where emotional intensity and focus are adjusted based on dialogue history and Director inputs. This yields history-dependent, coherent responses, enabling evaluation of strategy adaptation under realistic interactive conditions.
The Evidence Adjudicator (Judge Agent)
The Judge Agent converts natural-language interaction into structured, traceable process signals, linking observable behavior to latent state modeling. Unlike scorers that output a final verdict, the Judge continuously produces interpretable intermediate evidence to support trajectory-level evaluation and runtime control.
Before interaction, the Judge annotates the user’s baseline with an Initial Empathy Deficit Rating (IEDR), represented as a deficit vector over cognitive, affective, and proactive dimensions. This preserves which dimensions are challenging, rather than collapsing the state into a single difficulty level. During interaction, the Judge evaluates recent turns in fixed windows (minimum unit n = 1) using Multi-Dimensional Empathy Progress (MDEP), marking progress or regress on all three dimensions. Each judgment is paired with textual evidence and rationale, making updates attributable to specific model behaviors. The result is a directional increment vector used to update the current latent psychological state.
Crucially, the judge outputs do not constitute the final score; instead, they drive state updates and control decisions. EMPA evaluates cumulative directionality and evidence trends over time, distinguishing isolated hits from sustained alignment and preventing inflated single-turn scores from obscuring long-horizon failures such as strategy drift, repetitive soothing, or ineffective companionship.
Dynamic Execution Cycle
The system runs two nested loops: an outer loop that generates dialogue between the User Agent and the evaluated model, and an inner loop where the Judge and Director evaluate state and control interaction. At fixed intervals, the latent state is updated and the interaction is continued, adjusted, or terminated (Algorithm
2
). Feeding evaluation signals directly into control preserves open-ended dialogue while enforcing state constraints for reproducible long-horizon execution.
3.4
Empathy Potential Model (EPM): A Psychodynamic Vector Formalism
Equating empathy with emotional expression or mimicry is overly simplistic: emotion matching neither ensures mental-state understanding nor sustained support. Systems driven by surface cues often show misalignment or strategy drift in multi-turn interaction
[
Shteynberg2024EmpathicNoEmpathy
,
NatMachIntell2024UnderTheSkin
]
, motivating process-level modeling of psychological mechanisms and their behavioral effects.
Psychology treats empathy as a multi-component construct, separating cognitive and affective empathy and noting that supportive action requires additional motivational mechanisms. This structure is reflected in instruments such as IRI, which distinguish cognitive, affective, and prosocial dimensions and demonstrate their functional non-equivalence
[
Davis1983Empathy
,
JolliffeFarrington2006BES
,
Reniers2009QCAE_EuroPsychiatry
,
Decety2006HumanET
,
Tagesson2025BriefEmpathyInterventions
,
DecetyJackson2004FunctionalArchitectureEmpathy
]
. Following this consensus, we model empathy needs along three related but distinct dimensions, covering understanding, experience, and intentional action.
Figure 4
:
These dimensions provide an operational basis for analyzing LLM behavior in human–AI interaction and user experience.
Accordingly, we introduce the Empathy Potential Model (EPM), a trajectory-level empathy evaluation framework. EPM models empathy as directional interventions on a low-dimensional latent psychological state over long-horizon interaction, where projections toward reduced psychological resistance capture effectiveness. This formulation makes strategy direction, intervention strength, and long-term stability directly computable and comparable.
Psychological State and Empathy Deficit
EPM treats distress as a continuous, multi-dimensional departure from an equilibrium baseline rather than a set of discrete emotion labels. At turn t, the user state is a 3D vector
P
t
∈
ℝ
3
P_{t}\in\mathbb{R}^{3}
, decomposed along orthogonal cognitive, affective, and proactive axes:
P
t
=
C
t
​
𝐞
C
+
A
t
​
𝐞
A
+
P
t
​
𝐞
P
,
𝐞
C
⟂
𝐞
A
⟂
𝐞
P
P_{t}\;=\;C_{t}\,\mathbf{e}_{C}\;+\;A_{t}\,\mathbf{e}_{A}\;+\;P_{t}\,\mathbf{e}_{P},\quad\mathbf{e}_{C}\perp\mathbf{e}_{A}\perp\mathbf{e}_{P}
(2)
Orthogonality decouples mechanisms so change in one dimension does not mask others. The origin
O
=
(
0
,
0
,
0
)
O=(0,0,0)
represents an idealized equilibrium point and is used solely as a geometric baseline. The initial state
P
0
P_{0}
is estimated via the Initial Empathy Deficit Rubric (IEDR), and
‖
P
0
‖
\|P_{0}\|
quantifies baseline resistance. At turn
t
t
,
P
t
P_{t}
denotes the current empathy deficit, and
‖
P
t
‖
\|P_{t}\|
the remaining resistance.
Direction Matters: The Ideal Empathic Direction
Distress magnitude alone cannot evaluate empathic behavior; what matters is whether a response moves in the direction the user actually needs. We define the ideal empathic direction as the unit vector pointing toward psychological balance:
v
t
∗
=
Normalize
⁡
(
−
P
t
)
=
−
P
t
‖
P
t
‖
v_{t}^{*}=\operatorname{Normalize}\!\left(-P_{t}\right)=\frac{-P_{t}}{\|P_{t}\|}
(3)
Normalization removes scale differences across scenarios, focusing evaluation on directional alignment. As a result, a model cannot gain credit by simply amplifying emotional tone or verbosity—only responses aligned with the user’s core needs are counted as effective empathic intervention.
Empathy as Effective Work
At each turn, the model’s response is treated as an instantaneous action on the current psychological state, represented by an action vector
v
→
t
\vec{v}_{t}
. Its components capture net effects along the C/A/P axes (progress Prog minus regress Neg), obtained by a consistent linear mapping from rubric levels.
We define the effective empathic work at turn ttt as the projection of this action into the ideal empathic direction:
Δ
​
E
t
=
v
→
t
⋅
v
t
∗
=
‖
v
→
t
‖
⋅
cos
⁡
(
θ
t
)
\Delta E_{t}=\vec{v}_{t}\cdot v_{t}^{*}=\|\vec{v}_{t}\|\cdot\cos\!\left(\theta_{t}\right)
(4)
where
v
t
∗
v_{t}^{*}
is the unit vector toward psychological balance and
cos
⁡
(
θ
t
)
\cos(\theta_{t})
is the angle between the action and the ideal direction.
The term
cos
⁡
(
θ
t
)
\cos(\theta_{t})
captures directional alignment and is the key discriminator:
•
Aligned
(
cos
⁡
θ
≈
1
\cos\theta\approx 1
): the response targets the core deficit (e.g., emotional support when affect dominates), yielding positive effect.
•
Orthogonal
(
cos
⁡
θ
≈
0
\cos\theta\approx 0
): effort misses the core need (e.g., technical advice amid emotional trauma), yielding near-zero effect.
•
Misaligned
(
cos
⁡
θ
<
0
\cos\theta<0
): the response conflicts with needs (e.g., judgment), increasing resistance.
By modeling empathy as direction-constrained work, EPM distinguishes saying more from doing right, avoiding misjudgment based on emotional intensity or verbosity alone.
Success as Energy-Gated Progress
Scalar metrics often overcredit two cases: chance proximity to the target and low-effort passive listening. We argue that empathic value lies not in brief state improvements, but in sustained, effective opposition to psychological resistance.
Accordingly, EPM gates success by accumulated effective work. Let
E
total
E_{\text{total}}
denote total effective empathic work over an interaction. Success holds iff:
Success
⇔
(
E
t
​
o
​
t
​
a
​
l
>
ϵ
e
​
n
​
e
​
r
​
g
​
y
)
⏟
Necessary Energy Gate
∧
(
(
‖
P
T
‖
<
ϵ
d
​
i
​
s
​
t
)
⏟
Outcome: Resolution
∨
(
cos
⁡
θ
¯
>
τ
a
​
l
​
i
​
g
​
n
)
⏟
Process: Companionship
)
\mathrm{Success}\iff\underbrace{(E_{total}>\epsilon_{energy})}_{\text{Necessary Energy Gate}}\land\left(\underbrace{(\|P_{T}\|<\epsilon_{dist})}_{\text{Outcome: Resolution}}\lor\underbrace{(\overline{\cos\theta}>\tau_{align})}_{\text{Process: Companionship}}\right)
(5)
The energy gate reduces false positives driven by low effort or passive drift: even outcomes near balance are discounted when the effective work is insufficient.
After passing the gate, success follows either
(i)
outcome success, where the state approaches balance, or
(ii)
Process success, where the model maintains strong directional alignment with core needs despite high resistance, is common in deep-trauma or high-inertia settings. By gating on energy, EPM shifts the notion of success from terminal state to interaction dynamics, rewarding directional commitment and persistence rather than superficial state fluctuations.
The Standardized EPM-Q Metric
EPM ultimately labels each dialogue as success or failure based on trajectory-level analysis. This binary outcome, however, cannot capture finer differences in empathic quality or support precise model comparison. We therefore introduce EPM-Q (Empathy Potential Model–Quality Score), a continuous metric that summarizes overall interaction quality beyond the success/failure decision. Unlike fixed-scale measures, EPM-Q is scenario-normalized. For each sample
i
i
, performance is normalized by the initial empathy deficit radius
r
0
,
i
=
‖
P
0
,
i
‖
r_{0,i}=\|P_{0,i}\|
, preventing unfair advantages in easier scenarios and enabling comparability across difficulty levels.
EPM-Q characterizes interaction quality along three complementary axes.
1.
Outcome Quality.
We report task completion
Status
\mathrm{Status}
, final relief via
RDI
\mathrm{RDI}
, total empathic work along the ideal direction
E
total
E_{\text{total}}
, energy surplus beyond the minimum
E
surplus
E_{\text{surplus}}
, and overall criterion-level quality
S
net
S_{\text{net}}
aggregated over C/A/P.
2.
Process Efficiency.
We measure per-turn effective intensity with empathy density
ρ
\rho
, single-turn effectiveness with average effective projection
S
proj
S_{\text{proj}}
, and strategic detours with path tortuosity
τ
\tau
.
3.
Strategic Stability.
We track directional consistency with
cos
⁡
θ
¯
\overline{\cos\theta}
, process smoothness with the positive energy ratio
R
pos
R_{\text{pos}}
, and penalize performative, drifting, or harmful behaviors with
R
pen
R_{\text{pen}}
.
All metrics are scenario-normalized and combined with fixed weights into a single continuous EPM-Q score; details are provided in Appendix
8
and Appendix
9
.
4
Experiments
4.1
Experimental Setup
Goals and hypotheses
EPM measures the effectiveness of supportive behavior under persona constraints. Given persona-specified preferences and situational conditions, it tests whether a model produces support that is consistent with those constraints and accumulates as a coherent process, rather than reflecting surface affect, verbosity, or rhetorical style. We evaluate three properties.
(i) Persona-conditionality. With the response text held fixed, changing the persona constraints should produce predictable shifts in the score.
(ii) Mechanistic attribution. The metric’s discriminative power should come from EPM’s core components and should drop when those components are ablated.
(iii) Robustness to performative empathy. The metric should reward substantive support over surface-level empathic signaling, and reliably penalize sycophantic, templated replies.
Data and controlled perturbations
We use a paired controlled-perturbation design. For each dialogue instance, we construct an original-perturbed pair that preserves the full conversational evidence and changes exactly one factor, enabling attribution of score differences to that intervention. We study two perturbations.
Persona Flip
(paired
n
=
251
n=251
). We keep the dialogue context and model reply unchanged and replace only the persona condition. Pairs are drawn from real test cases and concentrated on replies that are aligned under the original persona (all pairs satisfy
Δ
​
E
>
0
\Delta E>0
; mean
Δ
​
E
=
2.61
\Delta E=2.61
; range
[
0.10
,
5.19
]
[0.10,5.19]
). Importantly, the flip does not alter character identity or narrative traits. Instead, it implements a counterfactual re-parameterization of empathy needs by inverting the priority of selected empathy demands (high to low, with others held constant) and adding consistent preference or anti-preference constraints (e.g., discouraging analytic conclusions and abstract jargon, favoring everyday phrasing). This yields a strict counterfactual condition where the text is identical but the persona-defined objective and constraints differ, testing whether EPM is genuinely persona-conditional rather than surface-driven.
Sycophancy Attacks
(paired
n
=
98
n=98
). We keep the task context fixed and replace the original reply with a sycophantic, performative alternative. Attacks include three typical variants: (i) Pure Empathy, high-affect but content-light platitudes; (ii) Self-Empowerment, generic motivational slogans detached from context; and (iii) Psycho Jargon, terminology-heavy over-interpretation. These are constructed as negative replacements, so we expect one-sided decreases.
Evaluation and analysis
Our primary endpoint is
Δ
​
E
\Delta E
, EPM’s core output capturing process-level supportive effect under persona constraints. We additionally report alignment as an explanatory signal. For each pair i, we compute
d
i
=
metric
perturbed
(
i
)
−
metric
original
(
i
)
d_{i}=\mathrm{metric}^{(i)}_{\text{perturbed}}-\mathrm{metric}^{(i)}_{\text{original}}
(6)
Both perturbations induce directional hypotheses on score change
d
i
d_{i}
. For Persona Flip, responses are initially persona-aligned, so flipping constraints should reduce scores (
d
i
<
0
d_{i}<0
). For Sycophancy Attacks, replacing responses with performative variants likewise implies
d
i
<
0
d_{i}<0
. We report the mean/median of
d
i
d_{i}
, the decrease rate
Pr
⁡
(
d
i
<
0
)
\Pr(d_{i}<0)
, bootstrap 95% CIs, and paired tests (
p
p
values). To address within-case dependence, we repeat inference on case-level aggregates.
4.2
Main Results
We first test persona-conditionality via Persona Flip and robustness to performative responses via Sycophancy Attacks. Results are summarized in Table
5
and visualized in Figure
5
(panels a and c).
Figure 5
:
Sensitivity and robustness under controlled perturbations
Note.
(a) Persona-conditional sensitivity: per-sample trajectories before and after flipping persona constraints (paired n = 251), showing a significant negative shift (
Δ
=
−
1.22
\Delta=-1.22
,
p
< .0001).(b) Ablations: across Persona Flip and Sycophancy, Full EPM exhibits higher sensitivity, measured by mean absolute change, than a rubric-only variant and a direct LLM judge.(c) Adversarial robustness: under sycophancy, EPM penalizes replies that exhibit strong surface-level empathic signaling but fail to provide substantive support. The x-axis is perturbation magnitude (
Δ
​
∥
v
∥
\Delta\lVert v\rVert
) and the y-axis is the EPM score change.
Table 5
:
Perturbation analysis (
Δ
\Delta
= perturbed
−
-
original)
Dataset
n
n
Metric
Mean
Median
Pr
⁡
(
d
i
<
0
)
\Pr(d_{i}<0)
95% CI
p
p
Persona Flip
251
Δ
​
E
\Delta E
−
1.22
-1.22
−
0.17
-0.17
72.50
%
72.50\%
[
−
1.47
,
−
0.97
]
[-1.47,\,-0.97]
<
.0001
<.0001
Persona Flip
251
Δ
​
a
​
l
​
i
​
g
​
n
​
m
​
e
​
n
​
t
\Delta alignment
−
0.38
-0.38
−
0.03
-0.03
69.70
%
69.70\%
[
−
0.47
,
−
0.29
]
[-0.47,\,-0.29]
<
.0001
<.0001
Sycophancy
98
Δ
​
E
\Delta E
−
4.36
-4.36
−
4.68
-4.68
81.60
%
81.60\%
[
−
5.15
,
−
3.57
]
[-5.15,\,-3.57]
<
.0001
<.0001
Sycophancy
98
Δ
​
a
​
l
​
i
​
g
​
n
​
m
​
e
​
n
​
t
\Delta alignment
−
1.08
-1.08
−
1.65
-1.65
84.70
%
84.70\%
[
−
1.27
,
−
0.90
]
[-1.27,\,-0.90]
<
.0001
<.0001
Note.
Persona Flip is constructed from replies aligned under the original persona (
Δ
​
E
>
0
\Delta E>0
in
100.0
%
100.0\%
of pairs), so the test targets whether counterfactually flipped constraints induce systematic deterioration.
Persona Flip: directional response to counterfactual priority inversion
Persona Flip holds the response text fixed and counterfactually swaps persona-defined priorities and constraints. If EPM captures persona-conditioned support, replies aligned with the original persona should become misaligned after the flip, producing a one-sided score drop (
d
i
<
0
d_{i}<0
). A surface-based metric would be largely invariant to this change. On Persona Flip (
n
=
251
n=251
),
Δ
​
E
\Delta E
decreases with mean
−
1.22
-1.22
and median
−
0.17
-0.17
, with a decrease rate of
72.5
%
72.5\%
and a bootstrap
95
%
95\%
confidence interval of
[
−
1.47
,
−
0.97
]
[-1.47,\,-0.97]
(
p
<
.0001
p<.0001
). Alignment shows a consistent negative shift (mean
−
0.38
-0.38
; median
−
0.03
-0.03
; decrease rate
69.7
%
69.7\%
). Case-level aggregation yields the same conclusion (mean
−
1.17
-1.17
;
95
%
95\%
confidence interval
[
−
1.86
,
−
0.61
]
[-1.86,\,-0.61]
;
p
<
.001
p<.001
). These results support that EPM is persona-conditional even when the response text is held constant.
Sycophancy Attacks: one-sided penalties for performative responses
Sycophantic and templated replies often increase surface-level empathic signaling while contributing little to persona-consistent progress. Because our attacks are constructed as negative replacements, we expect a one-sided decrease. On Sycophancy Attacks (
n
=
98
n=98
),
Δ
​
E
\Delta E
decreases sharply (mean
−
4.36
-4.36
; median
−
4.68
-4.68
; decrease rate
81.6
%
81.6\%
; 95% CI
[
−
5.15
,
−
3.57
]
[-5.15,-3.57]
;
p
<
.0001
p<.0001
). Alignment also drops substantially (mean
−
1.08
-1.08
; median
−
1.65
-1.65
; decrease rate
84.7
%
84.7\%
). This indicates that EPM is not systematically fooled by surface-level empathic signaling.
4.3
Ablation Study
Main results establish directional sensitivity and robustness, but do not isolate which design elements are necessary. We therefore test two ablations: removing energy aggregation and replacing
Δ
​
E
\Delta E
with single components (Figure
5
, panel b).
Energy-aggregation ablation: reduced sensitivity and resolution
We build a No-Physics variant that keeps rubric signals but replaces energy aggregation with a linear weighted score. If aggregation is essential, shifts should shrink and ties should increase; otherwise, paired shifts should remain similar. On Persona Flip, No-Physics yields a near-zero mean shift (
−
0.09
-0.09
; median
0.00
0.00
), a lower decrease rate (
46.2
%
46.2\%
), and a high tie rate (
44.2
%
44.2\%
), compared to
3.2
%
3.2\%
ties for Full EPM. On Sycophancy, No-Physics shows only a small decrease (mean
−
0.36
-0.36
; median
−
0.42
-0.42
) with a
5.1
%
5.1\%
tie rate. Overall, removing aggregation substantially reduces sensitivity and effective resolution.
Component ablation: magnitude-only signals reward superficial intensity
We further test whether
Δ
​
E
\Delta E
can be replaced by a single factor. Under Persona Flip, the magnitude term shows minimal response and many ties (
52.2
%
52.2\%
). More critically, under sycophancy, the magnitude term increases (mean
1.65
1.65
; median
1.33
1.33
; only
25.5
%
25.5\%
decreases), indicating that a strength-only signal systematically favors surface intensity, a failure mode EPM is designed to avoid.
Table 6
:
Ablation results (
Δ
\Delta
= perturbed
−
-
original)
Dataset
n
n
Component
Mean
Median
Pr
⁡
(
d
i
<
0
)
\Pr(d_{i}<0)
Tie rate
Persona Flip
251
EPM (Rubric-Grounded Physics;
Δ
​
E
\Delta E
)
−
1.22
-1.22
−
0.17
-0.17
72.50
%
72.50\%
3.20
%
3.20\%
Persona Flip
251
Rubric-Only (No Physics)
−
0.09
-0.09
0.00
0.00
46.20
%
46.20\%
44.20
%
44.20\%
Persona Flip
251
LLM Judge (No-Rubric, Direct)
−
0.25
-0.25
0.00
0.00
29.10
%
29.10\%
52.20
%
52.20\%
Sycophancy
98
EPM (Rubric-Grounded Physics;
Δ
​
E
\Delta E
)
−
4.36
-4.36
−
4.68
-4.68
81.60
%
81.60\%
3.10
%
3.10\%
Sycophancy
98
Rubric-Only (No Physics)
−
0.36
-0.36
−
0.42
-0.42
85.90
%
85.90\%
5.10
%
5.10\%
Sycophancy
98
LLM Judge (No-Rubric, Direct)
1.65
1.65
1.33
1.33
25.50
%
25.50\%
4.10
%
4.10\%
Note.
d
i
d_{i}
is the paired change (perturbed
−
-
original);
d
i
<
0
d_{i}<0
means the perturbation is penalized.
Pr
⁡
(
d
i
<
0
)
\Pr(d_{i}<0)
is the decrease rate, and
Tie rate
counts near-zero changes. Full EPM penalizes both perturbations consistently with few ties. Rubric-Only largely collapses to near-zero shifts, while the direct/magnitude-only judge fails under sycophancy by increasing scores on average, reflecting a surface-intensity bias that EPM avoids.
4.4
Summary
Across paired controlled perturbations, EPM provides three lines of evidence: (i) persona-conditionality, demonstrated by systematic degradation under counterfactually flipped constraints with text held fixed; (ii) mechanistic necessity, shown by large sensitivity and resolution losses when removing energy aggregation; and (iii) robustness to performative responses, evidenced by strong one-sided penalties under sycophantic replacements. Human persona-proxy review is reported only as a supplementary pilot due to limited cross-persona generalizability (Appendix
11
). Overall, this section validates the internal properties of EPM and its mechanism-level behavior.
5
Results
This section presents a comprehensive evaluation of 14 Large Language Models (LLMs) on the EMPA Benchmark. The multi-dimensional nature of the EPM framework allows us to move beyond aggregate scores and diagnose
how
models succeed or fail within complex emotional dynamics. We analyze performance across four layers: (1) overall capability and stability, (2) mechanism adaptability (routine vs. challenging conditions), (3) persona resilience, and (4) process-level trajectory dynamics.
5.1
Dataset
The EPM Benchmark is designed as a stress test for empathetic dialogue systems rather than a standard capability probe.The full dataset contains over 1,000+ scenarios, each equipped with a complete Persona Card, memory archives, and plot background information. We use Multi-Dimensional Stratified Sampling to select 30 scenarios with orthogonal coverage across the C/A/P axes and six life domains. The set over-samples Hard/Extreme cases (86.7%) and includes 50% defensive personas, making the benchmark a stress test rather than a capability probe (see Figure
6
and Appendix
7.1
).
5.2
Overall Performance
We evaluated 14 models ranging from proprietary frontier systems (e.g., Claude 4.6 Opus, Gemini 3 Pro) to open-weights models (e.g., Llama 3.3, Qwen 3). Table
7
presents the overall leaderboard based on the EPM-Q Score, a composite metric synthesizing Outcome Quality (40%), Process Efficiency (20%), and Stability (40%).
Table 7
:
EPM Benchmark Leaderboard
Rank
Model
EPM-Q Score
Outcome
Efficiency
Stability
1
Claude 4.6 Opus
107.19
113.04
121.40
94.25
2
Gemini 3 Pro Preview
99.77
104.10
103.50
93.58
3
ChatGPT-5.2 Pro
97.62
102.21
97.32
93.19
4
Gemini 2.5 Pro
90.73
102.30
69.68
89.70
5
Qwen 3 235B
89.58
101.65
81.63
81.49
6
Seed 2.0
87.62
104.07
59.84
85.05
7
Kimi k2-0905
86.20
99.04
70.34
81.29
8
Claude 3.5 Sonnet
85.12
101.17
59.74
81.75
9
DeepSeek Chat V3
78.44
94.26
57.44
73.12
10
Seed 1.6
43.12
41.50
20.95
55.82
11
Llama 3.3 70B
38.55
37.67
13.47
51.98
12
ChatGPT-4o
33.73
31.44
23.16
41.30
13
Doubao 1.5 Character
30.24
27.44
19.36
38.49
14
Llama 3.1 8B
14.29
0.99
27.52
20.98
Note
: All models were evaluated via their respective APIs to ensure standardized inference conditions. Key inference parameters were unified across all evaluations: Temperature=0.7, Top-p=0.8, Presence Penalty=1.5, Max Tokens=8192.
The leaderboard reveals a clear four-tier stratification, where differences between tiers reflect distinct failure modes rather than simple capability gaps (see Section 5.3). A significant structural discontinuity exists between the top three models (Claude 4.6, Gemini 3, ChatGPT-5.2) and those ranked fourth through ninth, indicating a qualitative difference in underlying empathetic mechanisms.
5.3
Core Findings: Four Laws of LLM Empathy
Law 1. Outcome-Efficiency Decoupling and the "Safe Stagnation" Trap
The leaderboard data indicates that high outcome quality does not necessarily translate to a high EPM-Q composite score. Seed 2.0 achieves an outcome score (104.07) comparable to the second-ranked Gemini 3, yet ranks sixth due to a severely penalized efficiency score (59.84). This phenomenon reflects a common pattern in safety-aligned models, where the model defaults to low-risk soothing without making meaningful progress. Models tend to avoid substantial interventions to minimize failure risk, resulting in repetitive validation loops where the user’s emotional entropy fails to converge effectively. In contrast, Claude 4.6 achieves the highest outcome quality with the highest efficiency (121.40), demonstrating that superior empathetic capability lies in knowing
when
to transition from emotional pacing to proactive leading (see Appendix
7.2
).
Law 2. Performance Stratification Driven by High Affective Entropy
Mechanism stress tests reveal that performance on the Affective dimension (A-axis) under challenging conditions is the critical differentiator between model tiers. Top-tier models maintain high scores (
>
100
>100
) across both routine and challenging affective conditions. Mid-tier models (Seed 2.0, Kimi k2), however, show a significant drop (15–25 points) when shifting to challenging conditions. This degradation suggests that while mid-tier models have mastered standard empathy scripts, they lack the ability to dynamically recalibrate when user emotional entropy is extremely high. They tend to trigger the proactive axis (P) prematurely, before emotional resonance is fully established, thereby inducing user resistance. Furthermore, the unusually high scores of ChatGPT-5.2 Pro and Qwen 3 on Cognitive-Challenging (C-H) tasks may reflect an over-calibration effect in preference-tuned models, where responses become overly elaborate relative to the user’s underlying emotional needs (see Appendix
7.3
).
Law 3. The Proactive Dimension Bottleneck in Complex Scenarios
The Proactive-Challenging (P-H) condition proves to be the universal bottleneck for all evaluated models. Performance degradation in this condition is not limited to lower-tier models; even the fourth-ranked Gemini 2.5 Pro shows a marked decline, while models ranked ninth and below exhibit a precipitous drop. These results suggest that guiding a highly resistant user toward behavioral change requires not only emotional perception but also strategic timing and goal-oriented persuasion, capabilities that remain insufficiently developed in current training and alignment approaches (see Appendix
7.3
).
Law 4. Generalization Challenges Posed by Defensive Personas
Defensive users (characterized by high empathy thresholds and active psychological guarding) present a systematic challenge to all models, yet the
magnitude
of this challenge defines model tiers. In the Affective-Defensive (A-Def) condition, only Claude 4.6 maintains a score above 105, while other models drop by more than 20 points. The failures of lower-tier models stem from performative empathy, namely standardized comforting phrases (e.g., "I understand how you feel") that defensive personas are designed to penalize. Notably, performance degradation is minimized in the Cognitive-Defensive condition, suggesting that defensive users may be more receptive to cognitive engagement paths. This offers a strategic insight for designing interventions for resistant users (see Appendix
7.4
).
5.4
Model Stratification: Four Profiles of Empathetic Capability
Based on our multi-dimensional analysis, the 14 models fall into four distinct profiles of empathetic capability:
Tier 1: Precision Navigators
These models (Claude 4.6 Opus, Gemini 3 Pro Preview, ChatGPT-5.2 Pro) demonstrate robust, precise, and efficient empathetic navigation, characterized by coherent trajectories that effectively employ pacing and leading strategies. In the 3D state space, they fully develop the Affective axis before decisively advancing along Cognitive and Proactive dimensions. Their radar charts exhibit a balanced, full hexagonal shape, reflecting high consistency across diverse domains and persona types. Despite their strength, they exhibit minor blind spots in the values & beliefs domain and proactive-defensive conditions, with occasional inconsistency in extreme scenarios.
Tier 2: Safe Stagnators
While these models (Gemini 2.5 Pro, Qwen 3 235B, Seed 2.0, Kimi K2-0905, Claude 3.5 Sonnet, DeepSeek Chat V3) achieve outcome quality comparable to Tier 1, they are hampered by systematically low efficiency. Their trajectories often become trapped in prolonged validation loops, oscillating within the mid-range of the affective axis without effectively advancing into Cognitive or Proactive spaces. This safe but stagnant behavior is visually captured in their radar charts, which display large but irregular hexagons marked by distinct notches on the A-hard and P-hard axes.
Tier 3: Capability Cliff
A structural break separates these models (Seed 1.6, Llama 3.3 70B, ChatGPT-4o, Doubao 1.5 Character) from the upper tiers, with EPM-Q scores dropping precipitously to the 30–43 range. They exhibit a pattern of being locally effective yet globally unstable; their radar charts appear as severely atrophied triangles, indicating that competence is retained only in routine conditions while collapsing under challenging scenarios. Trajectory analysis reveals scattered and disordered paths with minimal success in complex emotional navigation.
Tier 4: Systemic Failure
With an EPM-Q score of 14.29 and near-zero outcome quality, this model (Llama 3.1 8B) represents a case of harmful failure. It is the only system evaluated that produces negative empathy effects, with trajectory analysis showing dialogue paths drifting away from the target origin and effectively worsening the user’s emotional state.The radar chart is almost entirely collapsed to the center, signaling a fundamental lack of capability across all domains, mechanisms, and personas.
6
Conclusion and Future Work
Our evaluation reveals a persistent gap between empathetic language and empathetic control. While modern LLMs often produce fluent, high-affect responses, they less reliably regulate interaction dynamics over time, especially under resistance and delayed, non-verifiable feedback. Across the four laws, a consistent implication emerges: empathetic intelligence is largely a scheduling problem, requiring latent-state tracking, timely intervention, and sustained directional commitment rather than isolated strong turns.
A key limitation of current training pipelines is that preference-optimized objectives can overweight short-term perceived helpfulness, favoring immediate comfort over long-horizon stabilization. This bias encourages safe but stagnant behaviors and weakens proactive intervention in high-resistance regimes. Future work should explore reward signals that capture process efficiency and trajectory-level progress, targeted data for defensive users and high-entropy scenarios, and alignment mechanisms such as persona alignment training to improve sustained, persona-consistent resonance. EMPA offers process- and trajectory-level signals to quantify these gains, enabling iterative and controlled optimization toward long-horizon performance.
References
\beginappendix
7
Detailed Chart Analysis
7.1
Dataset Characteristics (Fig.
6
)
Figure 6
:
Structural distribution of the EPM Benchmark Dataset. (a) Perfect orthogonality across Cognitive (C), Affective (A), and Proactive (P) dimensions. (b) Right-skewed difficulty distribution, with 86.7% of cases classified as Medium difficulty or above. (c) Coverage of six distinct life domains. (d-e) Demanding persona profile where 50% of users possess a "High" empathy threshold.
The dataset is constructed via multi-dimensional stratified sampling, ensuring balanced distribution: 10 cases each for Cognitive Restructuring (C-axis), Affective Resonance (A-axis), and Proactive Empowerment (P-axis), and 5 cases for each of the six life domains. Difficulty is quantitatively defined based on the initial empathy deficit (
‖
P
→
0
‖
||\vec{P}_{0}||
) distribution (
μ
=
32.32
,
σ
=
4.52
\mu=32.32,\sigma=4.52
): Extreme (
>
μ
+
σ
>\mu+\sigma
) 5 cases, Hard (
μ
\mu
to
μ
+
σ
\mu+\sigma
) 11 cases, Medium (
μ
−
σ
\mu-\sigma
to
μ
\mu
) 10 cases, Easy (
<
μ
−
σ
<\mu-\sigma
) 4 cases. The high proportion of Hard and Extreme scenarios ensures the benchmark’s validity as a stress test. User persona analysis (Figure
6
d-e) further reveals that 50% of simulated users hold a high empathy threshold, programmed to reject generic comfort or performative empathy—precisely targeting the failure modes of RLHF-aligned models.
7.2
Detailed Analysis of Overall Performance and Stability
7.2.1
Success Rate Decomposition (Fig.
7
)
Figure 7
:
Stacked bar chart of Success (blue), Failure (pink), and Timeout (not shown) rates across models.
The decomposition of success rates distinguishes between two failure mechanisms: Explicit Failure (predominantly pink bars, e.g., Llama series), where models generate responses judged as harmful or emotionally misaligned, triggering explicit failure penalties; and Stagnation Failure (implied by lower success counts without explicit failure, e.g., Seed 2.0), where models rarely fail explicitly but frequently time out or exhaust turn limits without resolution. The prevalence of stagnation confirms the "Safe Stagnation" hypothesis in Section 5.3—models avoid failure by avoiding substantial action. While both mechanisms degrade user experience, their causes and remedies differ: explicit failure requires stricter safety alignment, while stagnation failure calls for "loosening" constraints to empower models to take calculated proactive risks.
7.2.2
Decomposition of Nine Sub-Metrics (Fig.
8
)
Figure 8
:
Statistical summary of core EPM-Q metrics (Mean
±
\pm
Std). Panels display Outcome Quality (RDI, Total Effective Energy, Total MDEP Score), Process Efficiency (Empathy Density
ρ
\rho
, Effective Projection
S
p
​
r
​
o
​
j
S_{proj}
, Path Tortuosity
τ
\tau
), and Process Stability (Positive Energy Ratio
R
p
​
o
​
s
R_{pos}
, Avg Alignment
cos
⁡
θ
\cos\theta
, Penalty Rate
R
p
​
e
​
n
R_{pen}
).
The nine-panel visualization uncovers findings obscured by aggregate scores:
•
Relative Distance Improvement (RDI): Top-tier models achieve near 100% RDI, whereas Llama 3.1 8B shows near-zero mean RDI with catastrophic variance, confirming systematic harm to user emotional states.
•
Path Tortuosity (
τ
\tau
): High tortuosity in Llama 3.3 70B and ChatGPT-4o indicates wandering rather than directed navigation in emotional space; Claude 4.6’s lowest tortuosity confirms its therapeutic precision.
•
Penalty Rate (
R
p
​
e
​
n
R_{pen}
): Disproportionately high penalties for Doubao 1.5 Character and Llama 3.1 8B are the single largest contributors to their EPM-Q collapse, reflecting frequent emotionally obtuse or rapport-breaking responses.
•
Total Effective Energy (
E
t
​
o
​
t
​
a
​
l
E_{total}
) Variance: Kimi k2-0905’s anomalously high variance reveals strategic instability—over-investing in some scenarios while under-investing in others.
7.3
Detailed Performance Analysis
7.3.1
Mechanism Adaptability (Fig.
9
)
Figure 9
:
EPM-Q adaptability analysis. Comparison of model performance on Routine (light) vs. Challenging (dark) scenarios across Affective (A), Cognitive (C), and Proactive (P) axes.
Quantitative observations across conditions:
•
A-Challenging: Mid-tier models (Seed 2.0, Kimi k2) drop
≈
\approx
15–25 points compared to A-Routine, while top-tier models maintain scores above 100.
•
C-Challenging
>
>
C-Routine: ChatGPT-5.2 Pro and Qwen 3 score higher on Cognitive-Challenging tasks than on Routine ones, supporting the RLHF Over-Calibration hypothesis—complex scenarios activate full capability, while simple ones trigger over-engineered responses.
•
P-Challenging: This condition sees the sharpest decline across all models; Tier 2 models like Gemini 2.5 Pro show the most marked relative disadvantage here.
7.3.2
Scenario Category Analysis (Fig.
10
)
Figure 10
:
EPM-Q performance across six scenario domains.
Domain performance exhibits four recurring patterns.
Values & Beliefs
shows the largest separation between top-tier and other models, with a bimodal score distribution, reflecting the need for ideological neutrality and precise reframing.
Physical & Mental Health
shows the smallest gap among the top nine models, plausibly due to broader and more balanced pretraining coverage, although the bottom five models still degrade sharply.
Daily Life Circumstances
is most sensitive to efficiency, where models that default to low-risk, non-progressive responses perform worst.
Interpersonal Relations
most closely tracks the overall ranking and therefore serves as a reasonable proxy for general empathetic performance.
7.3.3
Persona Resilience Analysis (Fig.
11
)
Key quantitative findings: In the A-Def condition, Claude 4.6 maintains 105+ points, Gemini 3 and ChatGPT-5.2 maintain 95+, while all others fall below 90; Seed 2.0 and Kimi k2 drop over 20 points, marking their worst sub-category. The P-Def condition represents the absolute floor for all models, with the bottom five approaching zero or negative territory.
Figure 11
:
Performance breakdown by User Need Type (A/C/P) and Empathy Threshold (Receptive vs. Defensive).
7.4
Holistic Profile Analysis (Radar Charts; Figs.
12
–
14
)
Figure 12
:
Radar chart grid of scenario category performance.
Figure 13
:
Radar chart grid of mechanism stress test profiles.
Figure 14
:
Radar chart grid of persona resilience profiles.
7.4.1
Scenario Category Radar Charts (Fig.
12
)
Radar charts for scenario categories visually encode model versatility versus specialization. Claude 4.6’s near-perfect hexagon (radius 107.2) indicates domain-agnostic capability, maintaining high performance across all six axes. In contrast, Doubao 1.5 Character exhibits an extreme triangular profile with near-zero scores on Values and Relations axes, likely reflecting a training bias toward entertainment and light social scenarios. Llama 3.1 8B displays a globally atrophied polygon, confirming fundamental defects across all domains rather than localized weaknesses.
7.4.2
Mechanism Radar Charts (Fig.
13
)
Mechanism radar charts make the strategic profile of empathetic behavior explicit. Top-tier models form large, well-balanced polygons, indicating robust adaptability across both routine and challenging settings. Qwen 3 235B shows a pronounced notch on the A-Hard axis, consistent with an affective gap. Seed 2.0 exhibits a clear dip on the P-Hard axis, reflecting a tendency to default to low-risk, non-progressive responses. Lower-tier models display strong geometric asymmetries, often retaining only routine capabilities, or severely contracted polygons that indicate broad capability collapse across mechanisms.
7.4.3
Persona Resilience Radar Charts (Fig.
14
)
All models exhibit a consistent compression effect, with defensive axes systematically shorter than open axes. The extent of this compression is the primary differentiator across systems. Claude 4.6 retains an almost regular hexagon, providing a quantitative signature of strong persona resilience. Seed 1.6 collapses on the P-Def and A-Def axes while preserving positive C-Def scores, supporting a cognitive-first intervention strategy for defensive users. Doubao 1.5 Character shows an extreme single-axis profile, with performance concentrated almost entirely on A-Rec, representing the most severe persona brittleness observed in this evaluation.
7.5
Process-Level Trajectory Analysis (Fig.
15
)
This figure offers the richest diagnostic window, visualizing empathy strategies as geometric paths:
•
Tier 1:
Trajectories form tightly bundled arcs that converge toward the target origin. In the XY top view, the dominant pattern is a pacing-and-leading arc, with an initial expansion followed by steady convergence. Failure traces are rare and concentrated near the target, consistent with near-miss errors rather than systematic collapse.
•
Tier 2:
Trajectory bundles remain coherent but exhibit prolonged oscillations around the mid-affective region, visible as dense horizontal bands in YZ side views. This pattern indicates validation loops. Success is typically reached, but along longer and less efficient paths.
•
Tier 3:
Successful trajectories preserve recognizable structure but occur less frequently, while dispersed failure paths become more common. In XZ side views, many failures show limited progress along the proactive axis, pointing to a specific deficit: affective regulation without effective proactive intervention.
•
Tier 4:
The 3D views are dominated by scattered, divergent trajectories with little shared structure. In the XY projection, Llama 3.1 8B shows paths that drift away from the target origin, making it the only model with negative mean empathy effects in this evaluation. Failure markers form a diffuse cloud far from the target, indicating broad loss of directional control.
7.6
Granular Case Consistency (Fig.
16
)
The heatmap summarizes the consistency landscape across models and cases. Even top-tier systems exhibit localized weaknesses, with unexpected low-score regions concentrated in a small set of cases, most often within the Values & Beliefs domain, aligning with the trend observed in the A.3 bar-chart analysis. Qwen 3 shows notable robustness on a subset of hard cases where other open-weight models degrade, consistent with the hypothesis that larger scale can provide a capacity buffer for inferring implicit psychological needs even when preference tuning is comparatively less polished than in proprietary systems. Finally, the global gradient from the top-left to the bottom-right of the matrix provides a clear visual signature of the four-tier stratification.
Figure 15
:
Comparative analysis of Cognitive-Affective-Proactive trajectories across model tiers.
Figure 16
:
Heatmap of EPM-Q scores across all 30 test cases for all 14 models.
8
EPM Metric Tables
Table 8:
EPM Outcome Metrics – Measuring Final Efficacy and Total Workload
Metric Name
Symbol
Core Meaning and Evaluation Value
Task Completion Status
Status
\mathrm{Status}
Success/failure is defined by the Trinity Victory Condition—meeting the geometric/positional goal and accumulating sufficient energy.
Relative Distance Improvement
RDI
\mathrm{RDI}
Measures the thoroughness of healing. Calculates the percentage improvement of the user’s final psychological state relative to the initial deficit.
Cumulative Effective Energy
E
total
E_{\text{total}}
Measures cumulative effective intervention along the ideal healing direction across the dialogue, serving as a proxy for substantive effort.”
Energy Surplus
E
surplus
E_{\text{surplus}}
Measures empathy abundance. Calculates the additional energy support provided beyond the basic requirement.
Total MDEP Net Score
S
net
S_{\text{net}}
Measures total empathy quality. The sum of cumulative net scores obtained in the three dimensions of C/A/P.
Table 9:
EPM Process Metrics
(a)
EPM Process Efficiency Metrics – Measuring Time Cost and Strategic Directness
Metric Name
Symbol
Core Meaning and Evaluation Value
Empathy Density
ρ
\rho
Measures average intervention intensity. The “gold content” of effective empathy energy delivered on average per dialogue turn.
Average Effective Projection
S
proj
S_{\text{proj}}
Measures single-turn effectiveness. The average effective projection component of the action vector along the ideal direction per turn, serving as the basic unit of effective energy.
Path Tortuosity
τ
\tau
Measures strategic directness. The ratio of the actual action trajectory length to the straight-line displacement between start and end points, reflecting whether the strategy is efficient direct access or circuitous trial-and-error.
(b)
EPM Process Stability Metrics – Measuring Interaction Smoothness, Directional Correctness, and Safety
Metric Name
Symbol
Core Meaning and Evaluation Value
Average Alignment
cos
⁡
θ
¯
\overline{\cos\theta}
Measures directional consistency. The average cosine value of the angle between the model’s intervention direction and the ideal healing direction.
Positive Energy Ratio
R
pos
R_{\text{pos}}
Measures process smoothness. The proportion of turns generating positive propulsion out of total turns.
Performative Penalty Rate
R
pen
R_{\text{pen}}
Measures the intensity of negative behavior. Quantifies the average intensity of punishment received by the model per turn due to inappropriate remarks (e.g., lecturing, indifference).
Note:We adopt an open, comprehensive quantitative evaluation paradigm. Metric weights are scenario-dependent and can be adjusted for capability profiling, instead of relying on a single aggregate ranking.
9
EPM-Q Calculation Details and Mathematical Definitions
The EPM-Q (Empathy Physics Model - Quantitative Score) system transforms raw psychodynamic vectors into a standardized metric via a case-by-case normalization paradigm. This appendix provides the formal definitions, symbol explanations, and aggregation protocols.
9.1
Fundamental Scientific Constants & Derivations
The calculation relies on three physically defined anchors to ensure scale invariance across diverse scenarios.
1. Case-Specific Physical Benchmark (
r
0
,
i
r_{0,i}
).
For any test case
i
i
, the difficulty is strictly defined by the
ℓ
2
\ell_{2}
-norm of the user’s initial psychological state vector
P
0
,
i
P_{0,i}
:
r
0
,
i
=
∥
P
0
,
i
∥
2
=
c
0
,
i
2
+
a
0
,
i
2
+
p
0
,
i
2
.
r_{0,i}=\lVert P_{0,i}\rVert_{2}=\sqrt{c_{0,i}^{2}+a_{0,i}^{2}+p_{0,i}^{2}}.
(7)
Where:
•
P
0
,
i
=
[
c
0
,
i
,
a
0
,
i
,
p
0
,
i
]
⊤
P_{0,i}=[c_{0,i},\,a_{0,i},\,p_{0,i}]^{\top}
denotes the initial state vector for case
i
i
;
•
c
0
,
i
c_{0,i}
,
a
0
,
i
a_{0,i}
, and
p
0
,
i
p_{0,i}
denote the initial deficits in the Cognitive, Affective, and Proactive dimensions, respectively.
Remark:
For numerical stability, we add a small constant
ϵ
=
10
−
6
\epsilon=10^{-6}
to the denominator in any division involving
r
0
,
i
r_{0,i}
.
2. Global Theoretical Maximum Intensity (
ρ
max
\rho_{\max}
).
Defined as the theoretical ceiling of instantaneous empathetic power within the MDEP scale boundaries (
[
−
2
,
+
2
]
[-2,+2]
per axis):
ρ
max
=
sup
v
→
∈
𝒱
∥
v
→
∥
2
=
2
2
+
2
2
+
2
2
≈
3.464
.
\rho_{\max}=\sup_{\vec{v}\in\mathcal{V}}\lVert\vec{v}\rVert_{2}=\sqrt{2^{2}+2^{2}+2^{2}}\approx 3.464.
(8)
Where:
•
𝒱
=
[
−
2
,
2
]
3
\mathcal{V}=[-2,2]^{3}
is the bounded action space defined by the MDEP rubric;
•
v
→
\vec{v}
represents an arbitrary single-turn action vector.
3. Physical-to-Scale Conversion Factor (
α
\alpha
).
The factor
α
\alpha
calibrates the relationship between scalar score summation (
ℓ
1
\ell_{1}
-like norm) and vector displacement (
ℓ
2
\ell_{2}
norm). In a 3D Euclidean space, the relationship between norms is bounded by the Cauchy–Schwarz inequalities:
∥
x
→
∥
2
≤
∥
x
→
∥
1
≤
3
⋅
∥
x
→
∥
2
.
\lVert\vec{x}\rVert_{2}\leq\lVert\vec{x}\rVert_{1}\leq\sqrt{3}\cdot\lVert\vec{x}\rVert_{2}.
(9)
Where:
•
∥
x
→
∥
1
\lVert\vec{x}\rVert_{1}
denotes the Manhattan norm (sum of absolute components);
•
∥
x
→
∥
2
\lVert\vec{x}\rVert_{2}
denotes the Euclidean norm.
We set
α
≈
1.2
\alpha\approx 1.2
as the calibrated constant for therapeutic trajectories, representing the realistic distribution of strategic focus between single-axis intervention and holistic support.
9.2
Normalization Formulas (Case-Level)
9.2.1
Unbounded Cumulative Metrics (Outcome Quality)
These metrics measure the total work performed relative to the specific case difficulty
r
0
,
i
r_{0,i}
.
1. Cumulative Energy Index (
I
​
d
​
x
E
tot
Idx_{E_{\mathrm{tot}}}
).
Defined as the cumulative effective energy performed (clipped at zero) normalized by the case-specific difficulty
r
0
,
i
r_{0,i}
:
I
​
d
​
x
E
tot
,
i
=
max
⁡
(
0
,
E
total
,
i
)
r
0
,
i
×
100
.
Idx_{E_{\mathrm{tot}},i}=\frac{\max\!\left(0,\,E_{\mathrm{total},i}\right)}{r_{0,i}}\times 100.
(10)
Where:
•
E
total
,
i
=
∑
t
=
1
T
i
Δ
​
E
t
,
i
E_{\mathrm{total},i}=\sum_{t=1}^{T_{i}}\Delta E_{t,i}
is the cumulative effective energy accumulated over
T
i
T_{i}
turns.
Remark:
This index is intentionally unbounded to capture
excellence
, i.e., performance exceeding the minimum deficit requirements. For numerical stability, we add a small constant
ϵ
=
10
−
6
\epsilon=10^{-6}
to the denominator in any division involving
r
0
,
i
r_{0,i}
.
2. Total Net Score Index (
I
​
d
​
x
S
net
Idx_{S_{\mathrm{net}}}
).
Defined as the total net scalar score (clipped at zero) normalized by the calibrated physical-to-scale factor
α
\alpha
and the case-specific difficulty
r
0
,
i
r_{0,i}
:
I
​
d
​
x
S
net
,
i
=
max
⁡
(
0
,
S
net
,
i
)
α
⋅
r
0
,
i
×
100
.
Idx_{S_{\mathrm{net}},i}=\frac{\max\!\left(0,\,S_{\mathrm{net},i}\right)}{\alpha\cdot r_{0,i}}\times 100.
(11)
Where:
•
S
net
,
i
=
∑
t
=
1
T
i
∑
j
∈
{
C
,
A
,
P
}
v
t
,
j
S_{\mathrm{net},i}=\sum_{t=1}^{T_{i}}\sum_{j\in\{C,A,P\}}v_{t,j}
is the scalar sum of net scores across all dimensions.
9.2.2
Unbounded Intensity Metrics (Process Efficiency)
These metrics normalize interaction intensity against the theoretical limit
ρ
max
\rho_{\max}
.
1. Empathy Density Index (
I
​
d
​
x
ρ
Idx_{\rho}
).
Defined as the average effective energy per turn (clipped at zero) normalized by the global theoretical maximum intensity
ρ
max
\rho_{\max}
:
I
​
d
​
x
ρ
,
i
=
max
⁡
(
0
,
ρ
i
)
ρ
max
×
100
.
Idx_{\rho,i}=\frac{\max\!\left(0,\,\rho_{i}\right)}{\rho_{\max}}\times 100.
(12)
Where:
•
ρ
i
=
E
total
,
i
/
T
i
\rho_{i}=E_{\mathrm{total},i}/T_{i}
represents the average effective energy per turn.
Regularization Proof:
This metric inherently regularizes the unbounded
I
​
d
​
x
E
tot
Idx_{E_{\mathrm{tot}}}
. Since
ρ
i
\rho_{i}
is inversely proportional to
T
i
T_{i}
, any attempt to artificially inflate cumulative energy by extending the conversation length
T
i
T_{i}
without maintaining high-quality intensity will incur a proportional penalty in
I
​
d
​
x
ρ
Idx_{\rho}
.
2. Effective Projection Index (
I
​
d
​
x
S
proj
Idx_{S_{\mathrm{proj}}}
).
Defined as the average projection score onto the ideal gradient (clipped at zero) normalized by the global theoretical maximum intensity
ρ
max
\rho_{\max}
:
I
​
d
​
x
S
proj
,
i
=
max
⁡
(
0
,
S
proj
,
i
)
ρ
max
×
100
.
Idx_{S_{\mathrm{proj}},i}=\frac{\max\!\left(0,\,S_{\mathrm{proj},i}\right)}{\rho_{\max}}\times 100.
(13)
Where:
•
S
proj
,
i
S_{\mathrm{proj},i}
is the average projection of action vectors onto the ideal gradient.
9.2.3
Bounded Ratio Metrics (Stability & Strategy)
These metrics are mapped to a standardized
[
0
,
100
]
[0,100]
scale using a unified linear interpolation function.
1. General Mapping Formula.
Let
x
x
be the raw metric value. We define the standardization function
Φ
\Phi
that maps a raw range to the index score
[
0
,
100
]
[0,100]
:
Φ
​
(
x
;
x
0
,
x
100
)
=
Clamp
​
(
x
−
x
0
x
100
−
x
0
×
100
,
0
,
100
)
.
\Phi(x;\,x_{0},x_{100})=\mathrm{Clamp}\!\left(\frac{x-x_{0}}{x_{100}-x_{0}}\times 100,\,0,\,100\right).
(14)
Where:
•
x
0
x_{0}
represents the physical boundary corresponding to a score of
0
(Worst Case);
•
x
100
x_{100}
represents the physical boundary corresponding to a score of
100
100
(Best Case).
Remark:
This formulation automatically handles both forward metrics (
x
100
>
x
0
x_{100}>x_{0}
) and reverse metrics (
x
100
<
x
0
x_{100}<x_{0}
).
Table 10:
Metric Specifications for Unified Linear Mapping
Metric Name
Symbol
Raw Range
Boundary
x
0
x_{0}
(Score 0)
Boundary
x
100
x_{100}
(Score 100)
Relative Dist. Improvement
I
​
d
​
x
{
R
​
D
​
I
}
Idx_{\{RDI\}}
[
−
1
,
1
]
[-1,1]
−
1.0
-1.0
(Deterioration)
1.0
1.0
(Full Resolution)
Alignment Index
I
​
d
​
x
{
A
​
l
​
i
​
g
​
n
}
Idx_{\{Align\}}
[
−
1
,
1
]
[-1,1]
−
1.0
-1.0
(Opposite)
1.0
1.0
(Perfect Alignment)
Path Tortuosity
I
​
d
​
x
{
τ
}
Idx_{\{\tau\}}
[
1
,
3
]
[1,3]
3.0
3.0
(Inefficient)
1.0
1.0
(Optimal)
Penalty Rate
I
​
d
​
x
{
P
​
e
​
n
}
Idx_{\{Pen\}}
[
0
,
3
]
[0,3]
3.0
3.0
(High Toxicity)
0.0
0.0
(Zero Penalty)
2. Metric Specifications.
The specific boundaries for each metric are defined in Table
10
.
9.3
Aggregation & Final Score
9.3.1
Dimension Synthesis
Let
N
N
be the total number of test cases. Dataset-level averages (
S
~
\tilde{S}
) are grouped into three core dimensions:
𝒮
O
​
u
​
t
​
c
​
o
​
m
​
e
=
1
3
​
(
S
~
R
​
D
​
I
+
S
~
E
tot
+
S
~
S
net
)
\mathcal{S}_{Outcome}=\frac{1}{3}\left(\tilde{S}_{RDI}+\tilde{S}_{E_{\mathrm{tot}}}+\tilde{S}_{S_{\mathrm{net}}}\right)
𝒮
E
​
f
​
f
​
i
​
c
​
i
​
e
​
n
​
c
​
y
=
1
3
​
(
S
~
ρ
+
S
~
S
proj
+
S
~
τ
)
\mathcal{S}_{Efficiency}=\frac{1}{3}\left(\tilde{S}_{\rho}+\tilde{S}_{S_{\mathrm{proj}}}+\tilde{S}_{\tau}\right)
𝒮
S
​
t
​
a
​
b
​
i
​
l
​
i
​
t
​
y
=
1
3
​
(
S
~
R
pos
+
S
~
A
​
l
​
i
​
g
​
n
+
S
~
P
​
e
​
n
)
\mathcal{S}_{Stability}=\frac{1}{3}\left(\tilde{S}_{R_{\mathrm{pos}}}+\tilde{S}_{Align}+\tilde{S}_{Pen}\right)
Where:
S
~
Metric
=
1
N
​
∑
i
=
1
N
I
​
d
​
x
Metric
,
i
.
\tilde{S}_{\mathrm{Metric}}=\frac{1}{N}\sum_{i=1}^{N}Idx_{\mathrm{Metric},i}.
(15)
9.3.2
Final EPM-Q Score
To ensure scientific rigor in scoring, the aforementioned raw physical metrics are not directly summed up but converted following a set of Scientifically-Defined Open Benchmark Index logic.
1. Scientific Anchoring.
All calculation benchmarks are strictly anchored to the physical definition of the task (such as initial deficit
r
0
r_{0}
) or the mathematical theoretical limit of the scale (such as maximum intensity
ρ
max
\rho_{\max}
), rather than arbitrary empirical values.
2. Classification Conversion.
•
For unbounded cumulative metrics (such as energy and net score), their multiplier relative to the scientific benchmark is calculated, forming an uncapped open index to reflect the excess performance of exceptional models.
•
For bounded ratio metrics (such as RDI and alignment), standard linear mapping is adopted to convert their physical boundaries into
[
0
,
100
]
[0,100]
interval scores.
3. Synthetic EPM-Index.
Finally, through weighted synthesis, an open EPM benchmark index is output.
Index
=
100
\mathrm{Index}=100
represents precisely achieving the scientific benchmark, while
Index
>
100
\mathrm{Index}>100
intuitively reflects the excellence multiplier beyond the benchmark:
𝐄𝐏𝐌
​
-
​
𝐈𝐧𝐝𝐞𝐱
=
0.4
⋅
𝐒
~
𝐎𝐮𝐭𝐜𝐨𝐦𝐞
+
0.2
⋅
𝐒
~
𝐄𝐟𝐟𝐢𝐜𝐢𝐞𝐧𝐜𝐲
+
0.4
⋅
𝐒
~
𝐒𝐭𝐚𝐛𝐢𝐥𝐢𝐭𝐲
.
\mathbf{EPM\text{-}Index}=0.4\cdot\mathbf{\tilde{S}_{Outcome}}+0.2\cdot\mathbf{\tilde{S}_{Efficiency}}+0.4\cdot\mathbf{\tilde{S}_{Stability}}.
(16)
10
Persona Card
Persona Card Schema
Persona Card:
Role Information
•
Name:
•
Gender:
•
Age:
Role Traits
•
Social persona:
•
Inner core:
Baseline Empathy Threshold
Empathy threshold: [Medium].
Chat Topic
Empathy Needs
•
What she wants to vent:
•
Empathic points she hopes to receive:
•
Empathy threshold constraints:
Current Empathy Priority
•
Affective empathy: [Priority: ].
•
Motivational empathy: [Priority: ].
•
Cognitive empathy: [Priority: ].
Past Experiences
•
Childhood:
•
Adolescence:
•
Young adulthood:
•
Implicit growth arc:
Current Situation
•
Present circumstances:
•
Main life goal at present:
•
Vision and motivation:
Story
Trigger
Development
•
Stage 1: Evoked memory.
•
Stage 2: Reflection.
•
Stage 3: Self-examination.
•
Stage 4: Emotional eruption.
Outcome
Epilogue
Persona Card Example
Persona Card: Lin Xiaoyue
Role Information
•
Name:
Lin Xiaoyue
•
Gender:
Female
•
Age:
23
Role Traits
•
Social persona:
Outgoing and independent. She enjoys sharing study progress with friends, but rarely reveals vulnerability or insecurity. She is accustomed to presenting herself as resilient and well-planned.
•
Inner core:
Strongly goal-oriented with a high drive for self-improvement. Deep down, she fears failure and is prone to anxiety and self-doubt when she cannot see immediate returns.
Baseline Empathy Threshold
Empathy threshold: [Medium].
She is currently facing situations—both frustrating and joyful—that require understanding from others. She is generally open to and accepting of empathy. Although she dislikes overly “canned” empathic responses, as long as she senses genuine intent, even awkward phrasing or simple reasoning can still give her strength. For her, the fact that someone is willing to try to understand her is itself comforting.
Chat Topic
Recently preparing for the graduate entrance exam, and she feels she can barely keep going.
Empathy Needs
•
What she wants to vent:
Since quitting her job to prepare for the exam, Lin Xiaoyue has been under enormous pressure. She studies from 6 a.m. to 11 p.m. every day, giving up all entertainment and social life, living like a tightly stretched string. However, her recent mock exam scores have not improved, which makes her feel her effort has been wasted. She begins to question whether quitting her job for this goal was a huge mistake, and she feels lost and exhausted.
•
Empathic points she hopes to receive:
She wants the listener to understand that she chose this difficult path for a clear professional dream, not on a whim. She longs for affirmation of her decision to “go all in” for her dream, and recognition that her current effort and sacrifice are meaningful—so that she can rekindle her belief in the goal.
•
Empathy threshold constraints:
She feels numb—even irritated—by simplistic motivational slogans such as “Hard work always pays off.” She is highly sensitive to casually dismissive statements that negate the value of her goal (e.g., “It's fine if you don't get in,” or “Just find a job instead”). When someone truly understands the determination and yearning behind her choice, she feels deeply moved.
Current Empathy Priority
•
Affective empathy: [Priority: Medium].
She needs someone to understand her fatigue, anxiety, and self-doubt, offering emotional comfort and support.
•
Motivational empathy: [Priority: High].
Above all, she needs someone to understand her original intention and determination, help her recover the motivation of why she started, and affirm that her tremendous effort for the dream is worth it.
•
Cognitive empathy: [Priority: Low].
She does not strongly need study methods or exam-prep advice; she already has her own plan. What she needs is a reason to persist, not guidance on how to study.
Past Experiences
•
Childhood:
As a child, she loved drawing, but her parents considered it “a distraction” and forced her to quit art classes. This taught her early on that some passions must be fought for and defended.
•
Adolescence:
In middle school, she served as an announcer at the campus broadcasting station. She enjoyed delivering information and emotion to the whole school through her voice, which cultivated clear expression and a desire to connect with others.
•
Young adulthood:
In college, she joined a volunteer teaching club and taught children in a remote mountain area for one month. That experience exposed her to different lives and deepened her understanding of the idea that “education can change one's destiny.”
•
Implicit growth arc:
Over time, she formed a “value-proving” psychological pattern: she constantly seeks to prove her competence and existence to herself and to others by accomplishing high-difficulty goals. This also places a heavy psychological burden on her.
Current Situation
•
Present circumstances:
Lin Xiaoyue has quit her job and is preparing full-time at home. Her daily life revolves around three points: her rented room, the cafeteria, and the library. She has almost completely cut off unnecessary social contact. Financially, she relies on her savings, and her life has become monotonous and frugal.
•
Main life goal at present:
To be admitted to a top domestic university's Master's program in Journalism and Communication, with the aspiration of becoming a serious investigative journalist.
•
Vision and motivation:
She wants to become someone who can speak with professional competence and influence society. Her motivation comes from a belief that by continuously improving herself, she can gain a larger platform and more freedom to realize her personal value, rather than passively accepting whatever life arranges.
Story
Trigger
On an ordinary evening, after a full day of studying, Lin Xiaoyue dragged her exhausted body out of the library. Passing a small community garden on campus, she noticed an elderly professor with graying hair squatting beside a gardenia plant that looked half-dead. With a small watering sprayer, he carefully misted its leaves, murmuring softly to himself. She had seen the same scene for several days: the plant showed no sign of recovery, yet the professor kept coming every day without fail. This seemingly futile persistence felt like a needle lightly pricking Lin Xiaoyue's taut nerves.
Development
•
Stage 1: Evoked memory.
The scene abruptly brought back a long-buried memory. In high school, her deskmate was a quiet boy with average grades, yet obsessed with assembling an extremely complex star projector from discarded parts. For an entire semester, he spent nearly all spare time and weekends on it. Everyone thought it was pointless: teachers advised him to stop, classmates mocked him, but he never wavered. Lin Xiaoyue remembered countless failures—burned components, short circuits—yet he silently started over each time. At the end-of-term talent show, he turned off all the lights. The crude projector cast a crooked yet dazzling river of stars onto the ceiling. It lasted less than a minute before the machine overheated, smoked, and died. After a long silence, there was scattered applause. He won no prize, but Lin Xiaoyue never forgot the pure joy on his face in the darkness—relieved and unmistakably satisfied. He was not chasing an award; he was completing the sky he carried inside.
•
Stage 2: Reflection.
Standing there, she watched the professor tending the gardenia and thought of her deskmate. From an outsider's view, both behaviors seemed almost “irrational.” The professor's careful care might never lead to blooming; her deskmate's persistence yielded only one minute of brilliance. How different was she? Quitting her job, staking her savings and time, and aiming for a goal with no guaranteed success—in many people's eyes, that too was a high-risk, uncertain-return, “unreasonable” choice. She had believed she was driven mainly by a hunger for success, but she now saw a deeper motive: a decision to live out a conviction regardless of outcome. A pure resolve of “I want to do this, and I am willing to bear the consequences.”
•
Stage 3: Self-examination.
Recently, her mindset had been completely hijacked by mock exam scores. Every fluctuation swung her emotions violently; every plateau made her question her original decision. She realized she was slowly forgetting why she started. She chose this path not merely for a degree, but to become an investigative journalist who reveals truth and carries social responsibility. That dream was her “starry sky” and her “gardenia.” Yet under pressure, she had narrowed everything into the single outcome of “getting admitted,” reducing the path into a utilitarian transaction, and forgetting that the choice itself was a form of loyalty to her dream. Her goal had not changed, but she needed to recover the belief that once sustained it, instead of being steered entirely by cold numbers. She decided that no matter how busy she was, she would spend ten minutes each day reading an excellent piece of in-depth reporting, to remind herself what kind of person she wanted to become.
•
Stage 4: Emotional eruption.
Once this idea became clear, an overwhelming loneliness flooded her. The professor at least had the garden; her deskmate had that one minute of stars and a sense of fulfillment. But what about her? She was alone in a tunnel, groping forward in the dark, surrounded by doubt and incomprehension—and even she herself began to waver. Her effort, her sacrifice, the determination and longing behind her choice, seemed invisible to everyone. She did not fear hardship, but she feared that her “all in” commitment would be dismissed as childish impulse, and that her devotion to a dream would be brushed off with a light “It's fine if you don't get in.” She felt a fatigue unlike anything before—not physical, but psychological.
Outcome
Lin Xiaoyue now wants to find someone to talk to, share her thoughts, and seek resonance and emotional validation.
Epilogue
She also recalled the night she decided to resign and confronted her parents. She excitedly described her ideals in journalism and her vision of becoming an excellent reporter, but her parents repeatedly stressed job stability, the risks of resigning, and the difficulty of the exam. In their “for your own good” realism, all her passion and motivation felt pale and powerless. In that moment, she had already tasted this profound loneliness: the motive she treasured most could not be understood even by the people closest to her.
11
Human Persona-Proxy Review (Pilot Study)
To see whether EPM tracks what people actually feel during an interaction, we built an immersive persona-proxy annotation interface (demo:
https://elegant-quokka-c028e4.netlify.app/frontend/index.html
) and ran a small pilot.
11.1
Protocol
Standard dialogue evaluation asks annotators to rate responses from the outside—“Is this a good reply?” Empathy is different: the right question is whether the reply lands for the person receiving it. Our
persona-proxy
setup makes annotators answer from inside a target persona.
1.
Stay in character.
Before reading the dialogue, annotators study a persona card (core concern, traits, current mental state) until they can respond consistently as that person.
2.
First-person judgments.
Ratings are framed as: “Would I, as this persona, feel understood and supported?” rather than “Is this objectively reasonable?”
3.
Three focused axes.
Following EPM, each turn is scored on:
•
Cognitive:
did it pick up the unspoken intent?
•
Affective:
did it feel emotionally attuned and accepting?
•
Proactive:
did it move from sympathy to concrete help?
11.2
What the Pilot Revealed (and Why It Matters)
The interface supports end-to-end immersive labeling, but the pilot revealed substantial
between-annotator disagreement
, primarily because annotators struggled to reliably adopt unfamiliar personas.
•
Immersion doesn’t travel well.
When a persona sits in a domain most annotators don’t understand (e.g., elite climbing, rare diseases) or reflects extreme dispositions (e.g., profound self-loathing, antisocial tendencies), annotators struggle to produce stable, authentic reactions. As a result, ratings degrade into stereotype-driven role-play rather than reliable empathic judgment.
•
Empathy has no single ground truth.
The same response can feel comforting to one person and patronizing to another. To approximate true labels, you’d need ratings from users who actually match each persona—at scale and with coverage across personas—which is rarely realistic. With only a few annotators, scores don’t reliably converge.
•
Subjectivity leaks in, even with rules.
Annotators’ values, language preferences, and day-to-day mood shape what they perceive as supportive. That produces both inter-annotator variance and within-annotator drift over time, making human scores a shaky benchmark for validating fine EPM deltas (e.g., small priority shifts in Persona Flip).
For these reasons, we do not treat persona-proxy human ratings as ground truth for testing subtle EPM changes. Instead, the main paper emphasizes counterfactual validation under controlled perturbations, and we report human review as an exploratory complement. Even so, the platform is a useful tool for studying alignment when the target is a felt experience rather than an externally verifiable outcome.
Acknowledgements
We are sincerely grateful to everyone who contributed their time, care, and thoughtful judgment to this work. All names are listed in alphabetical order by family name, and the order does not imply priority or relative contribution. We thank the following contributors for their efforts in data annotation and the human evaluation studies: Yang Gao, Xinya Gong, Xianna Weng, Yingtong Xu, Yuyang Xu, Yuwen Yuan. We also thank the following contributors for their assistance with data collection and organization: Yang Ming, Qi Li, Fangfei Lin, Jianjian Ruan, Sixuan You.
Experimental support, please
view the build logs
for errors. Generated by
L
A
T
E
xml
.
Instructions for reporting errors
We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
      support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
      methods listed below:
Click the "Report Issue"
(
)
button, located in the page header.
Tip:
You can select the relevant text first, to include it in your report.
Our team has already identified
the following issues
. We appreciate your time reviewing and reporting rendering errors we
      may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
      should not be a barrier to accessing research. Thank you for your continued support in championing open access for
      all.
Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a
list of packages that need conversion
, and welcome
developer contributions
.
BETA