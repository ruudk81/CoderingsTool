# Dimension-Specific Taxonomy Definitions

## Purpose

This document defines how Domain (L2), Facet (L3), and Attribute (L4) are interpreted within each of the 11 dimensions. It serves as the authoritative blueprint for the dimension definitions in `dimension_data.py`.

## Relationship to `taxonomy_logic.md`

`taxonomy_logic.md` defines the stable, dimension-independent meanings of each taxonomy level. This document extends those definitions with dimension-specific interpretations: the meaning of Domain, Facet, and Attribute adapts depending on which dimension has been selected, because each dimension captures a different type of information.

## Taxonomy hierarchy

Dimension (L1) → Domain (L2) → Facet (L3) → Attribute (L4).

Conceptual progression: `information type → subject → analytical lens → observable property`

Each level represents a qualitatively different analytical layer.

| Level | Name | Stable meaning | Question it answers |
|-------|------|---------------|-------------------|
| L1 | Dimension | The type of information or informational role | What type of information does this statement provide? |
| L2 | Domain | The subject the statement refers to | What is this statement about? |
| L3 | Facet | The analytical lens applied to the subject | Through what analytical lens is the subject being examined? |
| L4 | Attribute | A named observable property (not a verbatim span) | What specific characteristic is being described? |

The dimension is selected once per dataset. Once selected, its definition determines how Domain, Facet, and Attribute are interpreted for all downstream processing.

**Important:** Attribute (L4) is always a *named observable property* — never a verbatim quote from the response. Verbatim spans belong to the abstraction ladder (extraction metadata), which is separate from the taxonomy.

## Dimensions covered

11 dimensions total: 10 substantive dimensions in decision-tree priority order, plus 1 fallback (GENERAL_OTHER).

---

# 1 PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS

**Semantic variation:** statements propose actions or improvements

## Domain (L2) within PRESCRIPTIVE_CHANGE

**Definition:** A domain identifies the part of the system, organization, or context that is the target of the proposed change. Domains group proposed changes by which area should be changed.

**Key idea:** Domains specify *what part of the system* should change.

**Question it answers:** *What is the target of the change?*

## Facet (L3) within PRESCRIPTIVE_CHANGE

**Definition:** A facet identifies the analytical lens through which proposed changes are examined — specifically, the type or approach of change being proposed. Facets distinguish between different kinds of interventions aimed at the same target area; each must be independently analyzable.

**Key idea:** Facets specify *how* the target should change.

**Question it answers:** *What type of change is proposed?*

## Attribute (L4) within PRESCRIPTIVE_CHANGE

**Definition:** An attribute identifies the specific, concrete improvement or action being proposed. It is a named property that captures the precise nature of the suggestion.

**Key idea:** Attributes name the specific proposed improvement.

**Question it answers:** *What exactly is the proposed improvement?*

### Example

```
Response: "We need live chat and faster email replies"
Domain: Customer support
Facet: Response speed improvement
Attribute: Live chat availability
```

---

# 2 IDENTITY_DEFINITION

**Semantic variation:** statements define what something *is*

## Domain (L2) within IDENTITY_DEFINITION

**Definition:** A domain identifies the entity or concept whose identity is being defined or categorized. Domains group identity statements by what is being defined.

**Key idea:** Domains specify *what is being defined*.

**Question it answers:** *What is being defined?*

## Facet (L3) within IDENTITY_DEFINITION

**Definition:** A facet identifies the analytical lens through which an entity's identity is examined — specifically, the aspect of identity being addressed (purpose, scope, category, nature, or meaning). Each facet must be independently analyzable.

**Key idea:** Facets specify *which aspect of identity* is being defined.

**Question it answers:** *Which aspect of identity?*

## Attribute (L4) within IDENTITY_DEFINITION

**Definition:** An attribute identifies the specific defining characteristic or identity marker being articulated. It is a named property that captures the precise feature of the entity's identity.

**Key idea:** Attributes name the specific defining feature.

**Question it answers:** *What defining feature is mentioned?*

### Example

```
Response: "It's a place for professionals to network"
Domain: The platform
Facet: Purpose
Attribute: Professional networking function
```

---

# 3 ACTORS_TARGETS

**Semantic variation:** statements identify who is involved

## Domain (L2) within ACTORS_TARGETS

**Definition:** A domain identifies the subject or situation in which actors are involved. Domains group actor statements by the context they participate in.

**Key idea:** Domains specify *the subject or situation* in which actors are involved.

**Question it answers:** *In what situation are actors involved?*

## Facet (L3) within ACTORS_TARGETS

**Definition:** A facet identifies the analytical lens through which actor involvement is examined — specifically, the role or position the actor occupies (decision-maker, beneficiary, affected party, or responsible party). Each facet must be independently analyzable.

**Key idea:** Facets specify *what role* the actor plays.

**Question it answers:** *What role do they play?*

## Attribute (L4) within ACTORS_TARGETS

**Definition:** An attribute identifies the specific actor group or stakeholder type referenced. It is a named property that captures the precise party being discussed.

**Key idea:** Attributes name the specific actor or stakeholder group.

**Question it answers:** *Which specific actor group is referenced?*

### Example

```
Response: "Product managers should decide on feature priorities"
Domain: Decision-making
Facet: Responsibility
Attribute: Product management authority
```

---

# 4 CONTEXT_CONDITIONS

**Semantic variation:** statements specify when, where, or under what conditions

## Domain (L2) within CONTEXT_CONDITIONS

**Definition:** A domain identifies the situation, activity, or process to which contextual conditions apply. Domains group conditional statements by what situation is being discussed.

**Key idea:** Domains specify *what situation* is being discussed.

**Question it answers:** *What situation is being discussed?*

## Facet (L3) within CONTEXT_CONDITIONS

**Definition:** A facet identifies the analytical lens through which contextual conditions are examined — specifically, the type of contextual dimension (time, location, constraint, trigger, or environment). Each facet must be independently analyzable.

**Key idea:** Facets specify *what type of condition* is described.

**Question it answers:** *Time? location? constraint?*

## Attribute (L4) within CONTEXT_CONDITIONS

**Definition:** An attribute identifies the specific condition, circumstance, or contextual factor being mentioned. It is a named property that captures the precise situational feature.

**Key idea:** Attributes name the specific contextual condition.

**Question it answers:** *What specific condition is mentioned?*

### Example

```
Response: "It only works well during off-peak hours"
Domain: Product usage
Facet: Time
Attribute: Peak-hour performance dependency
```

---

# 5 MOTIVATIONS_DRIVERS

**Semantic variation:** statements express why people care or act

## Domain (L2) within MOTIVATIONS_DRIVERS

**Definition:** A domain identifies the object, activity, or situation that the motivation is about. Domains group motivational statements by what the motivation relates to.

**Key idea:** Domains specify *what the motivation is about*.

**Question it answers:** *What is the motivation about?*

## Facet (L3) within MOTIVATIONS_DRIVERS

**Definition:** A facet identifies the analytical lens through which motivations are examined — specifically, the type of motivation expressed (need, goal, fear, value, or aspiration). Each facet must be independently analyzable.

**Key idea:** Facets specify *what type of motivation* is expressed.

**Question it answers:** *Need? goal? fear? value?*

## Attribute (L4) within MOTIVATIONS_DRIVERS

**Definition:** An attribute identifies the specific reason, benefit, or motivational factor being stated. It is a named property that captures the precise driver.

**Key idea:** Attributes name the specific motivational factor.

**Question it answers:** *What specific reason is stated?*

### Example

```
Response: "I use it because it saves me time every morning"
Domain: Using the app
Facet: Convenience
Attribute: Time-saving benefit
```

---

# 6 EXPERIENCE_PERCEPTION

**Semantic variation:** statements describe lived experience

## Domain (L2) within EXPERIENCE_PERCEPTION

**Definition:** A domain identifies the part of the experience or journey being described. Domains group experiential statements by which aspect of the overall experience is discussed.

**Key idea:** Domains specify *which part of the experience* is described.

**Question it answers:** *Which part of the experience?*

## Facet (L3) within EXPERIENCE_PERCEPTION

**Definition:** A facet identifies the analytical lens through which experiences are examined — specifically, the experiential dimension being addressed (flow, atmosphere, interaction, sensation, or emotion). Each facet must be independently analyzable.

**Key idea:** Facets specify *what experiential quality* is described.

**Question it answers:** *Flow? atmosphere? interaction?*

## Attribute (L4) within EXPERIENCE_PERCEPTION

**Definition:** An attribute identifies the specific experiential feature observed or felt. It is a named property that captures the precise aspect of the experience.

**Key idea:** Attributes name the specific observed experience feature.

**Question it answers:** *What specific experience feature was observed or felt?*

### Example

```
Response: "The onboarding steps were confusing"
Domain: Onboarding
Facet: Flow
Attribute: Step clarity
```

---

# 7 EVALUATION_PRIORITIZATION

**Semantic variation:** statements express judgments or preferences

## Domain (L2) within EVALUATION_PRIORITIZATION

**Definition:** A domain identifies the object, service, or aspect being evaluated or judged. Domains group evaluative statements by what is being assessed.

**Key idea:** Domains specify *what is being evaluated*.

**Question it answers:** *What object is evaluated?*

## Facet (L3) within EVALUATION_PRIORITIZATION

**Definition:** A facet identifies the analytical lens through which evaluations are examined — specifically, the evaluation criterion being applied (speed, cost, quality, importance, or satisfaction). Each facet must be independently analyzable.

**Key idea:** Facets specify *on what criterion* the evaluation is based.

**Question it answers:** *Speed? cost? quality? importance?*

## Attribute (L4) within EVALUATION_PRIORITIZATION

**Definition:** An attribute identifies the specific evaluative signal or evidence of judgment. It is a named property that captures the precise characteristic being assessed.

**Key idea:** Attributes name the specific evaluation signal.

**Question it answers:** *What specific evidence of evaluation appears?*

### Example

```
Response: "Delivery was too slow last time"
Domain: Delivery
Facet: Speed
Attribute: Delivery timeliness
```

---

# 8 BEHAVIOR_FUNCTION

**Semantic variation:** statements describe what happens or how something works

## Domain (L2) within BEHAVIOR_FUNCTION

**Definition:** A domain identifies the process, system, or activity being described. Domains group behavioral statements by what system or process is discussed.

**Key idea:** Domains specify *what system or process* is described.

**Question it answers:** *What system or process?*

## Facet (L3) within BEHAVIOR_FUNCTION

**Definition:** A facet identifies the analytical lens through which behaviors or functions are examined — specifically, the functional stage or step (input, processing, output, or interaction). Each facet must be independently analyzable.

**Key idea:** Facets specify *which step or function* is described.

**Question it answers:** *Which step or function?*

## Attribute (L4) within BEHAVIOR_FUNCTION

**Definition:** An attribute identifies the specific action, behavior, or functional feature being described. It is a named property that captures the precise operational characteristic.

**Key idea:** Attributes name the specific behavioral or functional feature.

**Question it answers:** *What specific action occurs?*

### Example

```
Response: "The user enters their credit card at checkout"
Domain: Checkout process
Facet: Process stage
Attribute: Payment data entry
```

---

# 9 ATTRIBUTES_ASSOCIATIONS

**Semantic variation:** statements describe qualities, traits, or associations

## Domain (L2) within ATTRIBUTES_ASSOCIATIONS

**Definition:** A domain identifies the entity or object being described with qualities or associations. Domains group descriptive statements by what entity has the trait.

**Key idea:** Domains specify *what entity* has the trait.

**Question it answers:** *What entity has the trait?*

## Facet (L3) within ATTRIBUTES_ASSOCIATIONS

**Definition:** A facet identifies the analytical lens through which descriptive qualities are examined — specifically, the attribute category (visual, emotional, functional, or symbolic). Each facet must be independently analyzable.

**Key idea:** Facets specify *what type of quality* is described.

**Question it answers:** *Visual? emotional? functional?*

## Attribute (L4) within ATTRIBUTES_ASSOCIATIONS

**Definition:** An attribute identifies the specific quality, trait, or association being described. It is a named property that captures the precise descriptive characteristic.

**Key idea:** Attributes name the specific quality or trait.

**Question it answers:** *What specific quality is described?*

### Example

```
Response: "It feels very modern"
Domain: Brand
Facet: Personality
Attribute: Modernity perception
```

---

# 10 RELATIONS_DEPENDENCIES

**Semantic variation:** statements describe relationships between entities

## Domain (L2) within RELATIONS_DEPENDENCIES

**Definition:** A domain identifies the system or set of entities involved in the relationship. Domains group relational statements by what entities are connected.

**Key idea:** Domains specify *what entities are involved* in the relationship.

**Question it answers:** *What entities are involved?*

## Facet (L3) within RELATIONS_DEPENDENCIES

**Definition:** A facet identifies the analytical lens through which relationships are examined — specifically, the type of relationship (dependency, trade-off, influence, or comparison). Each facet must be independently analyzable.

**Key idea:** Facets specify *what type of relationship* is described.

**Question it answers:** *Dependency? trade-off? influence?*

## Attribute (L4) within RELATIONS_DEPENDENCIES

**Definition:** An attribute identifies the specific relational feature or linkage being described. It is a named property that captures the precise nature of the connection between entities.

**Key idea:** Attributes name the specific relational feature.

**Question it answers:** *What specific relationship is described?*

### Example

```
Response: "Going cheaper always means worse quality"
Domain: Pricing vs quality
Facet: Trade-off
Attribute: Price-quality correlation
```

---

# 11 GENERAL_OTHER

**Semantic variation:** statements do not clearly fit a specific dimension (fallback)

## Domain (L2) within GENERAL_OTHER

**Definition:** A domain identifies the general topic or subject area of the response. Because no specific dimension applies, domains simply group statements by what they are about.

**Key idea:** Domains specify *what topic* the response relates to.

**Question it answers:** *What is this about?*

## Facet (L3) within GENERAL_OTHER

**Definition:** A facet identifies the analytical lens through which general remarks are examined — specifically, the type of remark (uncertain response, meta-comment, general observation). Each facet must be independently analyzable.

**Key idea:** Facets specify *what type of general remark* this is.

**Question it answers:** *What type of remark is this?*

## Attribute (L4) within GENERAL_OTHER

**Definition:** An attribute identifies the specific feature or characteristic of the general remark. It is a named property that captures whatever concrete signal is present.

**Key idea:** Attributes name whatever specific feature can be identified.

**Question it answers:** *What specific feature is mentioned?*

### Example

```
Response: "I don't really know what to say about them"
Domain: General
Facet: Uncertain response
Attribute: Opinion absence
```

---

# Summary

This document defines dimension-specific interpretations for 11 dimensions (10 substantive + GENERAL_OTHER fallback). The taxonomy uses 4 qualitatively different analytical layers:

**Dimension (L1)** → **Domain (L2)** → **Facet (L3)** → **Attribute (L4)**

The core principle is that the **dimension acts as a meta-taxonomy**: once selected, the dimension determines how Domain, Facet, and Attribute should be interpreted for the analysis of the dataset. Each dimension answers a different question about the responses:

| Dimension | What it captures |
|-----------|-----------------|
| PRESCRIPTIVE_CHANGE_OUTCOME_ENABLERS | What should change? |
| IDENTITY_DEFINITION | What is it? |
| ACTORS_TARGETS | Who is involved? |
| CONTEXT_CONDITIONS | When/where does it apply? |
| MOTIVATIONS_DRIVERS | Why do people care? |
| EXPERIENCE_PERCEPTION | What was experienced? |
| EVALUATION_PRIORITIZATION | Is it good/bad? |
| BEHAVIOR_FUNCTION | What happens? |
| ATTRIBUTES_ASSOCIATIONS | What qualities exist? |
| RELATIONS_DEPENDENCIES | How do things interact? |
| GENERAL_OTHER | (fallback) |

The stable, dimension-independent definitions of Domain, Facet, and Attribute are defined in `taxonomy_logic.md`. This document extends those with dimension-specific meanings that adapt the interpretation to the type of information being analyzed.
