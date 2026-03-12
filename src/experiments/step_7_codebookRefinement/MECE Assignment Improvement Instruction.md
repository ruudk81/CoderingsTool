
# MECE Assignment Improvement Instruction

## 1. What is the issue

### Core problem

The current MECE assignment instructions enforce **conceptual exclusivity at the codebook level**, but **operational assignment happens at the idea level**, where multiple concepts can still be *contextually adjacent*.

As a result:

* Many ideas legitimately touch **two neighboring concepts** (e.g. *schaduw* as facility vs comfort).
* The instructions **forbid assignment to both**, instead of defining **which concept should dominate**.
* When no single code is explicitly “allowed,” the model defaults to **overig/anders** to avoid rule violations.

This causes:

* Overuse of *overig/anders* categories
* Artificial inflation of misc buckets
* Reduced analytical value, despite conceptual MECE compliance

**Important clarification**
This issue is **not caused by mixed sentiment or multi-idea responses**.
It occurs **even after responses are correctly split into atomic ideas**.

---

## 2. Why this happens (mechanism)

The instructions rely heavily on:

* *Mutual exclusion via hard exclusions*
* *Tell-apart rules that prohibit assignment if a neighboring concept is present*

This creates **assignment dead zones** where:

* An idea clearly fits the domain
* But is excluded from all specific codes
* Leaving only *overig/anders* as a safe option

In practice, this means the system optimizes for **instructional compliance**, not **semantic clarity**.

---

## 3. How to resolve (required changes)

### A. Replace exclusion logic with dominance logic

**Current pattern (problematic)**

> “Exclude this code if aspect X is also mentioned.”

**Replace with**

> “If multiple aspects are present, assign the code that represents the *primary experiential or causal focus* of the idea.”

This ensures:

* One clear assignment per idea
* No forced fallback to overig
* MECE preserved at the output level

---

### B. Define primary assignment axes explicitly

For each theme cluster, define **one dominant axis**, e.g.:

| Cluster                 | Primary axis                               |
| ----------------------- | ------------------------------------------ |
| Faciliteiten vs comfort | Physical provision vs experienced effect   |
| Muziek vs sfeer         | Program content vs emotional outcome       |
| Omgeving vs sfeer       | External conditions vs internal experience |

Assignment must follow the **primary axis**, not surface keywords.

---

### C. Narrow the use of “overig/anders”

Add a **hard constraint**:

> Use *overig/anders* **only if the idea does not reference**
>
> * a concrete object
> * a specific experience
> * an identifiable actor
> * a describable condition

If any of the above are present, a specific code **must** be chosen.

---

## 4. Concrete examples

### Example 1 — Schaduw & hitte

**Idea**

> “Door de hitte was het fijn dat er schaduwplekken waren.”

**Current outcome**

* Excluded from *Goede schaduwvoorziening* (mentions heat)
* Excluded from *Comfort door verkoeling* (mentions facility)
* Assigned to *overig/anders* ❌

**Corrected rule application**

* Primary focus = *comfort during heat*
* Facility is a **means**, not the focus

**Correct assignment**
✅ *Comfort door verkoeling en hydratatie (+)*

---

### Example 2 — Muziek en sfeer

**Idea**

> “De optredens zorgden voor een geweldige sfeer.”

**Current outcome**

* Ambiguous between *Goede live optredens* and *Sfeerbeleving*
* Risk of exclusion → overig ❌

**Corrected rule application**

* Primary axis = *cause of experience*
* Music is the driver

**Correct assignment**
✅ *Goede live optredens (+)*
(not sfeer)

---

### Example 3 — Publiek en sfeer

**Idea**

> “Het publiek maakte het festival extra gezellig.”

**Correct logic**

* Actor = publiek
* Effect = sfeer
* Primary axis = *social contributor*

**Correct assignment**
✅ *Sfeerverhogend publiek (+)*
Not *Sfeerbeleving* or *algemene omgevingsbeleving*

---

### Example 4 — When overig *is* allowed

**Idea**

> “Het geheel klopte gewoon.”

* No object
* No actor
* No concrete experience
* No causal explanation

**Correct assignment**
✅ *overige redenen — overig/anders (0)*

---

## 5. Final enforcement rule (recommended to add verbatim)

> **If an idea can be assigned to a specific code by identifying a dominant experiential, causal, or functional focus, it must not be assigned to an overig/anders category. Overig/anders is only permitted when no such dominant focus can be identified.**

---

If you want, next I can:

* Rewrite **one full theme** using dominance-based rules
* Or provide a **short classifier checklist** the model can apply before using *overig/anders*
