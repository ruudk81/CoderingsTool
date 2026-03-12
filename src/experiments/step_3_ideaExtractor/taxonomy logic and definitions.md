When designing a taxonomy like Dimension → Domain → Facet → Attribute, the best approach is to define each level by its role in abstraction and the question it answers. This keeps the taxonomy interpretable for both humans and LLMs.
Below is a clean set of definitions commonly used in information architecture, ontology design, and research frameworks.

1. Dimension
Definition
A dimension is the highest-level conceptual axis used to organize a problem space.It represents a major perspective or evaluation axis.
Key idea: the broad lens through which something is analyzed.
Question it answers
What overarching concept are we examining?
Examples
Customer Experience
Product Experience
Trust
Adoption
Usability
Example in survey taxonomy
Dimension: Product Experience

2. Domain
Definition
A domain is a major sub-area within a dimension that groups related phenomena.
It defines the main functional areas that belong to the dimension.
Question it answers
Which area of this dimension does the observation belong to?
Examples
Dimension: Product Experience

Domains:
- Usability
- Performance
- Design
- Reliability

3. Facet
Definition
A facet is a specific viewpoint or characteristic within a domain that describes how the domain can vary or be evaluated.
Facets describe different aspects of a domain's behavior or perception.
Question it answers
Which specific aspect of this domain are we focusing on?
Example
Domain: Usability

Facets:
- Navigation
- Learnability
- Interface clarity
- Efficiency

4. Attribute
Definition
An attribute is a concrete property or observable feature that describes the facet.
Attributes are the lowest taxonomy level before codes or measurements.
Question it answers
What specific characteristic or signal are we observing?
Example
Facet: Navigation

Attributes:
- Menu clarity
- Ease of finding features
- Breadcrumb visibility
- Logical page hierarchy

Full example taxonomy
Dimension: Product Experience

    Domain: Usability

        Facet: Navigation
            Attribute: Menu clarity
            Attribute: Ease of finding features
            Attribute: Page hierarchy clarity

        Facet: Learnability
            Attribute: Ease of first-time use
            Attribute: Need for guidance

    Domain: Performance

        Facet: Speed
            Attribute: Page load time
            Attribute: Response latency

How abstraction decreases
Dimension
   ↓
Domain
   ↓
Facet
   ↓
Attribute
Conceptual scope shrinks at each level:
Product Experience
   ↓
Usability
   ↓
Navigation
   ↓
Menu clarity

Mental model
Think of it like zoom levels:
Level	Zoom level
Dimension	Strategic concept
Domain	Major subfield
Facet	Specific aspect
Attribute	Observable property
One important rule for taxonomy quality
Each level should have clear semantic separation.
Bad example:
Domain: Usability
Facet: Ease of use
Attribute: Easy interface
These overlap too much.
Better:
Domain: Usability
Facet: Navigation
Attribute: Menu clarity

Why this structure works well with LLMs
LLMs perform better when classification steps follow progressive narrowing:
Step 1 → choose Dimension
Step 2 → choose Domain
Step 3 → choose Facet
Step 4 → identify Attribute
This reduces confusion and improves classification consistency.

✅ Short definitions you can use in documentation
Dimension
The highest-level analytical axis organizing the taxonomy.

Domain
A major conceptual area within a dimension.

Facet
A specific aspect or viewpoint of a domain.

Attribute
A concrete characteristic describing a facet.

If you want, I can also show you a very common taxonomy mistake in LLM annotation systems (Dimension–Domain collapse) and how to avoid it — it saves a lot of debugging later.
