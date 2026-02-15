TEMPLATE_LOOKUP = {
    "marker": "[ACTIONABLE_TAXONOMY_DIMENSION]",

    # =========================================================
    # Layer 1 - Primary axis focus / organizing framework
    # =========================================================
    "axes": {

        "WHAT": {
            "anchor": "entity_phenomenon",
            "noun_phrase_descriptor": "WHAT: the entity or phenomenon described",

            "dimension_description": (
                "The survey question concerns what exists or what has occurred.\n"
                "It typically refers to an entity or phenomenon such as a brand, organization, product, service, event, or experience.\n"
                "Variations in responses usually reflect differences in specific WHAT concepts, including attributes, features, properties, or aspects of the focal entity."
            ),
  
            "allowed_concepts": ["attribute", "feature", "property", "aspect", "capability", "specification", "structure"],
            "excluded_concepts": ["action", "intervention", "pathway", "reason", "motive", "stakeholder", "timing", "location", "improvement", "goal", "method"],

            "schema": "DESCRIBE_ENTITY_DESCRIPTOR",

            "instruction": (
                "Identify each distinct WHAT idea expressed in the response. "
                "For each idea, produce one concise descriptive realization of a WHAT concept, formatted according to the pattern."
            ),

            "pattern": "[ANCHOR_SUBJECT] → [ENTITY_PHENOMENON_DESCRIPTOR]",

            "slot_guidance": {
                "[ANCHOR_SUBJECT]": 
                    "The focal entity or phenomenon in {language}.",
                "[ENTITY_PHENOMENON_DESCRIPTOR]": 
                    "A concise phrase describing a WHAT concept of the entity or phenomenon."
            },

            "prompt_rules": {
                "instance_instruction": "short verbatim or descriptive phrase",
                "node_instruction": "short noun like phrase or label",
                "category_instruction": "short noun like phrase or label",
                "root_instruction": "short noun like phrase or label"
            }

        },

        "WHO": {
            "anchor": "target_actor",
            "noun_phrase_descriptor": "WHO: the target actor described",

            "dimension_description": (
                "The survey question is about who did what, who is involved, who is affected, or who is responsible.\n"
                "Differences in responses are mainly about describing a target actor, group, identity, or role, including target audiences, stakeholders, and beneficiaries."
            ),

            "allowed_concepts": ["stakeholder", "role", "group", "beneficiary", "affected_party", "responsible_party", "user_segment", "decision_maker"],
            "excluded_concepts": ["action", "reason_driver", "timing", "location", "product_feature", "method", "motivation", "attribute"],

            "schema": "RELATE_ACTOR_TARGET",

            "instruction": (
                "Identify each distinct WHO idea expressed in the response. "
                "For each idea, produce one concise descriptive realization of the WHO concept (e.g. actor, group, role, or identity," 
                "formatted according to the pattern."
            ),

            "pattern": "[ANCHOR_SUBJECT] → [ACTOR_TARGET]",

            "slot_guidance": {
                "[ANCHOR_SUBJECT]": 
                    "The focal entity, service, or topic that is related to an actor in {language}.",
                "[ACTOR_TARGET]": 
                    "A concise descriptive phrase that names and/or characterizes the relevant WHO target (e.g. actor, group, role, or identity)."
            },
            "prompt_rules": {
                "instance_instruction": "Select the minimal contiguous verbatim span that expresses or references exactly one taxonomy-instance (as extracted in Step 2).",
                "node_instruction": "Create a canonical, reusable label for the specific {taxonomy_actionable_type]-instance in light of {topic}. Generalize beyond the response wording; do not mix in other axes.",
                "category_instruction": "Assign a broader, stable taxonomy concept that the node belongs to, interpreted in relation to the topic. It must be more general than the node and reusable across responses.",
                "root_instruction": "Choose the primary topic aspect along the taxonomy axis that this category fundamentally concerns in light of the topic. Root names a part of the topic (not a more abstract axis concept) and should be stable across responses."
            }
        },

        "WHERE": {
            "anchor": "location_context",
            "noun_phrase_descriptor": "WHERE: the location, context, or setting described",

            "dimension_description": (
                "The survey question is about where things happen or in what setting they occur.\n"
                "Differences in responses are mainly about describing context, setting, or location (physical, institutional, or digital)."
            ),

            "allowed_concepts": ["location", "touchpoint", "channel", "setting", "context", "platform", "market_context", "situation"],
            "excluded_concepts": ["action", "reason_driver", "stakeholder", "timing", "product_feature", "method", "motivation", "attribute"],

            "schema": "LOCATE_LOCATION_CONTEXT",

            "instruction": (
                "Identify each distinct WHERE idea expressed in the response. "
                "For each idea, produce one concise descriptive realization of the WHERE concept (e.g. context, setting, location, or touchpoint)," 
                "formatted according to the pattern."
            ),

            "pattern": "[ANCHOR_SUBJECT] @ [LOCATION_CONTEXT]",

            "slot_guidance": {
                "[ANCHOR_SUBJECT]": 
                    "A neutral event, experience, or topic frame (noun phrase) that is consistent with the survey question in {language}.",
                "[LOCATION_CONTEXT]": 
                    "A concise descriptive phrase specifying the relevant WHERE concept (e.g. context, setting, location, or touchpoint)."
            },
            "prompt_rules": {
                "instance_instruction": "Select the minimal contiguous verbatim span that expresses or references exactly one taxonomy-instance (as extracted in Step 2).",
                "node_instruction": "Create a canonical, reusable label for the specific {taxonomy_actionable_type]-instance in light of {topic}. Generalize beyond the response wording; do not mix in other axes.",
                "category_instruction": "Assign a broader, stable taxonomy concept that the node belongs to, interpreted in relation to the topic. It must be more general than the node and reusable across responses.",
                "root_instruction": "Choose the primary topic aspect along the taxonomy axis that this category fundamentally concerns in light of the topic. Root names a part of the topic (not a more abstract axis concept) and should be stable across responses."
            }
        },

        "WHEN": {
            "anchor": "time_urgency",
            "noun_phrase_descriptor": "WHEN: a temporal pattern, timing, urgency, or frequency",

            "dimension_description": (
                "The survey question is about when things happened or how often they occur.\n"
                "Differences in responses are mainly about describing temporality, timing, urgency, or frequency."
            ),

            "allowed_concepts": ["time", "urgency", "frequency", "sequence", "availability_window", "lifecycle_stage", "deadline"],
            "excluded_concepts": ["action", "reason_driver", "stakeholder", "location", "product_feature", "method", "motivation", "attribute"],

            "schema": "LOCATE_TIME_URGENCY",

            "instruction": (
                "Identify each distinct WHEN idea expressed in the response. "
                "For each idea, produce one concise descriptive realization of the WHEN concept (e.g. timing, frequency, sequence, or urgency)," 
                "formatted according to the pattern."
            ),

            "pattern": "[ANCHOR_SUBJECT] @ [TIME_URGENCY]",

            "slot_guidance": {
                "[ANCHOR_SUBJECT]": 
                    "A neutral event, need, or topic frame (noun phrase) that is consistent with the survey question in {language}.",
                "[TIME_URGENCY]": 
                    "A concise descriptive phrase expressing the WHEN concept (e.g. timing, frequency, sequence, or urgency)"
            },
            "prompt_rules": {
                "instance_instruction": "short verbatim or descriptive phrase",
                "node_instruction": "short noun like phrase or label",
                "category_instruction": "short noun like phrase or label",
                "root_instruction": "short noun like phrase or label"
            }
        },

        "HOW": {
            "anchor": "outcome_enabler",
            "noun_phrase_descriptor": "HOW: an outcome-enabling mechanism or pathway",

            "dimension_description": (
                "The survey question is about how outcomes are achieved, how things unfold, or how people act, cope, or solve problems.\n"
                "Differences in responses are mainly about:\n"
                "  A) Outcome enablers: interventions, changes, mechanisms that enable an outcome;\n"
                "  B) Execution pathways: steps, procedures, workflows, or process descriptions."
            ),

            "allowed_concepts": [
                "prescription",
                "recommendation",
                "intervention",
                "action_mechanism",
                "execution_pathway",
                "procedure",
                "workflow_step",
                "tool",
                "tactic",
                "other_outcome_enabler"
            ],
            "excluded_concepts": ["static_attribute", "reason_driver", "stakeholder_identity", "timing", "location", "current_attribute"],

            "schema": "EXPRESS_OUTCOME_ENABLER_DESCRIPTIVE",

            "instruction": (
                "Identify each distinct HOW idea expressed in the response.\n"
                "For each idea, produce one concise descriptive realization of the HOW concept (e.g. recommendation, intervention, execution pathway, action mechanism or procedure), "
                "formatted according to the pattern. \n"
                "Use prescriptive wording only if the survey question explicitly asks for recommendations."
            ),

            "pattern": "[OUTCOME] → [OUTCOME_ENABLER]",

            "slot_guidance": {
                "[OUTCOME]": 
                    "A concise noun phrase naming the intended or observed outcome, result, or state of affairs.",
                "[OUTCOME_ENABLER]": 
                    "A concise phrase specifying the the HOW concept (e.g. recommendation, intervention, execution pathway, action mechanism or procedure)."
            },
            "prompt_rules": {
                "instance_instruction": "Select the minimal contiguous verbatim span that expresses or references exactly one taxonomy-instance (as extracted in Step 2).",
                "node_instruction": "Create a canonical, reusable label for the specific {taxonomy_actionable_type]-instance in light of {topic}. Generalize beyond the response wording; do not mix in other axes.",
                "category_instruction": "Assign a broader, stable taxonomy concept that the node belongs to, interpreted in relation to the topic. It must be more general than the node and reusable across responses.",
                "root_instruction": "Choose the primary topic aspect along the taxonomy axis that this category fundamentally concerns in light of the topic. Root names a part of the topic (not a more abstract axis concept) and should be stable across responses."
            }
        },

        "WHY": {
            "anchor": "reason_driver",
            "noun_phrase_descriptor": "WHY: reasons, causes, and explanations",

            "dimension_description": (
                "The survey question is about why people prefer one thing over another, why things happened, or why people acted as they did.\n"
                "Differences in responses are mainly about reasons, causes, and explanations."
            ),

            "allowed_concepts": ["reason", "motivation", "concern", "constraint", "goal", "tradeoff_driver", "intention", "desired_outcome", "justification", "other_reason"],
            "excluded_concepts": ["product_feature", "intervention", "process_step", "stakeholder", "timing", "location", "method", "attribute"],

            "schema": "EXPRESS_REASON_DRIVER",

            "instruction": (
                "Identify each distinct WHY idea expressed in the response. "
                "For each idea, represent it as a cause or reason linked to the explained effect (e.g outcome, preference, choice, or behavior), "
                "formatted according to the pattern."
            ),

            # Canonical direction for WHY
            "pattern": "[CAUSE] → [EFFECT]",

            "slot_guidance": {
                "[CAUSE]": 
                    "A concise phrase specifying the reason, driver, or constraint that explains something (WHY).",
                "[EFFECT]": 
                    "A concise noun phrase naming the preference, choice, behavior, or outcome being explained."
            },
            "prompt_rules": {
                "instance_instruction": "Select the minimal contiguous verbatim span that expresses or references exactly one taxonomy-instance (as extracted in Step 2).",
                "node_instruction": "Create a canonical, reusable label for the specific {taxonomy_actionable_type]-instance in light of {topic}. Generalize beyond the response wording; do not mix in other axes.",
                "category_instruction": "Assign a broader, stable taxonomy concept that the node belongs to, interpreted in relation to the topic. It must be more general than the node and reusable across responses.",
                "root_instruction": "Choose the primary topic aspect along the taxonomy axis that this category fundamentally concerns in light of the topic. Root names a part of the topic (not a more abstract axis concept) and should be stable across responses."
            }
        }
    },

    # =========================================================
    # Layer 2 — Template structure
    # =========================================================
    "type_system": {
        "aliases": {
            "noun_like_phrase": [
                "noun_phrase",
                "gerund_nominal",
                "compound_noun",
                "nominalized_adjective",
                "quality_of_construction"
            ]
        },

        "definitions": {
            "noun_phrase": "A noun phrase (1–8 tokens) naming an entity/aspect (e.g., 'appointment availability').",
            "gerund_nominal": "A nominalized -ing form used as a noun (e.g., 'overcrowding').",
            "compound_noun": "A compound noun or noun+modifier (e.g., 'wait time variability').",
            "nominalized_adjective": "An adjective-like property expressed as a noun-ish label (e.g., 'uneven quality').",
            "quality_of_construction": "A 'quality of X' phrase (e.g., 'quality of communication')."
        }
    },
                
            
    "template_schemas": {
        # Notes:
        # - Each schema is intended to bind to exactly ONE axis (MECE guardrail).
        # - Slot text must fit naturally into the schema pattern.

        # WHAT — entity_descriptor (descriptive only: properties/attributes/aspects AS-IS)
        "DESCRIBE_ENTITY_DESCRIPTOR": {
            "slots": {
                "ANCHOR_SUBJECT": {"required": True, "type": "noun_phrase"},
                "ENTITY_DESCRIPTOR":{ "required": True, "type": "noun_like_phrase" },
            },
            "structural_forms": [
                "<anchor> → <descriptor>",
                "<descriptor> of <anchor>"
            ],
            "notes": [
                "Descriptor must be WHAT-only (attribute/feature/aspect).",
                "Descriptor must be noun-like; avoid actions, causes, actors, times, places."
            ]
        },

        # WHY — reason_driver (motivations/goals/values/concerns/trade-offs)
        "EXPRESS_REASON_DRIVER": {
            "slots": {
                "CAUSE": {"required": True, "type": "noun_phrase"},
                "EFFECT": {"required": True, "type": "noun_like_phrase"}
            },
            "structural_forms": [
                "<cause> → <effect>",
                "<effect> because of <cause>"
            ],
            "notes": [
                "Do not invert directionality unless the prompt explicitly asks for effect-to-cause.",
                "Avoid methods/interventions here; those belong in HOW."
            ]
        },

        # HOW — descriptive outcome-enabler
        "EXPRESS_OUTCOME_ENABLER_DESCRIPTIVE": {
            "slots": {
                "OUTCOME": {"required": True, "type": "noun_phrase"},
                "OUTCOME_ENABLER": {"required": True, "type": "noun_like_phrase"}
            },
            "structural_forms": [
                "<outcome> → <enabler>",
                "<outcome> is achieved by <enabler>"
            ],
            "notes": [
                "Avoid modals (should/could/must) unless the question explicitly asks for recommendations.",
            ]
        },

        # WHO — actor_target
        "RELATE_ACTOR_TARGET": {
            "slots": {
                "ANCHOR_SUBJECT": {"required": True, "type": "noun_phrase"},
                "ACTOR_TARGET": {"required": True, "type": "noun_like_phrase"}
            },
            "structural_forms": [
                "<anchor> → <actor>",
                "<service> is for <actor>"
            ],
            "notes": [
                "ACTOR_TARGET must be a person/group/role; do not encode motivations, methods, time, or place."
            ]
        },

        # WHEN — time_urgency
        "LOCATE_TIME_URGENCY": {
            "slots": {
                "ANCHOR_SUBJECT": {"required": True, "type": "noun_phrase"},
                "TIME_URGENCY": {"required": True, "type": "noun_like_phrase"}
            },
            "structural_forms": [
                "<anchor> @ <time>",
                "<issue> occurs during <time>"
            ],
            "notes": [
                "TIME_URGENCY should express timing/frequency/sequence/urgency as a noun_like_phrase."
            ]
        },

        # WHERE — location_context
        "LOCATE_LOCATION_CONTEXT": {
            "slots": {
                "ANCHOR_SUBJECT": {"required": True, "type": "noun_phrase"},
                "LOCATION_CONTEXT": {"required": True, "type": "noun_like_phrase"}
            },
            "structural_forms": [
               "<anchor> @ <place/context>",
              "<experience> takes place in <place/context>"
            ],
            "notes": [
                "LOCATION_CONTEXT should be physical, institutional, or digital; keep it as a noun_like_phrase."
            ]
        }
    },

    # =========================================================
    # Layer 3 — Dimension-conditioned semantic taxonomy
    # =========================================================
    # Universal classification scheme reinterpreted per coding dimension.
    # Prevents dimensional leakage by providing structured categories
    # with dimension-specific interpretations.

    "dimension_taxonomy": {
        "priority_rules": [
            "If subjective judgment → evaluation",
            "If time-bound or situational → state",
            "If action/purpose → function",
            "If inherent property → attribute",
            "If category/type → identity"
        ],

        "dimensions": {

            "WHAT": {
                "axis_interpretation": {
                    "identity": "What it is",
                    "attribute": "What it has",
                    "function": "What it does",
                    "state": "Condition it is in",
                    "evaluation": "Judgment about it",
                    "relation": "How it connects to others"
                },
                "decision_reminder": [
                    "If subjective judgment → evaluation",
                    "If time-bound or situational → state",
                    "If action/purpose → function",
                    "If inherent property → attribute",
                    "If category/type → identity"
                ]
            },

            "WHY": {
                "axis_interpretation": {
                    "identity": "The type of goal or value being pursued",
                    "attribute": "An enduring priority or value orientation",
                    "function": "The purpose the entity serves for the actor",
                    "state": "A temporary concern or situational pressure",
                    "evaluation": "A value judgment driving preference",
                    "relation": "Trade-offs or competing motivations"
                },
                "decision_reminder": [
                    "If subjective judgment → evaluation",
                    "If temporary pressure → state",
                    "If purpose → function",
                    "If enduring value orientation → attribute",
                    "If type of goal → identity"
                ]
            },

            "HOW": {
                "axis_interpretation": {
                    "identity": "Type of mechanism or approach",
                    "attribute": "Characteristic of the method (e.g., scalable, automated)",
                    "function": "What the method accomplishes",
                    "state": "Stage in a process",
                    "evaluation": "Judgment of effectiveness",
                    "relation": "Dependency between steps"
                },
                "decision_reminder": [
                    "If it describes how to get from current to desired state → function",
                    "If it describes what currently exists → identity",
                    "If subjective judgment of method → evaluation",
                    "If characteristic of method → attribute",
                    "If process dependency → relation"
                ]
            },

            "WHO": {
                "axis_interpretation": {
                    "identity": "Actor type or role",
                    "attribute": "Characteristic of the actor",
                    "function": "Role the actor performs",
                    "state": "Temporary role or involvement",
                    "evaluation": "Judgment of actor",
                    "relation": "Power or dependency relationship"
                },
                "decision_reminder": [
                    "If actor type or category → identity",
                    "If inherent characteristic of actor → attribute",
                    "If role/function performed → function",
                    "If temporary involvement → state",
                    "If judgment of actor → evaluation",
                    "If power/dependency relationship → relation"
                ]
            },

            "WHEN": {
                "axis_interpretation": {
                    "identity": "Type of lifecycle stage",
                    "attribute": "Typical duration",
                    "function": "Temporal role (e.g., trigger phase)",
                    "state": "Current time-bound condition",
                    "evaluation": "Urgency assessment",
                    "relation": "Sequence dependency"
                },
                "decision_reminder": [
                    "If type of time period or stage → identity",
                    "If typical duration → attribute",
                    "If temporal trigger or role → function",
                    "If current time-bound condition → state",
                    "If urgency judgment → evaluation",
                    "If sequence dependency → relation"
                ]
            },

            "WHERE": {
                "axis_interpretation": {
                    "identity": "Type of setting",
                    "attribute": "Characteristic of environment",
                    "function": "Role environment plays",
                    "state": "Temporary situational context",
                    "evaluation": "Judgment of context",
                    "relation": "Contextual dependency"
                },
                "decision_reminder": [
                    "If type of setting → identity",
                    "If characteristic of environment → attribute",
                    "If environmental role → function",
                    "If temporary situation → state",
                    "If judgment of context → evaluation",
                    "If contextual dependency → relation"
                ]
            }
        }
    }
}
