TEMPLATE_LOOKUP = {
    "marker": "[ACTIONABLE_TAXONOMY_DIMENSION]",
    "axes": {
        "WHAT": {
            "dimension_description": (
                "entity_descriptor — attributes, features, characteristics, properties, or aspects"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language} (e.g., 'is/are/has')",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen entity_descriptor (attributes/features/etc.)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the stable FEATURE / ATTRIBUTE / PROPERTY being referenced (WHAT). "
                    "Use a reusable noun phrase naming the attribute (e.g., 'salt content', 'portion size', "
                    "'packaging separation'). Avoid embedding reasons, actions, actors, timing, or location."
                ),
                "category_instruction": (
                    "Category MUST be a stable parent grouping for features (WHAT), suitable for clustering many nodes, "
                    "e.g., 'ingredients', 'taste', 'texture', 'portioning', 'packaging', 'variety', 'price'. "
                    "Choose the most domain-relevant grouping."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the feature/attribute itself (WHAT) "
                    "(1–3 words preferred). Avoid verbs and generic meta-nouns."
                ),
                "focus_rules": [
                    "Instance must be the shortest verbatim span that still identifies the feature.",
                    "Prefer attribute nouns over full clauses."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_STATE": {"required": True, "allowed": ["is", "are", "has"]},
                    "SCAFFOLD": {"required": False, "allowed": ["with", "about", "for", "of"]}
                },
                "examples": [
                    "The meal is [taste].",
                    "The packaging has [recyclability]."
                ]
            },
        },

        "WHY": {
            "dimension_description": (
                "reason_driver — the underlying reason, motivation, concern, constraint, or goal "
                "that explains a preference/behavior/response (not a feature, process step, or outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen reason_driver (motivation/constraint/goal/etc.)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the underlying REASON / MOTIVATION / CONCERN / CONSTRAINT / GOAL (WHY). "
                    "Use a reusable noun phrase naming the driver (e.g., 'health concern', 'time saving', "
                    "'budget constraint', 'safety concern'). Do NOT encode actions, product features, actors, "
                    "timing, or location as the node."
                ),
                "category_instruction": (
                    "Category MUST be a stable parent grouping for reasons (WHY), suitable for clustering many nodes, "
                    "e.g., 'health goals', 'cost constraints', 'convenience goals', 'quality concerns', "
                    "'ethical concerns', 'risk/safety concerns'."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the reason/driver (WHY) "
                    "(1–4 words preferred). Avoid verbs and meta-language about 'opinion/perception'."
                ),
                "focus_rules": [
                    "Instance must be the shortest verbatim span that still expresses the reason.",
                    "Prefer the underlying driver over surface justifications."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_STATE} {CAUSE_PREP} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_STATE": {"required": True, "allowed": ["is", "comes", "exists"]},
                    "CAUSE_PREP": {"required": True, "allowed": ["because of", "due to", "on account of"]}
                },
                "examples": [
                    "Dissatisfaction comes because of [price].",
                    "The choice is due to [health concerns]."
                ]
            },
        },

        "HOW": {
            "dimension_description": (
                "Differences are about how an outcome would be achieved or carried out, including:"
                "A) Change-enabling mechanisms: actions, changes, interventions, tools, or mechanisms that make the outcome possible; "
                "B) Execution pathways: steps, processes, workflows, procedures, or ways of carrying something out. "
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen outcome enabler (recommendations, tactics, methods, etc.)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the TYPE OF ACTION / CHANGE / MECHANISM (HOW). "
                    "Use a reusable noun phrase describing the intervention type, preferably as a nominalized change "
                    "(e.g., 'reduction of X', 'increase of X', 'adjustment of X', 'separation of X', "
                    "'enrichment with X', 'avoidance of X'). "
                    "Do NOT use only the affected feature name as the node."
                ),
                "category_instruction": (
                    "Category MUST be a stable broader class of interventions (HOW), suitable for clustering many nodes, "
                    "e.g., 'ingredient changes', 'recipe adjustments', 'portion size changes', 'packaging redesign', "
                    "'process changes', 'labeling/information changes'."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the main lever/intervention focus (HOW) "
                    "(1–4 words preferred). Avoid verbs and avoid generic meta-nouns like 'improvement', "
                    "'optimization', 'adjustment' when they are not specific."
                ),
                "focus_rules": [
                    "Instance must be the shortest verbatim span that still captures the core recommendation/action.",
                    "If the response is only a noun (e.g., 'Salt.'), keep INSTANCE verbatim but encode an actionable HOW in NODE via a nominalized change."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_DIRECTIVE} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_DIRECTIVE": {"required": True, "allowed": ["must", "must not", "should"]}
                },
                "examples": [
                    "The product must [increase portion sizes].",
                    "The company must not [use too much salt]."
                ]
            },
        },

        "WHO": {
            "dimension_description": (
                "actor_target — the people, groups, roles, stakeholders, or beneficiaries involved/affected/"
                "responsible/addressed (not an action, not a reason, not an outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen actor_target (people/groups/roles/beneficiaries)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the ACTOR / GROUP / ROLE / BENEFICIARY (WHO). "
                    "Use a reusable noun phrase naming the stakeholder (e.g., 'children', 'busy professionals', "
                    "'elderly consumers', 'vegetarians'). Do NOT encode actions, reasons, timing, or location."
                ),
                "category_instruction": (
                    "Category MUST be a stable parent grouping for stakeholders (WHO), suitable for clustering many nodes, "
                    "e.g., 'life-stage groups', 'dietary groups', 'health-related groups', 'usage-context groups', "
                    "'accessibility-needs groups'."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the stakeholder group (WHO) "
                    "(1–4 words preferred). Avoid verbs and avoid abstract labels that do not denote people/groups."
                ),
                "focus_rules": [
                    "Instance should be the shortest verbatim span that identifies the stakeholder.",
                    "Prefer explicit groups over implied audiences unless clearly stated."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_STATE} {WHO_PREP} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_STATE": {"required": True, "allowed": ["is made", "is intended"]},
                    "WHO_PREP": {"required": True, "allowed": ["by", "for"]}
                },
                "examples": [
                    "The decision is made by [the head chef].",
                    "The meal is intended for [older adults]."
                ]
            },
        },

        "WHEN": {
            "dimension_description": (
                "time_urgency — timing, urgency, sequence, or frequency associated with when something occurs "
                "or is expected (not actions, reasons, or outcomes)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen time_urgency (timing/urgency/frequency/sequence)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the TIMING / URGENCY / FREQUENCY / SEQUENCE aspect (WHEN). "
                    "Use a reusable noun phrase like 'daily frequency', 'peak-time availability', "
                    "'immediate need', 'regular rotation'. Do NOT encode actions, reasons, actors, or locations."
                ),
                "category_instruction": (
                    "Category MUST be a stable parent grouping for timing concepts (WHEN), suitable for clustering many nodes, "
                    "e.g., 'frequency', 'urgency', 'seasonality', 'sequence/ordering', 'availability windows'."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the timing/urgency concept (WHEN) "
                    "(1–4 words preferred). Avoid verbs."
                ),
                "focus_rules": [
                    "Instance should be the shortest verbatim span that contains the timing cue.",
                    "Prefer explicit temporal expressions over inferred ones."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_EVENT} {TIME_PREP} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_EVENT": {"required": True, "allowed": ["occurs", "happens"]},
                    "TIME_PREP": {"required": True, "allowed": ["during", "at", "after", "before", "in"]}
                },
                "examples": [
                    "The problem occurs during [reheating].",
                    "Delivery happens in [the evening]."
                ]
            },
        },

        "WHERE": {
            "dimension_description": (
                "location_context — the physical/digital location, channel, setting, or situational context "
                "in which something occurs/is encountered (not an action, not a reason, not an outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen location_context (place/channel/setting/context)"
                ),
            },
            "prompt_rules": {
                "node_instruction": (
                    "Node MUST encode the LOCATION / CHANNEL / SETTING / CONTEXT (WHERE). "
                    "Use a reusable noun phrase like 'in-store availability', 'online channel', "
                    "'at-home use', 'workplace setting'. Do NOT encode actions, reasons, actors, or timing."
                ),
                "category_instruction": (
                    "Category MUST be a stable parent grouping for contexts (WHERE), suitable for clustering many nodes, "
                    "e.g., 'purchase channels', 'consumption settings', 'service touchpoints', 'digital contexts'."
                ),
                "taxonomy_phrase_instruction": (
                    "Taxonomy_phrase should be a short noun phrase naming the location/context (WHERE) "
                    "(1–4 words preferred). Avoid verbs."
                ),
                "focus_rules": [
                    "Instance should be the shortest verbatim span that contains the context cue.",
                    "Prefer concrete contexts over abstract 'situations'."
                ],
            },
            "template_structure": {
                "pattern": "{SUBJECT} {VERB_EVENT} {PLACE_PREP} [ACTIONABLE_TAXONOMY_DIMENSION].",
                "slots": {
                    "SUBJECT": {"required": True, "type": "noun_phrase"},
                    "VERB_EVENT": {"required": True, "allowed": ["happens", "takes place"]},
                    "PLACE_PREP": {"required": True, "allowed": ["in", "on", "at", "via"]}
                },
                "examples": [
                    "The purchase happens via [the online store].",
                    "Consumption takes place at [home]."
                ]
            },
        },
    },
}





TEMPLATE_LOOKUP_BACKUP = {
    "marker": "[ACTIONABLE_TAXONOMY_DIMENSION]",
    "axes": {
        "WHAT": {
            "dimension_description": (
                "entity_descriptor — attributes, features, characteristics, properties, or aspects"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language} (e.g., 'is/are/has')",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen entity_descriptor (attributes/features/etc.)"
                ),
            },
        },

        "WHY": {
            "dimension_description": (
                "reason_driver — the underlying reason, motivation, concern, constraint, or goal "
                "that explains a preference/behavior/response (not a feature, process step, or outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen reason_driver (motivation/constraint/goal/etc.)"
                ),
            },
        },

        "HOW": {
            "dimension_description": ("Differences are about how an outcome would be achieved or carried out, including:"
            "A) Change-enabling mechanisms: actions, changes, interventions, tools, or mechanisms that make the outcome possible; "
            "B) Execution pathways: steps, processes, workflows, procedures, or ways of carrying something out. "
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                "Placeholder for the chosen outcome enabler (recommendations, tactics, methods, etc.)"
                ),
            },
        },

        "WHO": {
            "dimension_description": (
                "actor_target — the people, groups, roles, stakeholders, or beneficiaries involved/affected/"
                "responsible/addressed (not an action, not a reason, not an outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen actor_target (people/groups/roles/beneficiaries)"
                ),
            },
        },

        "WHEN": {
            "dimension_description": (
                "time_urgency — timing, urgency, sequence, or frequency associated with when something occurs "
                "or is expected (not actions, reasons, or outcomes)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen time_urgency (timing/urgency/frequency/sequence)"
                ),
            },
        },

        "WHERE": {
            "dimension_description": (
                "location_context — the physical/digital location, channel, setting, or situational context "
                "in which something occurs/is encountered (not an action, not a reason, not an outcome)"
            ),
            "template_pattern": "{SUBJECT} {VERB_STATE} {SCAFFOLD} [ACTIONABLE_TAXONOMY_DIMENSION].",
            "required_form": (
                "a natural-language phrase that directly answers the research question in {language}"
            ),
            "slot_guidance": {
                "SUBJECT": "The canonical subject the sentence is grammatically about in {language}",
                "VERB_STATE": "a minimal linking verb/state verb appropriate in {language}",
                "SCAFFOLD": "words that connect the subject to the [ACTIONABLE_TAXONOMY_DIMENSION]",
                "[ACTIONABLE_TAXONOMY_DIMENSION]": (
                    "Placeholder for the chosen location_context (place/channel/setting/context)"
                ),
            },
        },
    },
}
