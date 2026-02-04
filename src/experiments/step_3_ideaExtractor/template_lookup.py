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
