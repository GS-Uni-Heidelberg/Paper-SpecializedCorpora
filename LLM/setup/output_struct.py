TextClassification = {
    "name": "print_annotation",
    "description": "Print the correct label for the input text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic_ai": {
                "type": "string",
                "description": (
                    "Whether the input text is '1_hauptthema', "
                    "'2_nebenthema' or '3_kein_thema'. "
                    "Must be one of the three strings to be valid input."
                ),
            },
        },
        "required": ["topic_ai"],
    }
}


TextClassificationExplained = {
    "name": "print_annotation_explained",
    "description": "Print an explanation and the correct label for the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concept_ai_mentioned": {
                "type": "array",
                "items": {
                    "type": "boolean"
                },
                "description": (
                    "Whether the concept of AI, according to our definition "
                    "is present in the text or not."
                ),
            },
            "leave_out_test": {
                "type": "string",
                "description": (
                    "Whether, when the passages with ai are removed, "
                    "the text is incoherent (1), lacking detail (2) "
                    "or barely different (3)."
                ),
            },
            "topic_ai": {
                "type": "string",
                "description": (
                    "Whether the input text is '1_hauptthema', "
                    "'2_nebenthema' or '3_kein_thema'. "
                    "Must be one of the three strings to be valid input."
                ),
            },
        },
        "required": [
            "concept_ai_mentioned",
            "leave_out_test",
            "topic_ai",
        ],
    }
}
