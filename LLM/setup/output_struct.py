from pydantic import BaseModel
from typing import get_origin, get_args
import json


class BaseModelJson(BaseModel):
    """Pydantic Base class with a class method for creating
    a JSON representation."""

    @classmethod
    def json_representation(cls):
        """Return a JSON schema representation of the class.
        Needed for the batch API.
        """
        # Get field information from the model
        fields = cls.__annotations__

        # Build properties dynamically
        properties = {}
        required = []

        for field_name, field_type in fields.items():
            # Determine JSON type based on Python type
            json_type = "string"  # Default

            # Handle generic types like list[str], dict, etc.
            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin is list:  # Handle lists
                json_type = "array"
                if args:  # If the list has a specific type (e.g., list[str])
                    item_type = args[0]
                    if item_type == str:
                        item_json_type = "string"
                    elif item_type == int:
                        item_json_type = "integer"
                    elif item_type == float:
                        item_json_type = "number"
                    elif item_type == bool:
                        item_json_type = "boolean"
                    else:
                        item_json_type = "object"  # Default for complex types
                    properties[field_name] = {
                        "type": json_type,
                        "items": {"type": item_json_type},
                    }
                    required.append(field_name)
                    continue
            elif origin is dict:  # Handle dictionaries
                json_type = "object"
            elif field_type == bool:
                json_type = "boolean"
            elif field_type == int:
                json_type = "integer"
            elif field_type == float:
                json_type = "number"

            # Add to properties
            properties[field_name] = {"type": json_type}

            # Add to required fields (assuming all fields are required)
            required.append(field_name)

        return {
            'type': 'json_schema',
            'json_schema': {
                'name': cls.__name__,  # Using cls.__name__ directly
                'schema': {
                    'type': 'object',
                    'properties': properties,
                    'required': required,
                    'additionalProperties': False,
                },
                'strict': True,
            }
        }

    @classmethod
    def get_last_key(cls):
        """Return the last key in the class."""
        return list(cls.__annotations__.keys())[-1]

    def as_string(self):
        """Return a string representation of the model."""
        # dict is deprecated in pydantic, use model_dump instead
        # and convert to JSON string
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=4)


class TopicAI(BaseModelJson):
    """Represents the topic of a text regarding AI."""
    topic_ai: str


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
