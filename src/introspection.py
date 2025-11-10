"""
Introspection utilities for Pydantic models.

Enables automated metadata extraction for path translation, reserved fields,
and schema evolution tracking.
"""

from typing import Type, Any, Union
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from src.markers import PathMarker


def get_path_fields(model: Type[BaseModel]) -> list[str]:
    """
    Find all fields marked with PathMarker.

    Handles Python 3.12+ type aliases and Union types (e.g., PathField | None).

    Args:
        model: Pydantic model class to inspect

    Returns:
        List of field names containing filesystem paths

    Example:
        >>> from src.models import UserRecord
        >>> get_path_fields(UserRecord)
        ['cwd', 'projectPaths']
    """
    from typing import get_origin, get_args
    import types

    def check_for_path_marker(annotation: Any) -> bool:
        """Check if annotation contains PathMarker, handling type aliases and unions."""
        # Check Annotated directly
        if get_origin(annotation) is not None:
            args = get_args(annotation)
            for arg in args[1:]:
                if isinstance(arg, PathMarker):
                    return True

        # Check Python 3.12+ type alias (__value__ attribute)
        if hasattr(annotation, '__value__'):
            actual_type = annotation.__value__
            if get_origin(actual_type) is not None:
                args = get_args(actual_type)
                for arg in args[1:]:
                    if isinstance(arg, PathMarker):
                        return True

        # Check Union types (e.g., PathField | None)
        origin = get_origin(annotation)
        if origin is Union or (origin is not None and getattr(origin, '__name__', '') == 'UnionType'):
            for union_arg in get_args(annotation):
                if check_for_path_marker(union_arg):
                    return True

        return False

    path_fields = []

    for field_name, field_info in model.model_fields.items():
        if check_for_path_marker(field_info.annotation):
            path_fields.append(field_name)

    return path_fields


def get_reserved_fields(model: Type[BaseModel]) -> dict[str, dict[str, Any]]:
    """
    Find all reserved (always-null) fields.

    Returns a mapping of field names to their metadata for fields that are
    reserved for future use and always null in current data.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict mapping field name to json_schema_extra metadata

    Example:
        >>> from src.models import UserRecord
        >>> get_reserved_fields(UserRecord)
        {'skills': {'status': 'reserved'}, 'mcp': {'status': 'reserved'}}
    """
    reserved = {}

    for field_name, field_info in model.model_fields.items():
        if (
            field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and field_info.json_schema_extra.get('status') == 'reserved'
        ):
            reserved[field_name] = field_info.json_schema_extra

    return reserved


def get_field_version_info(model: Type[BaseModel]) -> dict[str, str]:
    """
    Get version information for fields that track when they were added.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict mapping field name to version string

    Example:
        >>> from src.models import UserRecord
        >>> get_field_version_info(UserRecord)
        {'thinkingMetadata': '2.0.35'}
    """
    version_info = {}

    for field_name, field_info in model.model_fields.items():
        if (
            field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and 'added_in_version' in field_info.json_schema_extra
        ):
            version_info[field_name] = field_info.json_schema_extra['added_in_version']

    return version_info


def get_literal_values(field_info: FieldInfo) -> list[Any] | None:
    """
    Extract literal values from a field definition.

    Args:
        field_info: Pydantic field info

    Returns:
        List of literal values if field uses Literal type, None otherwise

    Example:
        >>> from src.models import UserRecord
        >>> field_info = UserRecord.model_fields['userType']
        >>> get_literal_values(field_info)
        ['external']
    """
    # This would require parsing the annotation
    # For now, return None - can be enhanced if needed
    return None


def model_summary(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Generate a comprehensive summary of a model's metadata.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict with model statistics and metadata

    Example:
        >>> from src.models import UserRecord
        >>> summary = model_summary(UserRecord)
        >>> summary['total_fields']
        16
        >>> summary['path_fields']
        ['cwd', 'projectPaths']
    """
    path_fields = get_path_fields(model)
    reserved = get_reserved_fields(model)
    version_info = get_field_version_info(model)

    # Count field types
    optional_fields = []
    required_fields = []

    for field_name, field_info in model.model_fields.items():
        if field_info.is_required():
            required_fields.append(field_name)
        else:
            optional_fields.append(field_name)

    return {
        'model_name': model.__name__,
        'total_fields': len(model.model_fields),
        'required_fields': required_fields,
        'optional_fields': optional_fields,
        'path_fields': path_fields,
        'reserved_fields': list(reserved.keys()),
        'versioned_fields': version_info,
        'docstring': model.__doc__,
    }


def print_model_summary(model: Type[BaseModel]) -> None:
    """
    Print a human-readable summary of a model.

    Args:
        model: Pydantic model class to inspect
    """
    summary = model_summary(model)

    print(f"Model: {summary['model_name']}")
    print(f"  Total fields: {summary['total_fields']}")
    print(f"  Required: {len(summary['required_fields'])}")
    print(f"  Optional: {len(summary['optional_fields'])}")

    if summary['path_fields']:
        print(f"  Path fields: {', '.join(summary['path_fields'])}")

    if summary['reserved_fields']:
        print(f"  Reserved fields: {', '.join(summary['reserved_fields'])}")

    if summary['versioned_fields']:
        print(f"  Versioned fields:")
        for field, version in summary['versioned_fields'].items():
            print(f"    {field}: added in {version}")


if __name__ == '__main__':
    # Demo
    from src.models import UserRecord, AssistantRecord, Message

    print('=' * 80)
    print('Model Introspection Demo')
    print('=' * 80)
    print()

    for model in [UserRecord, AssistantRecord, Message]:
        print_model_summary(model)
        print()
