"""
Artifact handling modules for session clone/archive/restore/delete operations.

This package contains modules for collecting, transforming, and writing
session-specific artifacts like tool results, todos, plan files, etc.
"""

from __future__ import annotations

from .agent_ids import (
    AGENT_FILENAME_PATTERN,
    apply_agent_id_mapping,
    extract_agent_ids_from_files,
    extract_base_agent_id,
    generate_agent_id_mapping,
    generate_clone_agent_id,
    transform_agent_filename,
)
from .custom_title import (
    CLONE_SUFFIX_PATTERN,
    extract_base_custom_title,
    extract_custom_title_from_file,
    extract_custom_title_from_records,
    generate_clone_custom_title,
)
from .paths import (
    MissingCwdError,
    extract_source_project_path,
)
from .plan_files import (
    SLUG_RECORD_TYPES,
    apply_slug_mapping,
    collect_plan_files,
    extract_base_slug,
    extract_slugs_from_records,
    generate_clone_slug,
    write_plan_files,
)
from .session_env import (
    SESSION_ENV_DIR,
    create_session_env_dir,
    validate_session_env_empty,
)
from .todos import (
    TODOS_DIR,
    collect_todos,
    transform_todo_filename,
    write_todos,
)
from .tool_results import (
    collect_tool_results,
    get_tool_results_dir,
    write_tool_results,
)

__all__ = [
    # custom_title
    'CLONE_SUFFIX_PATTERN',
    'extract_custom_title_from_file',
    'extract_custom_title_from_records',
    'extract_base_custom_title',
    'generate_clone_custom_title',
    # agent_ids
    'AGENT_FILENAME_PATTERN',
    'extract_agent_ids_from_files',
    'extract_base_agent_id',
    'generate_agent_id_mapping',
    'generate_clone_agent_id',
    'transform_agent_filename',
    'apply_agent_id_mapping',
    # plan_files
    'SLUG_RECORD_TYPES',
    'extract_base_slug',
    'extract_slugs_from_records',
    'collect_plan_files',
    'generate_clone_slug',
    'write_plan_files',
    'apply_slug_mapping',
    # session_env
    'SESSION_ENV_DIR',
    'validate_session_env_empty',
    'create_session_env_dir',
    # todos
    'TODOS_DIR',
    'collect_todos',
    'transform_todo_filename',
    'write_todos',
    # tool_results
    'get_tool_results_dir',
    'collect_tool_results',
    'write_tool_results',
    # paths
    'MissingCwdError',
    'extract_source_project_path',
]
