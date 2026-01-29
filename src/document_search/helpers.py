"""Helper functions for session chunking.

Small, focused functions for:
- Tool type/category via pattern matching on models
- Text extraction and formatting
"""

from __future__ import annotations

import src.schemas.session.models as models
from src.document_search.types import (
    LSPOperation,
    ToolCategory,
    ToolType,
)

# =============================================================================
# Tool Type and Category
# =============================================================================


def get_tool_type(tool_input: models.ToolInput) -> ToolType:
    """Get the ToolType for a tool input using pattern matching.

    Returns the tool type based on the typed ToolInput class.
    Raises ValueError for unknown types (catches new tools we haven't modeled).
    """
    match tool_input:
        # File operations
        case models.ReadToolInput():
            return 'Read'
        case models.WriteToolInput() | models.MalformedWriteToolInput():
            return 'Write'
        case models.EditToolInput():
            return 'Edit'
        case models.GlobToolInput():
            return 'Glob'
        case models.GrepToolInput():
            return 'Grep'
        case models.NotebookEditToolInput():
            return 'NotebookEdit'

        # Execution
        case models.BashToolInput():
            return 'Bash'
        case models.TaskToolInput():
            return 'Task'
        case models.TaskOutputToolInput():
            return 'TaskOutput'
        case models.AgentOutputToolInput():
            return 'AgentOutput'
        case models.BashOutputToolInput():
            return 'BashOutput'
        case models.KillShellToolInput():
            return 'KillShell'

        # Web
        case models.WebSearchToolInput():
            return 'WebSearch'
        case models.WebFetchToolInput():
            return 'WebFetch'

        # LSP
        case models.LSPToolInput():
            return 'LSP'

        # Planning
        case models.EnterPlanModeToolInput():
            return 'EnterPlanMode'
        case models.ExitPlanModeToolInput():
            return 'ExitPlanMode'

        # User interaction
        case models.AskUserQuestionToolInput():
            return 'AskUserQuestion'
        case models.TodoWriteToolInput():
            return 'TodoWrite'

        # MCP management
        case models.MCPSearchToolInput():
            return 'MCPSearch'
        case models.ListMcpResourcesToolInput():
            return 'ListMcpResources'
        case models.ReadMcpResourceToolInput():
            return 'ReadMcpResource'

        # Skills
        case models.SkillToolInput():
            return 'Skill'

        # Task management
        case models.TaskCreateToolInput():
            return 'TaskCreate'
        case models.TaskUpdateToolInput():
            return 'TaskUpdate'
        case models.TaskListToolInput():
            return 'TaskList'

        # MCP (third-party) - must be last
        case models.MCPToolInput():
            return 'mcp'

        case _:
            raise ValueError(f'Unknown tool input type: {type(tool_input).__name__}')


def get_tool_category(tool_input: models.ToolInput) -> ToolCategory:
    """Get the category for a tool input using type matching."""
    match tool_input:
        # File operations
        case (
            models.ReadToolInput()
            | models.WriteToolInput()
            | models.MalformedWriteToolInput()
            | models.EditToolInput()
            | models.GlobToolInput()
            | models.GrepToolInput()
            | models.NotebookEditToolInput()
        ):
            return 'file_operation'

        # Execution
        case (
            models.BashToolInput()
            | models.TaskToolInput()
            | models.TaskOutputToolInput()
            | models.AgentOutputToolInput()
            | models.BashOutputToolInput()
            | models.KillShellToolInput()
        ):
            return 'execution'

        # Web
        case models.WebSearchToolInput() | models.WebFetchToolInput():
            return 'web'

        # LSP
        case models.LSPToolInput():
            return 'lsp'

        # Planning
        case models.EnterPlanModeToolInput() | models.ExitPlanModeToolInput():
            return 'planning'

        # User interaction
        case models.AskUserQuestionToolInput() | models.TodoWriteToolInput():
            return 'interaction'

        # MCP management
        case models.MCPSearchToolInput() | models.ListMcpResourcesToolInput() | models.ReadMcpResourceToolInput():
            return 'mcp_management'

        # Skills
        case models.SkillToolInput():
            return 'skill'

        # Task management
        case models.TaskCreateToolInput() | models.TaskUpdateToolInput() | models.TaskListToolInput():
            return 'task_management'

        # MCP (third-party) - must be last
        case models.MCPToolInput():
            return 'mcp'

        case _:
            # Fail explicitly so we notice new tools
            raise ValueError(f'Unknown tool input type: {type(tool_input).__name__}')


# =============================================================================
# LSP Operation
# =============================================================================


def get_lsp_operation(tool_input: models.ToolInput) -> LSPOperation | None:
    """Extract LSP operation if this is an LSP tool."""
    if isinstance(tool_input, models.LSPToolInput):
        return tool_input.operation
    return None


# =============================================================================
# Text Extraction
# =============================================================================


def extract_text_from_message(
    message: models.Message,
    include_thinking: bool = False,
) -> str:
    """Extract text content from a message.

    Args:
        message: The Message object
        include_thinking: Whether to include ThinkingContent blocks

    Returns:
        Concatenated text content
    """
    content = message.content

    if isinstance(content, str):
        return content

    if not isinstance(content, (list, tuple)):
        return ''

    parts: list[str] = []
    for block in content:
        if isinstance(block, models.TextContent):
            parts.append(block.text)
        elif isinstance(block, models.ThinkingContent) and include_thinking:
            parts.append(block.thinking)

    return '\n\n'.join(parts)


def extract_file_path(tool_input: models.ToolInput) -> str | None:
    """Extract file path from tool input if applicable."""
    match tool_input:
        case models.ReadToolInput():
            return str(tool_input.file_path)
        case models.WriteToolInput() | models.MalformedWriteToolInput():
            return str(tool_input.file_path)
        case models.EditToolInput():
            return str(tool_input.file_path)
        case models.NotebookEditToolInput():
            return str(tool_input.notebook_path)
        case models.LSPToolInput():
            return str(tool_input.filePath)
        case _:
            return None


# =============================================================================
# Text Formatting
# =============================================================================


def format_tool_use_text(tool_use: models.ToolUseContent) -> str:
    """Format tool invocation as searchable text.

    Includes tool name and key parameters for semantic search.
    """
    lines = [f'Tool: {tool_use.name}']
    input_data = tool_use.input

    match input_data:
        case models.ReadToolInput():
            lines.append(f'File: {input_data.file_path}')

        case models.WriteToolInput() | models.MalformedWriteToolInput():
            lines.append(f'File: {input_data.file_path}')
            preview = input_data.content[:500]
            lines.append(f'Content: {preview}...' if len(input_data.content) > 500 else f'Content: {preview}')

        case models.EditToolInput():
            lines.append(f'File: {input_data.file_path}')
            lines.append(f'Old: {input_data.old_string[:200]}...')
            lines.append(f'New: {input_data.new_string[:200]}...')

        case models.BashToolInput():
            lines.append(f'Command: {input_data.command}')
            if input_data.description:
                lines.append(f'Description: {input_data.description}')

        case models.GrepToolInput():
            lines.append(f'Pattern: {input_data.pattern}')
            if input_data.path:
                lines.append(f'Path: {input_data.path}')

        case models.GlobToolInput():
            lines.append(f'Pattern: {input_data.pattern}')
            if input_data.path:
                lines.append(f'Path: {input_data.path}')

        case models.WebSearchToolInput():
            lines.append(f'Query: {input_data.query}')

        case models.WebFetchToolInput():
            lines.append(f'URL: {input_data.url}')
            lines.append(f'Prompt: {input_data.prompt}')

        case models.TaskToolInput():
            if input_data.description:
                lines.append(f'Description: {input_data.description}')
            lines.append(f'Agent type: {input_data.subagent_type}')
            prompt_preview = input_data.prompt[:500]
            lines.append(
                f'Prompt: {prompt_preview}...' if len(input_data.prompt) > 500 else f'Prompt: {prompt_preview}'
            )

        case models.LSPToolInput():
            lines.append(f'Operation: {input_data.operation}')
            lines.append(f'File: {input_data.filePath}')
            lines.append(f'Position: {input_data.line}:{input_data.character}')

        case models.SkillToolInput():
            lines.append(f'Skill: {input_data.skill}')
            if input_data.args:
                lines.append(f'Args: {input_data.args}')

        case models.AskUserQuestionToolInput():
            lines.extend(f'Question: {q.question}' for q in input_data.questions)

        case models.MCPToolInput():
            # Generic handling for MCP tools
            extra = input_data.get_extra_fields()
            for key, value in extra.items():
                if isinstance(value, str):
                    if len(value) <= 200:
                        lines.append(f'{key}: {value}')
                    else:
                        lines.append(f'{key}: {value[:200]}...')
                elif value is not None:
                    lines.append(f'{key}: {value!r}')

        case _:
            # For tools we don't specifically handle, try common fields
            if hasattr(input_data, 'query'):
                lines.append(f'Query: {input_data.query}')
            if hasattr(input_data, 'prompt'):
                lines.append(f'Prompt: {input_data.prompt}')

    return '\n'.join(lines)


def format_tool_result_text(
    tool_result: models.ToolResultContent,
    tool_use: models.ToolUseContent | None,
) -> str:
    """Format tool result as searchable text.

    Args:
        tool_result: The tool result content
        tool_use: The original tool invocation (for context)

    Returns:
        Formatted text for indexing
    """
    lines: list[str] = []

    # Add tool context if available
    if tool_use:
        lines.append(f'Tool Result: {tool_use.name}')
    else:
        lines.append('Tool Result')

    if tool_result.is_error:
        lines.append('[Error]')

    # Format content
    content = tool_result.content
    if content is None:
        lines.append('(no output)')
    elif isinstance(content, str):
        if len(content) <= 1000:
            lines.append(content)
        else:
            lines.append(f'{content[:1000]}...')
    elif isinstance(content, (list, tuple)):
        # Sequence of content blocks
        for block in content:
            if isinstance(block, models.TextContent):
                text = block.text
                if len(text) <= 500:
                    lines.append(text)
                else:
                    lines.append(f'{text[:500]}...')

    return '\n'.join(lines)


# =============================================================================
# Text Splitting
# =============================================================================


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text at paragraph boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split('\n\n')
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len + 2 > max_chars and current:
            # Save current chunk and start new one
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    # Don't forget last chunk
    if current:
        chunks.append('\n\n'.join(current))

    return chunks
