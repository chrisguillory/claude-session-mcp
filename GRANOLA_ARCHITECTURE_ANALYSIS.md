# Granola MCP Server Architecture Analysis

## Project Overview

**Location**: `/Users/chris/granola-mcp/`
**Type**: MCP (Model Context Protocol) Server
**Language**: Python 3.11+
**Framework**: FastMCP (async MCP framework)
**Purpose**: Provides Claude Code with access to Granola meeting notes, transcripts, and data management via MCP tools

---

## 1. Project Structure & File Organization

```
granola-mcp/
├── granola-mcp.py           # Main MCP server (36KB, 1130 lines)
├── src/
│   ├── __init__.py          # Package marker
│   ├── models.py            # Pydantic models (485 lines)
│   ├── logging.py           # DualLogger utility (31 lines)
│   └── helpers.py           # Helper functions (278 lines)
├── pyproject.toml           # Ruff formatting config
├── README.md                # API research and installation guide
├── CLAUDE.md                # Development guidance for Claude Code
└── .cursorindexingignore    # Cursor IDE config
```

### File Purposes

| File | Purpose | Key Content |
|------|---------|------------|
| `granola-mcp.py` | Main server entry point | FastMCP setup, lifespan management, 17 MCP tools |
| `src/models.py` | Pydantic models | Strict API response validation, 40+ model classes |
| `src/logging.py` | Logging abstraction | DualLogger: logs to stdout + MCP context |
| `src/helpers.py` | Utility functions | Auth, ProseMirror conversion, markdown analysis |

---

## 2. MCP Server Implementation

### Architecture Pattern

```python
# granola-mcp.py structure:
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import ToolAnnotations

# 1. Global state management
_temp_dir: tempfile.TemporaryDirectory | None = None
_http_client: httpx.AsyncClient | None = None

# 2. Lifespan context manager (resource lifecycle)
@asynccontextmanager
async def lifespan(server):
    """Manage resources - cleanup on shutdown."""
    # Initialize resources (temp dir, HTTP client)
    try:
        yield {}
    finally:
        # Cleanup resources
        
# 3. Create FastMCP server instance
mcp = FastMCP('granola', lifespan=lifespan)

# 4. Define tools with decorators
@mcp.tool(annotations=ToolAnnotations(...))
async def tool_name(...) -> ResultType:
    """Docstring with Args/Returns."""
    logger = DualLogger(ctx)
    # Implementation
    
# 5. Main entry point with CLI argument parsing
def main() -> None:
    args = parse_args()
    mcp.run()
```

### Key Components

#### A. Lifespan Management (Lines 72-91)
```python
@asynccontextmanager
async def lifespan(server):
    """Manage resources - cleanup on shutdown."""
    global _temp_dir, _export_dir, _http_client

    # Initialize
    _temp_dir = tempfile.TemporaryDirectory()
    _export_dir = Path(_temp_dir.name)
    _http_client = httpx.AsyncClient(timeout=30.0)

    try:
        yield {}
    finally:
        # Cleanup on shutdown
        if _http_client:
            await _http_client.aclose()
        if _temp_dir:
            _temp_dir.cleanup()
```

**Pattern**: Uses `@asynccontextmanager` for resource lifecycle. Resources are:
- Initialized before `yield`
- Available during server operation
- Guaranteed cleanup in `finally` block

#### B. HTTP Client Global State
```python
_http_client: httpx.AsyncClient | None = None
```
- Shared async HTTP client across all tools
- Initialized once during lifespan
- Reused for API calls (connection pooling)

#### C. Caching Pattern (Lines 101-131)
```python
@cached(ttl=None, cache=Cache.MEMORY)
async def _get_documents_cached(
    limit: int, offset: int, list_id: str | None = None
) -> list:
    """Fetch with automatic caching."""
    # API call
    
# Usage:
batch = await _get_documents_cached(limit=40, offset=0, list_id=None)
```

**Features**:
- Session-based caching (TTL=None: persists until server restart)
- Cache key auto-generated from parameters: `(limit, offset, list_id)`
- In-memory storage using `aiocache`
- Used for pagination (batch fetches of 40)

### MCP Tool Definition Pattern

#### Tool Decorator with Annotations
```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='Tool Display Name',
        readOnlyHint=True,      # Read-only (no side effects)
        idempotentHint=True     # Safe to call multiple times
    )
)
async def tool_name(
    param1: str,
    param2: int | None = None,
    ctx: Context,               # MCP context (optional, for logging)
) -> ResultType:
    """
    One-sentence summary.
    
    Extended description here.
    
    Args:
        param1: Description
        param2: Description (default: None)
        ctx: MCP context
        
    Returns:
        ResultType with description
    """
    logger = DualLogger(ctx)
    await logger.info(f'Starting operation')
    
    # Implementation with error handling
    try:
        result = await _http_client.post(...)
        result.raise_for_status()
    except Exception as e:
        await logger.error(f'Failed: {e}')
        raise
        
    # Validate with Pydantic
    data = ResponseModel.model_validate(result.json())
    
    # Return structured result
    return ResultModel(field1=..., field2=...)
```

### ToolAnnotations Usage

From lines 139-947, patterns observed:

**Read-only tools** (query/list operations):
```python
@mcp.tool(annotations=ToolAnnotations(title='List Meetings', readOnlyHint=True))
async def list_meetings(...) -> list[MeetingListItem]:
```

**Write operations** (create/update/delete):
```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='Delete Meeting',
        readOnlyHint=False,      # Has side effects
        idempotentHint=True      # Safe to retry
    )
)
async def delete_meeting(...) -> DeleteMeetingResult:
```

**Non-idempotent write** (can't safely retry):
```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='Create Workspace',
        readOnlyHint=False,
        idempotentHint=False     # Not safe to retry (creates duplicates)
    )
)
async def create_workspace(...) -> CreateWorkspaceResult:
```

### Tools Implemented (17 total)

**Discovery & Organization**:
- `list_meetings()` - With client-side filtering and pagination
- `get_meeting_lists()` - Get collections
- `get_meetings()` - Batch retrieval by IDs

**Download/Export**:
- `download_note()` - AI-generated notes as Markdown
- `download_transcript()` - Meeting transcript
- `download_private_notes()` - User's private notes

**Management**:
- `delete_meeting()` / `undelete_meeting()` - Soft delete via timestamps
- `list_deleted_meetings()` - List deleted IDs
- `update_meeting()` - Update title/attendees

**Workspaces**:
- `list_workspaces()` - Get all workspaces
- `create_workspace()` - Create new workspace
- `delete_workspace()` - Delete workspace

---

## 3. Pydantic Model Patterns & Type Definitions

### File: `src/models.py` (485 lines)

#### Base Model with Strict Validation

```python
class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields."""
    model_config = pydantic.ConfigDict(
        extra='forbid',      # Reject unknown fields
        strict=True          # Strict type checking
    )
```

**Key settings**:
- `extra='forbid'`: Fail immediately if API adds unknown fields (catch breaking changes)
- `strict=True`: No type coercion (string "123" ≠ int 123)

#### Nested Model Hierarchy

```python
# Level 1: Core data structures
class PersonName(BaseModel):
    fullName: str
    givenName: str | None = None
    familyName: str | None = None

class PersonDetails(BaseModel):
    name: PersonName
    jobTitle: str | None = None
    linkedin: LinkedInInfo | None = None

# Level 2: Composite types
class PersonInfo(BaseModel):
    person: PersonDetails
    company: CompanyDetails

# Level 3: Domain objects
class Creator(BaseModel):
    name: str
    email: str
    details: PersonInfo

class People(BaseModel):
    creator: Creator
    attendees: list[Attendee]
    manual_attendee_edits: list[ManualAttendeeEdit] | None = None

# Level 4: Main document
class GranolaDocument(BaseModel):
    id: str
    title: str | None
    people: People | None
    created_at: str
    notes: Notes
    # ... 30+ fields
    last_viewed_panel: LastViewedPanel | None = None

# Level 5: API Response
class DocumentsResponse(BaseModel):
    docs: list[GranolaDocument]
    deleted: list[str]
```

#### Required vs Optional Fields Pattern

```python
class GranolaDocument(BaseModel):
    # REQUIRED (no default, no Optional)
    id: str
    created_at: str
    user_id: str
    updated_at: str
    
    # NULLABLE (Optional[Type] or Type | None with = None)
    title: str | None = None
    people: People | None = None
    deleted_at: str | None = None
    
    # ALWAYS NULL (typed as None, never has value)
    chapters: None = None
    summary: None = None
    notification_config: None = None
```

#### Special Field Handling

**ProseMirror with Field Alias**:
```python
class ProseMirrorAttrs(BaseModel):
    id: str
    timestamp_to: str | None = pydantic.Field(
        default=None,
        alias='timestamp-to'  # JSON has hyphen, Python uses underscore
    )
```

**Conditional Field Exclusion**:
```python
class MeetingListItem(BaseModel):
    participants: list[ParticipantInfo] | None = pydantic.Field(
        default=None,
        exclude_if=lambda v: v is None  # Omit from response if None
    )
```

#### Response Type Hierarchy

```python
# API Response Models (full data from Granola API)
class DocumentsResponse(BaseModel):
    docs: list[GranolaDocument]
    deleted: list[str]

# Tool Result Models (simplified for MCP response)
class MeetingListItem(BaseModel):
    id: str
    title: str
    created_at: str
    participant_count: int
    participants: list[ParticipantInfo] | None = None

# Download Result Models (include metadata)
class NoteDownloadResult(BaseModel):
    path: str
    size_bytes: int
    section_count: int        # Structural metrics
    bullet_count: int
    word_count: int
    panel_title: str
    template_slug: str | None
```

### Type Definition Patterns

**Union types for flexibility**:
```python
content: dict | str  # Can be ProseMirror JSON (dict) or HTML (str)
```

**List patterns**:
```python
attendees: list[Attendee]                   # Required list
optional_list: list[str] | None = None      # Optional list
```

**Literal types (when available)**:
```python
source: str  # "microphone" or "system"
# Could be: Literal["microphone", "system"]
```

---

## 4. Error Handling and Logging Patterns

### Logging: `src/logging.py`

```python
from mcp.server.fastmcp import Context

class DualLogger:
    """Logs to both stdout and MCP client context."""
    
    def __init__(self, ctx: Context):
        self.ctx = ctx
    
    def _timestamp(self) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    async def info(self, msg: str):
        print(f'[{self._timestamp()}] [INFO] {msg}')
        await self.ctx.info(msg)
    
    async def debug(self, msg: str):
        print(f'[{self._timestamp()}] [DEBUG] {msg}')
        await self.ctx.debug(msg)
    
    async def warning(self, msg: str):
        print(f'[{self._timestamp()}] [WARNING] {msg}')
        await self.ctx.warning(msg)
    
    async def error(self, msg: str):
        print(f'[{self._timestamp()}] [ERROR] {msg}')
        await self.ctx.error(msg)
```

**Pattern**: 
- Dual output: stdout (for local debugging) + MCP context (for client visibility)
- Timestamps on all messages
- `async` methods for MCP context calls

### Usage in Tools

```python
async def download_note(
    document_id: str,
    filename: str,
    ctx: Context
) -> NoteDownloadResult:
    logger = DualLogger(ctx)
    await logger.info(f'Downloading notes for document {document_id}')
    
    try:
        # API calls
        response = await _http_client.post(...)
        response.raise_for_status()
        
        # Validation
        data = ResponseModel.model_validate(response.json())
        
        # File operations
        file_path.write_text(content)
        
        await logger.info(f'Downloaded to {file_path}')
        return NoteDownloadResult(...)
        
    except Exception as e:
        await logger.error(f'Failed: {e}')
        raise  # Re-raise for MCP framework to handle
```

### Error Handling Patterns

**HTTP Error Handling**:
```python
response = await _http_client.post(url, json=payload, headers=headers)
response.raise_for_status()  # Raises httpx.HTTPStatusError on 4xx/5xx
```

**Data Validation**:
```python
try:
    data = DocumentsResponse.model_validate(response.json())
except pydantic.ValidationError as e:
    # Pydantic validation failed - API changed or response malformed
    raise
```

**Custom Errors**:
```python
if not doc_data.docs:
    raise ValueError(f'Document {document_id} not found')

if not summary_panel:
    # Fall back to first panel
    summary_panel = panels[0]
```

**No Silent Failures**:
- All API errors immediately propagate
- Validation errors fail fast with clear messages
- Pydantic catches schema mismatches immediately

---

## 5. Tool Annotations and Documentation Style

### Tool Annotation Components

```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='User-Visible Title',      # Display name in Claude Code
        readOnlyHint=True,               # Set False if tool modifies data
        idempotentHint=True              # Set False if retry-unsafe
    )
)
```

| Annotation | Values | Purpose |
|-----------|--------|---------|
| `title` | string | Human-readable tool name shown in Claude Code UI |
| `readOnlyHint` | True/False | Indicates read-only (no side effects) |
| `idempotentHint` | True/False | Safe to call multiple times with same parameters |

### Tool with No Annotations

```python
@mcp.tool()
async def create_workspace(...) -> CreateWorkspaceResult:
    """..."""
```

Defaults to: no title override, assumed read-write, not idempotent

### Docstring Style

```python
async def tool_name(
    param1: str,
    param2: int | None = None,
    ctx: Context,
) -> ResultType:
    """
    One-sentence summary describing what the tool does.
    
    Extended description providing context, use cases, or
    important details about implementation.
    
    Args:
        param1: Description of what param1 does
        param2: Description of optional param2 (default: None)
        ctx: MCP context for logging
        
    Returns:
        ResultType with description of fields returned
    """
```

### Real Examples

**Read-only with detailed description**:
```python
@mcp.tool(annotations=ToolAnnotations(title='List Meetings', readOnlyHint=True))
async def list_meetings(...) -> list[MeetingListItem]:
    """
    List Granola meetings with optional client-side filtering.

    Fetches meetings in batches of 40 (with caching) and filters by title and/or date.
    The Granola API does not support server-side search, so filtering is done client-side.
    Results are cached per pagination window for performance.

    Args:
        title_contains: Optional substring to filter by title
        case_sensitive: Whether title filtering should be case-sensitive (default: False)
        list_id: Optional list ID to filter meetings by list (server-side filtering)
        limit: Maximum number of meetings to return. Use 0 to return all (default: 20)
        created_at_gte: Filter meetings created on or after this date (ISO 8601: "YYYY-MM-DD")
        created_at_lte: Filter meetings created on or before this date (ISO 8601: "YYYY-MM-DD")
        include_participants: Include full participant details (default: False for efficiency)

    Returns:
        List of meetings with id, title, date, and metadata
    """
```

**Write operation with warnings**:
```python
@mcp.tool()
async def delete_workspace(ctx: Context, workspace_id: str) -> DeleteWorkspaceResult:
    """
    Delete a workspace.

    Deletes a workspace by setting its deleted_at timestamp. This is a soft delete
    operation - the workspace data is retained but marked as deleted.

    WARNING: This is a destructive operation. Use with caution.

    Args:
        ctx: MCP context
        workspace_id: ID of the workspace to delete

    Returns:
        DeleteWorkspaceResult with workspace_id and deletion timestamp
    """
```

---

## 6. CLI vs MCP Interface Patterns

### CLI Entry Point (Lines 1085-1129)

```python
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Granola MCP Server')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable PyCharm remote debugging',
    )
    parser.add_argument(
        '--debug-host',
        default=os.environ.get('DEBUG_HOST', 'host.docker.internal'),
        help='Debug host (default: host.docker.internal or $DEBUG_HOST)',
    )
    parser.add_argument(
        '--debug-port',
        type=int,
        default=int(os.environ.get('DEBUG_PORT', '5678')),
        help='Debug port (default: 5678 or $DEBUG_PORT)',
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the Granola MCP server."""
    args = parse_args()

    if args.debug:
        print(f'Enabling PyCharm remote debugging: {args.debug_host}:{args.debug_port}')
        import pydevd_pycharm
        pydevd_pycharm.settrace(
            args.debug_host,
            port=args.debug_port,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    print('Starting Granola MCP server')
    mcp.run()

if __name__ == '__main__':
    main()
```

**CLI Features**:
- Debug flag for PyCharm remote debugging
- Environment variable support for debug host/port
- Simple entry point that calls `mcp.run()`

### Script Header (Lines 1-12)

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "aiocache",
#   "fastmcp>=2.12.5",
#   "httpx",
#   "markdownify",
#   "pydantic>=2.0",
#   "pydevd-pycharm~=241.18034.82",
# ]
# ///
```

**Pattern**:
- `#!/usr/bin/env -S uv run --script` - Makes script executable as Python script
- PEP 723 inline script metadata (requires Python 3.11+)
- Dependencies declared inline for portability
- Version constraints for reproducibility

### MCP vs CLI Distinction

**MCP Interface** (exported via @mcp.tool):
- Async functions with specific signatures
- Context parameter for logging
- Structured return types (Pydantic models)
- Called by Claude Code via MCP protocol

**CLI Interface** (command line execution):
- `parse_args()` for argument parsing
- `main()` entry point
- Debug flags for development
- Environment variables for configuration

**No clash**: CLI args only control startup behavior (debug mode), not the MCP tools themselves.

---

## 7. Key Implementation Patterns & Architectures

### Pattern: Async Generator for Pagination

```python
async def list_meetings(...) -> list[MeetingListItem]:
    """..."""
    
    async def document_generator():
        """Async generator that yields documents in batches of 40."""
        offset = 0
        batch_size = 40
        while True:
            batch = await _get_documents_cached(
                limit=batch_size, offset=offset, list_id=list_id
            )
            if not batch:
                break
            for doc in batch:
                yield doc
            offset += batch_size
    
    results = []
    async for doc in document_generator():
        # Client-side filtering
        if should_include(doc):
            results.append(convert_to_result(doc))
        
        if limit > 0 and len(results) >= limit:
            break
    
    return results
```

**Benefits**:
- Lazy evaluation of API calls
- Memory efficient (batch streaming)
- Easy to add client-side filters
- Automatic caching per batch

### Pattern: Metadata Extraction

```python
# Extract metadata from markdown
def analyze_markdown_metadata(markdown: str) -> dict:
    lines = markdown.split('\n')
    heading_breakdown = {'h1': 0, 'h2': 0, 'h3': 0}
    section_count = 0
    
    for line in lines:
        if line.startswith('### '):
            heading_breakdown['h3'] += 1
            section_count += 1
        elif line.startswith('## '):
            heading_breakdown['h2'] += 1
        elif line.startswith('# '):
            heading_breakdown['h1'] += 1
    
    bullet_count = sum(1 for line in lines if re.match(r'^\s*[-*]\s', line))
    word_count = len([w for markdown.split() if w.strip()])
    
    return {
        'section_count': section_count,
        'bullet_count': bullet_count,
        'heading_breakdown': heading_breakdown,
        'word_count': word_count,
    }
```

**Usage**:
```python
metadata = analyze_markdown_metadata(markdown)
return NoteDownloadResult(
    path=str(file_path),
    size_bytes=len(markdown.encode('utf-8')),
    section_count=metadata['section_count'],
    bullet_count=metadata['bullet_count'],
    heading_breakdown=metadata['heading_breakdown'],
    word_count=metadata['word_count'],
    panel_title=summary_panel.title,
    template_slug=summary_panel.template_slug,
)
```

### Pattern: ProseMirror JSON Conversion

```python
def prosemirror_to_markdown(content: dict, depth: int = 0) -> str:
    """Recursive conversion with depth tracking for nested lists."""
    
    if not isinstance(content, dict):
        return ''
    
    node_type = content.get('type', '')
    
    # Document root - join children with double newlines
    if node_type == 'doc':
        children = content.get('content', [])
        return '\n\n'.join(prosemirror_to_markdown(child, depth) for child in children)
    
    # Heading - use level attribute
    if node_type == 'heading':
        level = content.get('attrs', {}).get('level', 1)
        text = extract_text(content)
        return f'{"#" * level} {text}'
    
    # Lists - track depth for indentation
    if node_type == 'bulletList':
        items = content.get('content', [])
        lines = []
        for item in items:
            if item.get('type') == 'listItem':
                item_lines = process_list_item(item, depth)
                lines.extend(item_lines)
        return '\n'.join(lines)
    
    # ... more node types
```

**Features**:
- Recursive tree traversal
- Depth tracking for nested lists
- Text extraction with mark handling (bold, italic, links)
- Proper joining (double newline for block, space for inline)

### Pattern: Download Tools (Consistent Architecture)

All download tools follow this pattern:

```python
async def download_type(document_id: str, filename: str, ctx: Context) -> ResultType:
    logger = DualLogger(ctx)
    await logger.info(f'Downloading ...')
    
    headers = get_auth_headers()
    
    # 1. Fetch document metadata
    doc_response = await _http_client.post(url, json={...}, headers=headers)
    doc_response.raise_for_status()
    doc_data = DocumentsResponse.model_validate(doc_response.json())
    document = doc_data.docs[0]
    
    # 2. Fetch specific content
    content_response = await _http_client.post(
        content_url, json={'document_id': document_id}, headers=headers
    )
    content_response.raise_for_status()
    
    # 3. Convert to markdown
    markdown = convert_to_markdown(content)
    
    # 4. Add header with title/date
    title = document.title or '(Untitled)'
    date_str = format_date(document.created_at)
    full_markdown = f'# {title}\n\n{date_str}\n\n{markdown}'
    
    # 5. Calculate metadata
    metadata = analyze_metadata(full_markdown)
    
    # 6. Write to file
    file_path = _export_dir / filename
    file_path.write_text(full_markdown, encoding='utf-8')
    
    await logger.info(f'Downloaded to {file_path}')
    
    return ResultType(
        path=str(file_path),
        size_bytes=len(full_markdown.encode('utf-8')),
        **metadata
    )
```

---

## 8. Dependencies and Requirements

### Top-level dependencies (`granola-mcp.py` header):

```python
requires-python = ">=3.11,<3.13"
dependencies = [
    "aiocache",              # Session-based in-memory caching
    "fastmcp>=2.12.5",       # FastMCP server framework
    "httpx",                 # Async HTTP client
    "markdownify",           # HTML to Markdown conversion
    "pydantic>=2.0",         # Type validation and serialization
    "pydevd-pycharm~=241",   # PyCharm remote debugging
]
```

### Import organization in `granola-mcp.py`:

```python
# Standard library
from __future__ import annotations
import argparse
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

# Third party
import httpx
from aiocache import Cache, cached
import markdownify
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

# Local
from src.helpers import (
    analyze_markdown_metadata,
    convert_utc_to_local,
    get_auth_headers,
    prosemirror_to_markdown,
)
from src.logging import DualLogger
from src.models import (
    # 20+ model imports
)
```

---

## 9. Code Style and Conventions

### Python Version and Features

- **Target**: Python 3.11+ (uses `from __future__ import annotations` for compatibility)
- **Type hints**: Full type annotations on all functions
- **Async/await**: Async throughout (async context managers, async generators, async for)

### Naming Conventions

- **Private functions**: `_get_documents_cached()` (single leading underscore)
- **Global state**: `_temp_dir`, `_http_client` (capitalized for constants)
- **Tool functions**: `list_meetings()`, `download_note()` (snake_case, descriptive)
- **Model classes**: `GranolaDocument`, `MeetingListItem` (PascalCase)

### String Style

From `pyproject.toml`:
```toml
[tool.ruff.format]
quote-style = "single"
```

**Single quotes** enforced throughout (unless string contains single quotes).

### Docstring Format

```python
def function_name(param1: str, param2: int) -> ResultType:
    """
    One-line summary (imperative mood).
    
    Extended description providing context, edge cases,
    or important implementation details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with (default: value) if applicable
        
    Returns:
        ResultType with description of what's in the result
        
    Raises:
        ValueError: When specific condition occurs
    """
```

### Code Organization

```python
# 1. Imports (grouped: future, stdlib, third-party, local)
# 2. Global state declarations
# 3. Lifespan/initialization code
# 4. Server instance creation
# 5. Helper functions (private, internal utilities)
# 6. MCP tool definitions
# 7. CLI entry point
# 8. Main guard
```

### Error Handling Convention

```python
try:
    response = await _http_client.post(...)
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    raise  # Re-raise with context
except Exception as e:
    await logger.error(f'Description: {e}')
    raise  # Always re-raise - let MCP framework handle
```

**Pattern**: All errors propagate - no silent failures or error suppression.

---

## 10. Configuration and Environment

### Python Script Execution (PEP 723)

The script uses PEP 723 inline script metadata for portability:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [...]
# ///
```

This allows running as:
```bash
uv run --script granola-mcp.py
# or
granola-mcp.py  # With execute permission
```

### Installation Command

```bash
claude mcp add --scope user --transport stdio granola -- uv run --script ~/granola-mcp/granola-mcp.py
```

Breakdown:
- `claude mcp add` - Add MCP server to Claude Code
- `--scope user` - User-scoped (not global)
- `--transport stdio` - Use stdio for communication
- `granola` - Server name
- `-- uv run --script ...` - Command to run the server

### Debug Flags

```bash
uv run --script granola-mcp.py --debug --debug-host localhost --debug-port 5678
```

Environment variable support:
- `DEBUG_HOST` - Override debug host (default: `host.docker.internal`)
- `DEBUG_PORT` - Override debug port (default: `5678`)

---

## 11. API Integration Pattern

### Authentication

```python
def get_auth_token() -> str:
    """Read WorkOS access token from local Granola app storage."""
    granola_dir = Path.home() / 'Library' / 'Application Support' / 'Granola'
    supabase_file = granola_dir / 'supabase.json'
    
    if not supabase_file.exists():
        raise FileNotFoundError(f'Granola auth file not found at {supabase_file}')
    
    with open(supabase_file) as f:
        data = json.load(f)
    
    tokens = json.loads(data['workos_tokens'])
    return tokens['access_token']

def get_auth_headers() -> dict[str, str]:
    """Get HTTP headers with Bearer token."""
    token = get_auth_token()
    return {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
```

**Pattern**: Read from local app storage (no credential management in code).

### API Request Pattern

```python
# Prepare
headers = get_auth_headers()
url = 'https://api.granola.ai/v2/get-documents'
payload = {'limit': 40, 'offset': 0}

# Execute
response = await _http_client.post(url, json=payload, headers=headers)
response.raise_for_status()  # Raises on 4xx/5xx

# Validate
data = ResponseModel.model_validate(response.json())  # Strict Pydantic validation

# Process
for doc in data.docs:
    # ...
```

### Endpoints Used

| Endpoint | Method | Parameters | Returns |
|----------|--------|-----------|---------|
| `/v2/get-documents` | POST | `limit`, `offset`, `include_last_viewed_panel`, `id`, `list_id` | `{docs: [...], deleted: [...]}` |
| `/v1/get-document-panels` | POST | `document_id` | `[{id, title, content, ...}]` |
| `/v1/get-document-transcript` | POST | `document_id` | `[{document_id, text, start_timestamp, ...}]` |
| `/v1/get-documents-batch` | POST | `document_ids` | `{docs: [...]}` |
| `/v1/get-document-lists-metadata` | POST | `include_document_ids`, `include_only_joined_lists` | `{lists: {...}}` |
| `/v1/update-document` | POST | `id`, fields to update | `{id, ...}` |
| `/v1/get-workspaces` | POST | `{}` | `{workspaces: [...]}` |
| `/v2/create-workspace` | POST | `display_name`, `is_locked`, `logo_url`, ... | `{workspace_id, ...}` |
| `/v1/delete-workspace` | POST | `workspace_id` | `{workspace_id, deleted_at}` |

---

## Summary: Architecture Principles

### Design Principles Observed

1. **Separation of Concerns**
   - `granola-mcp.py`: MCP server logic
   - `src/models.py`: Data validation
   - `src/logging.py`: Logging abstraction
   - `src/helpers.py`: Utility functions

2. **Fail Fast**
   - Strict Pydantic validation (forbid extra fields, strict types)
   - All errors propagate (no silent failures)
   - HTTP errors raised immediately

3. **Resource Management**
   - Lifespan context manager for initialization/cleanup
   - Shared HTTP client (connection pooling)
   - Temp directory auto-cleanup

4. **API Abstraction**
   - Authentication handled centrally
   - Cached requests reduce API load
   - Response validation at boundaries

5. **User Experience**
   - Dual logging (stdout + MCP context)
   - Structured result types (Pydantic models)
   - Comprehensive tool documentation
   - Sensible defaults (readOnlyHint, idempotentHint)

6. **Developer Experience**
   - Type hints throughout
   - Clear docstrings with examples
   - CLI support for debugging
   - Modular code structure

---

## Replication Guidelines for claude-session-mcp

When building a similar MCP server, follow these patterns:

1. **Project structure**: Main file + `src/` directory for models, helpers, logging
2. **MCP setup**: Lifespan context manager, global state for resources
3. **Tools**: Async functions with `@mcp.tool` decorator, ToolAnnotations, Context parameter
4. **Models**: Strict Pydantic with `extra='forbid'`, type unions for flexibility
5. **Errors**: Fail fast, no silent failures, propagate all errors
6. **Logging**: DualLogger pattern for visibility in both CLI and MCP context
7. **API calls**: Async HTTP client, centralized auth, response validation
8. **Documentation**: Comprehensive docstrings with Args/Returns
9. **Code style**: Type hints, single quotes, snake_case, comprehensive docstrings
