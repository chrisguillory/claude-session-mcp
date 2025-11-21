# Granola MCP Server Exploration Summary

## Directory Analyzed
`/Users/chris/granola-mcp/` - MCP server providing access to Granola meeting notes and data

## Key Files Located

### Main Server File
- **Path**: `/Users/chris/granola-mcp/granola-mcp.py`
- **Size**: 1,130 lines
- **Purpose**: Complete MCP server implementation with 17 tools

### Supporting Modules (in `/Users/chris/granola-mcp/src/`)
1. **models.py** (485 lines) - Pydantic models for strict API validation
2. **logging.py** (31 lines) - DualLogger utility class
3. **helpers.py** (278 lines) - Authentication, conversion, and utility functions
4. **__init__.py** - Package marker

### Configuration & Documentation
- **pyproject.toml** - Ruff formatting configuration (single quotes)
- **README.md** - API research findings, endpoints, installation instructions
- **CLAUDE.md** - Developer guidance for Claude Code integration

---

## Architecture Overview

### 1. MCP Server Structure

The server follows this pattern:

1. **Global State Management** (lines 63-69)
   - `_temp_dir`: Temporary directory for exports
   - `_http_client`: Shared async HTTP client
   - `_export_dir`: Path to temp directory

2. **Lifespan Management** (lines 72-91)
   - Uses `@asynccontextmanager` decorator
   - Initializes resources (HTTP client, temp dir)
   - Guarantees cleanup on shutdown

3. **FastMCP Server** (line 94)
   - `mcp = FastMCP('granola', lifespan=lifespan)`
   - Single global instance

4. **Tool Definitions** (lines 139-1077)
   - 17 async tools decorated with `@mcp.tool`
   - Each uses `ToolAnnotations` for metadata
   - Context parameter for logging

5. **CLI Entry Point** (lines 1085-1129)
   - Parse command-line arguments (--debug, --debug-host, --debug-port)
   - Support for PyCharm remote debugging
   - Environment variable support

### 2. Project Dependencies

From inline script metadata (PEP 723):
```
requires-python = ">=3.11,<3.13"
dependencies = [
    aiocache,           # Session-based caching
    fastmcp>=2.12.5,    # MCP server framework
    httpx,              # Async HTTP client
    markdownify,        # HTML to Markdown
    pydantic>=2.0,      # Type validation
    pydevd-pycharm,     # Debug support
]
```

### 3. MCP Tools Implemented (17 total)

**Meeting Discovery & Organization**:
- `list_meetings()` - List with client-side filtering, caching, pagination
- `get_meeting_lists()` - Get meeting collections
- `get_meetings()` - Batch retrieval by IDs

**Download/Export**:
- `download_note()` - AI-generated notes as Markdown
- `download_transcript()` - Meeting transcripts
- `download_private_notes()` - User's private notes

**Management**:
- `delete_meeting()` / `undelete_meeting()` - Soft delete/restore
- `list_deleted_meetings()` - List deleted IDs
- `update_meeting()` - Update title/attendees

**Workspaces**:
- `list_workspaces()` - Get all workspaces
- `create_workspace()` - Create new workspace
- `delete_workspace()` - Delete workspace

---

## Implementation Patterns

### Pattern 1: Tool Definition Template

```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='Display Name',
        readOnlyHint=True,      # False if modifies data
        idempotentHint=True     # False if not retry-safe
    )
)
async def tool_function(
    param1: str,
    param2: int | None = None,
    ctx: Context,              # For logging
) -> ResultType:
    """
    One-sentence summary.
    
    Extended description with implementation details.
    
    Args:
        param1: Description
        param2: Description (default: None)
        ctx: MCP context
        
    Returns:
        ResultType with description
    """
    logger = DualLogger(ctx)
    await logger.info('Operation starting')
    
    try:
        # API call
        response = await _http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Validation with strict Pydantic
        data = ResponseModel.model_validate(response.json())
        
        # Processing and result
        result = ResultType(field=data.value)
        await logger.info('Operation completed')
        return result
        
    except Exception as e:
        await logger.error(f'Failed: {e}')
        raise  # Always propagate errors
```

### Pattern 2: Pydantic Models with Strict Validation

```python
class BaseModel(pydantic.BaseModel):
    """Base model with strict validation."""
    model_config = pydantic.ConfigDict(
        extra='forbid',     # Reject unknown fields (catch API changes)
        strict=True         # No type coercion
    )

# Nested hierarchy for complex data
class PersonName(BaseModel):
    fullName: str
    givenName: str | None = None

class PersonDetails(BaseModel):
    name: PersonName
    jobTitle: str | None = None

class Creator(BaseModel):
    name: str
    email: str
    details: PersonInfo

class GranolaDocument(BaseModel):
    # Required fields (no default)
    id: str
    created_at: str
    
    # Nullable fields
    title: str | None = None
    people: People | None = None
    
    # Always null fields
    chapters: None = None
```

### Pattern 3: Logging (DualLogger)

Logs to both stdout and MCP context:

```python
class DualLogger:
    def __init__(self, ctx: Context):
        self.ctx = ctx
    
    async def info(self, msg: str):
        print(f'[{timestamp}] [INFO] {msg}')   # CLI
        await self.ctx.info(msg)               # MCP Client
```

### Pattern 4: Session-Based Caching

```python
@cached(ttl=None, cache=Cache.MEMORY)
async def _get_documents_cached(
    limit: int, offset: int, list_id: str | None = None
) -> list:
    """Cache with session lifetime (cleared on restart)."""
    response = await _http_client.post(...)
    response.raise_for_status()
    return response.json()
```

### Pattern 5: Async Generator for Pagination

```python
async def list_meetings(...) -> list[MeetingListItem]:
    async def document_generator():
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
        # Client-side filtering applied here
        if should_include(doc):
            results.append(convert_to_result(doc))
        if limit > 0 and len(results) >= limit:
            break
    return results
```

### Pattern 6: Download Tools (Consistent Architecture)

All download tools follow same pattern:
1. Log operation start
2. Fetch document metadata
3. Fetch specific content (panels, transcript, etc.)
4. Convert to Markdown
5. Add header (title, date)
6. Calculate metadata (word count, sections, etc.)
7. Write to temp file
8. Log completion
9. Return result with metadata

---

## Code Style Conventions

### String Formatting
- Single quotes everywhere (enforced by Ruff)
- Example: `title = 'Display Name'`

### Type Hints
- Full type annotations on all functions
- Union types: `str | None`
- Async functions: `async def name(...) -> ReturnType`

### Naming Conventions
- Private functions: `_helper_function()` (single underscore)
- Global state: `_HTTP_CLIENT`, `_temp_dir`
- Tools: `list_meetings()` (snake_case, descriptive)
- Models: `GranolaDocument` (PascalCase)

### Docstrings
Standard format:
```python
"""
One-line summary in imperative mood.

Extended description providing context,
implementation details, and important notes.

Args:
    param1: Description of parameter
    param2: Description (default: value)
    
Returns:
    ReturnType with field descriptions
    
Raises:
    ValueError: When specific condition occurs
"""
```

### Code Organization
1. Imports (future, stdlib, third-party, local)
2. Global state declarations
3. Lifespan/initialization
4. Server instance creation
5. Helper functions (private)
6. MCP tool definitions (public)
7. CLI entry point
8. Main guard

---

## API Integration

### Authentication Pattern

```python
def get_auth_token() -> str:
    """Read from local Granola app storage (~Library/Application Support/Granola/supabase.json)."""
    granola_dir = Path.home() / 'Library' / 'Application Support' / 'Granola'
    supabase_file = granola_dir / 'supabase.json'
    
    with open(supabase_file) as f:
        data = json.load(f)
    
    tokens = json.loads(data['workos_tokens'])
    return tokens['access_token']

def get_auth_headers() -> dict[str, str]:
    """Returns Bearer token in Authorization header."""
    token = get_auth_token()
    return {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
```

### API Request Pattern

```python
headers = get_auth_headers()
url = 'https://api.granola.ai/v2/get-documents'
payload = {'limit': 40, 'offset': 0}

response = await _http_client.post(url, json=payload, headers=headers)
response.raise_for_status()  # Raises on 4xx/5xx

data = DocumentsResponse.model_validate(response.json())  # Strict validation
```

### Granola API Endpoints Used

- `POST https://api.granola.ai/v2/get-documents` - List/search meetings
- `POST https://api.granola.ai/v1/get-document-panels` - AI-generated notes
- `POST https://api.granola.ai/v1/get-document-transcript` - Transcripts
- `POST https://api.granola.ai/v1/get-documents-batch` - Batch retrieval
- `POST https://api.granola.ai/v1/get-document-lists-metadata` - Meeting lists
- `POST https://api.granola.ai/v1/update-document` - Update fields
- `POST https://api.granola.ai/v1/get-workspaces` - Workspaces
- `POST https://api.granola.ai/v2/create-workspace` - Create workspace
- `POST https://api.granola.ai/v1/delete-workspace` - Delete workspace

---

## Error Handling Philosophy

### Fail Fast Pattern
1. All HTTP errors immediately raise (via `response.raise_for_status()`)
2. Validation errors propagate (via Pydantic strict validation)
3. Custom errors raised for missing data
4. No silent failures or error suppression

### Implementation
```python
try:
    response = await _http_client.post(...)
    response.raise_for_status()
    data = ResponseModel.model_validate(response.json())
    
    if not data.docs:
        raise ValueError(f'Not found: {id}')
    
    # Process...
    await logger.info('Success')
    return result
    
except Exception as e:
    await logger.error(f'Operation failed: {e}')
    raise  # Always re-raise
```

---

## Installation & Usage

### Installation Command
```bash
claude mcp add --scope user --transport stdio granola -- uv run --script ~/granola-mcp/granola-mcp.py
```

### Running Locally
```bash
uv run --script ~/granola-mcp/granola-mcp.py
```

### Debug Mode
```bash
uv run --script ~/granola-mcp/granola-mcp.py --debug --debug-host localhost --debug-port 5678
```

### Environment Variables
- `DEBUG_HOST` - Debug host (default: host.docker.internal)
- `DEBUG_PORT` - Debug port (default: 5678)

---

## Key Design Principles

1. **Separation of Concerns**
   - Main server logic in `granola-mcp.py`
   - Data models in `src/models.py`
   - Utilities in `src/helpers.py`
   - Logging abstraction in `src/logging.py`

2. **Resource Lifecycle Management**
   - Lifespan context manager ensures cleanup
   - Shared HTTP client for connection pooling
   - Temp directory auto-cleanup on shutdown

3. **Strict Validation**
   - Pydantic models fail fast on changes
   - No type coercion (strict=True)
   - Unknown fields rejected (extra='forbid')

4. **Fail-Fast Error Handling**
   - All errors propagate immediately
   - Logging before re-raising
   - No silent failures

5. **Comprehensive Logging**
   - Dual output (stdout + MCP context)
   - Timestamps on all messages
   - Per-tool logging for debugging

6. **Async Throughout**
   - All I/O operations async
   - Async context managers for resources
   - Async generators for pagination

7. **Type Safety**
   - Full type hints on all functions
   - Pydantic for API response validation
   - Union types for flexibility

---

## Replication Guidelines

When building a similar MCP server (e.g., claude-session-mcp):

1. Copy project structure (main file + src/ directory)
2. Implement lifespan context manager for resource management
3. Define tools with @mcp.tool decorator and ToolAnnotations
4. Create Pydantic models with strict validation
5. Implement DualLogger for dual output
6. Use async/await throughout
7. Centralize authentication and API logic
8. Include comprehensive docstrings
9. Use single quotes (Ruff format)
10. Add CLI entry point with debug support

