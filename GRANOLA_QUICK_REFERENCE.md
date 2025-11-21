# Granola MCP Server - Quick Reference Guide

## File Structure
```
granola-mcp/
├── granola-mcp.py        Main server (1130 lines)
├── src/
│   ├── models.py         Pydantic models (485 lines)
│   ├── logging.py        DualLogger (31 lines)
│   └── helpers.py        Utilities (278 lines)
└── pyproject.toml        Ruff config
```

## Core Architecture

### 1. Server Setup (granola-mcp.py)
```python
# Global state
_temp_dir: tempfile.TemporaryDirectory | None = None
_http_client: httpx.AsyncClient | None = None

# Lifespan manager
@asynccontextmanager
async def lifespan(server):
    global _temp_dir, _http_client
    # Initialize resources
    try:
        yield {}
    finally:
        # Cleanup resources

# Create server
mcp = FastMCP('granola', lifespan=lifespan)

# Define tools
@mcp.tool(annotations=ToolAnnotations(...))
async def tool_name(...) -> ResultType:
    logger = DualLogger(ctx)
    # Implementation
```

### 2. Pydantic Models (src/models.py)
```python
class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra='forbid',     # Reject unknown fields
        strict=True         # Strict type checking
    )

# Strict validation catches API changes immediately
```

### 3. Logging (src/logging.py)
```python
class DualLogger:
    async def info(self, msg: str):
        print(f'[{timestamp}] [INFO] {msg}')      # stdout
        await self.ctx.info(msg)                  # MCP context
```

### 4. Helpers (src/helpers.py)
- `get_auth_token()` - Read from local Granola storage
- `get_auth_headers()` - Bearer token headers
- `prosemirror_to_markdown()` - Convert JSON to Markdown
- `analyze_markdown_metadata()` - Extract metrics

## Tool Definition Pattern

```python
@mcp.tool(
    annotations=ToolAnnotations(
        title='Display Name',           # Show in Claude Code
        readOnlyHint=True,             # Or False if writes
        idempotentHint=True            # Or False if not retry-safe
    )
)
async def tool_name(
    param1: str,
    param2: int | None = None,
    ctx: Context,                      # For logging
) -> ResultType:
    """
    One-line summary.
    
    Extended description with implementation details.
    
    Args:
        param1: Description
        param2: Description (default: None)
        ctx: MCP context
        
    Returns:
        ResultType with description
    """
    logger = DualLogger(ctx)
    await logger.info('Starting...')
    
    # API call
    response = await _http_client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    # Validate with Pydantic
    data = ResponseModel.model_validate(response.json())
    
    # Return result
    return ResultModel(field1=data.x, field2=data.y)
```

## Key Patterns

### Caching Pattern
```python
@cached(ttl=None, cache=Cache.MEMORY)
async def _get_documents_cached(limit: int, offset: int, list_id: str | None = None) -> list:
    # Session-based caching (TTL=None: persists until server restart)
    # Cache key auto-generated from parameters
    response = await _http_client.post(...)
    return response.json()
```

### Async Generator Pattern
```python
async def list_meetings(...) -> list[MeetingListItem]:
    async def document_generator():
        offset = 0
        while True:
            batch = await _get_documents_cached(limit=40, offset=offset, list_id=list_id)
            if not batch:
                break
            for doc in batch:
                yield doc
            offset += batch_size
    
    results = []
    async for doc in document_generator():
        if should_include(doc):
            results.append(convert_to_result(doc))
        if limit > 0 and len(results) >= limit:
            break
    return results
```

### Download Tools Pattern
```python
async def download_note(document_id: str, filename: str, ctx: Context) -> NoteDownloadResult:
    # 1. Fetch document metadata
    # 2. Fetch specific content
    # 3. Convert to markdown
    # 4. Add header with title/date
    # 5. Calculate metadata
    # 6. Write to file
    # 7. Return result with metadata
```

### Metadata Extraction Pattern
```python
def analyze_markdown_metadata(markdown: str) -> dict:
    lines = markdown.split('\n')
    heading_breakdown = {'h1': 0, 'h2': 0, 'h3': 0}
    section_count = 0
    
    for line in lines:
        if line.startswith('### '):
            heading_breakdown['h3'] += 1
            section_count += 1
        # ... count bullets, words, etc.
    
    return {
        'section_count': section_count,
        'bullet_count': bullet_count,
        'heading_breakdown': heading_breakdown,
        'word_count': word_count,
    }
```

## Pydantic Model Patterns

### Strict Base Model
```python
class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid', strict=True)
```

### Required vs Optional
```python
class Example(BaseModel):
    # Required - no default
    id: str
    created_at: str
    
    # Nullable - has default None
    title: str | None = None
    people: People | None = None
    
    # Always null - type is None
    chapters: None = None
```

### Field Aliases
```python
timestamp_to: str | None = pydantic.Field(
    default=None,
    alias='timestamp-to'  # JSON field name
)
```

### Conditional Field Exclusion
```python
participants: list[ParticipantInfo] | None = pydantic.Field(
    default=None,
    exclude_if=lambda v: v is None
)
```

## Error Handling

### HTTP Errors
```python
response = await _http_client.post(url, json=payload, headers=headers)
response.raise_for_status()  # Raises on 4xx/5xx
```

### Validation Errors
```python
try:
    data = ResponseModel.model_validate(response.json())
except pydantic.ValidationError:
    # Pydantic validation failed - API changed
    raise
```

### Custom Errors
```python
if not doc_data.docs:
    raise ValueError(f'Document {document_id} not found')
```

### Error Pattern
```python
try:
    # API call
    response = await _http_client.post(...)
    response.raise_for_status()
    
    # Validation
    data = ResponseModel.model_validate(response.json())
    
    # Processing
    result = process(data)
    
    await logger.info('Success')
    return result
    
except Exception as e:
    await logger.error(f'Failed: {e}')
    raise  # Always re-raise
```

## Code Style

### String Quotes
Single quotes everywhere (enforced by Ruff):
```python
title = 'User-Visible Title'
```

### Type Hints
All functions have type hints:
```python
async def tool_name(param: str, ctx: Context) -> ResultType:
```

### Docstrings
All public functions have docstrings:
```python
"""
One-line summary.

Extended description.

Args:
    param: Description

Returns:
    ResultType description
"""
```

### Naming
- Private functions: `_helper_function()`
- Global state: `_HTTP_CLIENT` or `_temp_dir`
- Tools: `list_meetings()` (snake_case)
- Models: `MeetingListItem` (PascalCase)

## CLI Entry Point

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Granola MCP Server')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-host', default=os.environ.get('DEBUG_HOST', 'host.docker.internal'))
    parser.add_argument('--debug-port', type=int, default=int(os.environ.get('DEBUG_PORT', '5678')))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace(args.debug_host, port=args.debug_port, ...)
    print('Starting Granola MCP server')
    mcp.run()

if __name__ == '__main__':
    main()
```

## Installation

```bash
claude mcp add --scope user --transport stdio granola -- uv run --script ~/granola-mcp/granola-mcp.py
```

## Dependencies

```
aiocache              # In-memory caching
fastmcp>=2.12.5       # MCP server framework
httpx                 # Async HTTP client
markdownify           # HTML to Markdown
pydantic>=2.0         # Type validation
pydevd-pycharm        # Debug support (optional)
```

## Key Concepts

### Lifespan Management
- Resources initialized at server startup
- Guaranteed cleanup at shutdown
- Used for HTTP client, temp directories, etc.

### Session-based Caching
- In-memory cache persists for server lifetime
- Cache key auto-generated from function parameters
- Cleared on server restart

### Dual Logging
- Print to stdout (for CLI debugging)
- Send to MCP context (for client visibility)
- Timestamps on all messages

### Strict Validation
- Pydantic models fail fast on API changes
- No silent failures or type coercion
- All errors propagate immediately

### Async Throughout
- Async context managers for resources
- Async generators for pagination
- Async/await for all I/O operations

## Testing Workflow

1. Make code changes
2. Reconnect MCP server: `/mcp reconnect granola`
3. Test tools in Claude Code
4. Check both stdout (CLI) and MCP context logs

## Common Modifications

### Add a New Tool
1. Create async function with `@mcp.tool` decorator
2. Add ToolAnnotations with title, readOnlyHint, idempotentHint
3. Include ctx: Context parameter for logging
4. Validate response with Pydantic model
5. Return structured result model

### Add API Endpoint
1. Create Pydantic model for response
2. Use `_http_client` for async HTTP calls
3. Call `get_auth_headers()` for authentication
4. Validate response with model_validate()
5. Return structured result

### Update Models
1. Modify Pydantic models in `src/models.py`
2. Use strict validation (forbid extra fields)
3. Strict types will catch API changes immediately

### Add Helper Function
1. Add to `src/helpers.py`
2. Use type hints and docstrings
3. No async needed unless I/O required
4. Test with sample data
