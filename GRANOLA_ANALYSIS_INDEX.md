# Granola MCP Server Analysis Index

This directory contains comprehensive documentation on the architecture and patterns used in the Granola MCP server at `/Users/chris/granola-mcp/`.

## Documents Overview

### 1. GRANOLA_EXPLORATION_SUMMARY.md
**Quick overview of the Granola MCP server exploration**
- Directory structure and key files
- Architecture overview (5 main components)
- 6 key implementation patterns with code examples
- API integration approach
- Installation and usage commands
- Design principles and replication guidelines

**Use this when**: You need a concise overview before diving into details.

### 2. GRANOLA_QUICK_REFERENCE.md
**Fast lookup reference guide**
- File structure
- Core architecture components (server setup, models, logging, helpers)
- Tool definition pattern template
- 5 key implementation patterns
- Pydantic model patterns
- Error handling patterns
- Code style conventions
- CLI entry point structure
- Common modifications checklist

**Use this when**: You need quick code snippets or examples to copy/paste.

### 3. GRANOLA_ARCHITECTURE_ANALYSIS.md
**Comprehensive deep-dive analysis (11 sections, 30KB)**
- Project overview and structure
- Complete MCP server implementation details
  - Architecture pattern breakdown
  - Lifespan management with code
  - HTTP client global state
  - Caching pattern details
  - Tool definition patterns
  - ToolAnnotations usage
  - All 17 tools categorized
- Pydantic model patterns with 5 layers
- Error handling and logging
- Tool annotations and documentation style
- CLI vs MCP interface patterns
- 7 key implementation patterns
- Dependencies and requirements
- Code style and conventions
- Configuration and environment
- API integration pattern
- Detailed analysis of all concepts

**Use this when**: You need comprehensive understanding of the entire architecture.

---

## Quick Navigation

### For Different Scenarios

**I want to add a new MCP tool:**
1. Read GRANOLA_QUICK_REFERENCE.md > Tool Definition Pattern
2. Check GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 5 (Tool Annotations)
3. Copy pattern from GRANOLA_EXPLORATION_SUMMARY.md > Implementation Patterns

**I want to understand Pydantic models:**
1. Start with GRANOLA_QUICK_REFERENCE.md > Pydantic Model Patterns
2. Read GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 3 (Pydantic Patterns)
3. Review examples in GRANOLA_EXPLORATION_SUMMARY.md

**I want to understand async patterns:**
1. Check GRANOLA_QUICK_REFERENCE.md > Key Patterns (Async Generator, Caching)
2. Read GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 7 (Key Patterns)
3. Review GRANOLA_EXPLORATION_SUMMARY.md > Implementation Patterns 4-6

**I want to replicate the architecture for claude-session-mcp:**
1. Read GRANOLA_EXPLORATION_SUMMARY.md > Replication Guidelines
2. Review GRANOLA_ARCHITECTURE_ANALYSIS.md > All 11 sections sequentially
3. Use GRANOLA_QUICK_REFERENCE.md as coding reference during implementation

**I need specific code snippets:**
1. Search GRANOLA_QUICK_REFERENCE.md for pattern name
2. Find complete examples in GRANOLA_EXPLORATION_SUMMARY.md > Implementation Patterns
3. See detailed code in GRANOLA_ARCHITECTURE_ANALYSIS.md

---

## Key Insights Summary

### Architecture Highlights

1. **Modular Structure**
   - Main server file (granola-mcp.py) + src/ directory
   - Separation: server logic, models, logging, helpers
   - Clean imports and organized code

2. **Resource Lifecycle**
   - Lifespan context manager ensures cleanup
   - Shared HTTP client across all tools
   - Automatic temp directory cleanup

3. **Strict Validation**
   - Pydantic models with `extra='forbid'` and `strict=True`
   - Fail fast on API changes
   - No silent failures

4. **Dual Logging**
   - DualLogger logs to both stdout and MCP context
   - Timestamps on all messages
   - Visibility in both CLI and client

5. **Async Throughout**
   - All I/O operations async
   - Async generators for pagination
   - Async context managers for resources

6. **Tool Pattern**
   - @mcp.tool decorator with ToolAnnotations
   - Context parameter for logging
   - Structured return types (Pydantic models)

7. **Caching Strategy**
   - Session-based in-memory caching
   - Auto cache key from parameters
   - Cleared on server restart

### Code Style

- Single quotes (enforced by Ruff)
- Full type hints on all functions
- Comprehensive docstrings (Args/Returns format)
- Private functions with underscore prefix
- Snake_case for tools, PascalCase for models

### Implementation Patterns

1. **Caching Pattern**: @cached decorator with ttl=None
2. **Async Generator Pattern**: Batch fetching with filtering
3. **Metadata Extraction Pattern**: Analyze markdown for metrics
4. **ProseMirror Conversion Pattern**: Recursive tree traversal
5. **Download Tools Pattern**: Consistent 7-step architecture
6. **Logging Pattern**: DualLogger for dual output

---

## Files in Granola MCP Repository

Located at `/Users/chris/granola-mcp/`:

```
├── granola-mcp.py          (1,130 lines) - Main MCP server
├── src/
│   ├── models.py           (485 lines) - Pydantic models
│   ├── logging.py          (31 lines) - DualLogger utility
│   ├── helpers.py          (278 lines) - Auth, conversion, utilities
│   └── __init__.py
├── pyproject.toml          - Ruff formatting config
├── README.md               - API research
├── CLAUDE.md               - Development guidance
└── .cursorindexingignore
```

---

## Starting Points by Role

### Backend Engineer
1. GRANOLA_EXPLORATION_SUMMARY.md (overall picture)
2. GRANOLA_ARCHITECTURE_ANALYSIS.md sections 2-4 (server, models, logging)
3. GRANOLA_QUICK_REFERENCE.md (coding patterns)

### Full Stack Developer
1. GRANOLA_QUICK_REFERENCE.md (quick overview)
2. GRANOLA_EXPLORATION_SUMMARY.md (implementation patterns)
3. GRANOLA_ARCHITECTURE_ANALYSIS.md (deep dive as needed)

### DevOps/Infrastructure
1. GRANOLA_EXPLORATION_SUMMARY.md > Installation & Usage
2. GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 10 (Configuration)
3. Check pyproject.toml and dependencies

### Protocol/Integration Specialist
1. GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 2 (MCP Server)
2. GRANOLA_ARCHITECTURE_ANALYSIS.md > Section 5 (Tool Annotations)
3. GRANOLA_QUICK_REFERENCE.md > Tool Definition Pattern

---

## Reference Tables

### 17 MCP Tools

**Discovery & Organization (3)**
- list_meetings() - With client-side filtering
- get_meeting_lists() - Get collections
- get_meetings() - Batch retrieval

**Download/Export (3)**
- download_note() - AI-generated notes
- download_transcript() - Meeting transcripts
- download_private_notes() - Private notes

**Management (5)**
- delete_meeting() - Soft delete
- undelete_meeting() - Restore deleted
- list_deleted_meetings() - List deleted
- update_meeting() - Update fields
- (Plus 1 additional)

**Workspaces (3)**
- list_workspaces()
- create_workspace()
- delete_workspace()

### Pydantic Model Hierarchy

1. **BaseModel** - Strict validation (forbid extra, strict types)
2. **Core structures** - PersonName, PersonDetails, Creator
3. **Domain objects** - GranolaDocument, People
4. **API responses** - DocumentsResponse
5. **Tool results** - MeetingListItem, NoteDownloadResult

### API Endpoints (9 total)

All POST to `api.granola.ai`:
- `/v2/get-documents` - List/search
- `/v1/get-document-panels` - Notes
- `/v1/get-document-transcript` - Transcripts
- `/v1/get-documents-batch` - Batch fetch
- `/v1/get-document-lists-metadata` - Lists
- `/v1/update-document` - Updates
- `/v1/get-workspaces` - Workspaces
- `/v2/create-workspace` - Create
- `/v1/delete-workspace` - Delete

---

## How This Analysis Was Generated

This analysis was created through systematic exploration of the `/Users/chris/granola-mcp/` directory:

1. Directory structure mapping
2. File content analysis (all Python files)
3. Architecture pattern identification
4. Code snippet extraction and documentation
5. Cross-referencing between files
6. Consolidation into three comprehensive documents

The analysis covers:
- Project structure and organization
- MCP server implementation patterns
- Pydantic model design
- Error handling approaches
- Logging strategies
- Tool annotation usage
- CLI interface design
- Code style conventions
- API integration patterns
- Caching mechanisms
- Async/await usage

---

## Document Statistics

| Document | Lines | Size | Focus |
|----------|-------|------|-------|
| GRANOLA_EXPLORATION_SUMMARY.md | 380 | 12 KB | Overview, patterns, principles |
| GRANOLA_QUICK_REFERENCE.md | 320 | 9.6 KB | Code snippets, quick lookup |
| GRANOLA_ARCHITECTURE_ANALYSIS.md | 1100 | 32 KB | Deep dive, comprehensive |
| **Total** | **1,800+** | **53.6 KB** | Complete architecture docs |

---

## Next Steps

To apply these patterns to `claude-session-mcp`:

1. Create project structure: `src/` directory with models.py, logging.py, helpers.py
2. Implement lifespan context manager for resource management
3. Create base Pydantic model with strict validation
4. Implement DualLogger for dual output
5. Define MCP tools with @mcp.tool decorator
6. Add CLI entry point with argument parsing
7. Create Pydantic models for API responses
8. Implement helper functions for API calls
9. Add comprehensive docstrings and type hints
10. Test tool execution and logging

Use GRANOLA_QUICK_REFERENCE.md as your coding template throughout.

