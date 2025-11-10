# claude-session-mcp

MCP server for Claude Code session management and cross-machine transfer.

## Models

- **935 lines**, 54 Pydantic classes covering all 6 record types
- **100% validated** on 36,305 real session records
- **Compatible with**: Claude Code 2.0.35 - 2.0.36
- **Round-trip fidelity**: Use `model_dump(exclude_unset=True)` to preserve original JSON structure

### Why Pydantic

**Round-trip serialization**: `exclude_unset=True` preserves null fields from input while excluding never-set defaults—critical for forward compatibility when Claude Code schema evolves.

**Nested discriminated unions**: `Field(union_mode='left_to_right')` handles secondary discriminators (e.g., `type='system'` + `subtype='local_command'`) with explicit ordering.

**Cross-field validators**: `@field_validator` with `info.data` access enables business logic validation (e.g., enforcing only `mcp__*` tools use dict fallback).

**Performance**: 5-10x faster batch validation (36K+ records) via Rust-backed pydantic-core.

**Data pipeline ecosystem**: Python's async I/O, scientific libraries, and MCP SDK integration optimized for session analysis workflows.