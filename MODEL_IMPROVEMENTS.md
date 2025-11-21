# Model Improvements Tracking

Current validation: **99.997%** (33,530 / 33,531 records)
Only failure: 1 corrupted JSON file (can't be fixed with models)

## Summary

This document tracks all remaining improvements to achieve 100% strictly typed models:
1. Replace `str` with `Literal` where values are known
2. Replace `dict[str, Any]` with specific Pydantic models
3. Fix "Unknown structure" comments with proper types

---

## 1. String Fields to Convert to Literals

### High Priority (Known Enum Values)

- [ ] **Line 132**: `StopReason.type: str` → Needs analysis of values
- [ ] **Line 181**: `GlobToolResult.mode: str` → Needs analysis
- [] **Line 191**: `GrepToolResult.mode: str` → Needs analysis
- [ ] **Line 215**: `WriteToolResult.type: str` → Comment says 'create', 'overwrite'
- [ ] **Line 231**: `TaskToolResult.status: str` → Needs analysis
- [ ] **Line 291**: `BaseRecord.type: str` → Already using Literal in subclasses
- [ ] **Line 308**: `UserRecord.userType: str` → Likely 'external', 'internal', etc.
- [ ] **Line 374**: `SystemRecord.systemType: str` → Needs analysis

### Medium Priority (May Have Known Values)

- [ ] **Line 103**: `Message.model: str` → Claude model names (claude-3-5-sonnet, etc.)
- [ ] **Line 105**: `Message.stop_reason: str` → Likely 'end_turn', 'max_tokens', 'stop_sequence'
- [ ] **Line 159**: `BashToolResult.returnCodeInterpretation: str` → Needs analysis
- [ ] **Line 167**: `BashToolResult.status: str` → Likely 'success', 'error', 'interrupted'
- [ ] **Line 341**: `AssistantRecord.model: str` → Same as Message.model

### Low Priority (Likely Free-form)

- [ ] Line 43: `ThinkingContent.thinking` - Free-form text
- [ ] Line 51: `TextContent.text` - Free-form text
- [ ] Line 58-59: IDs (uuid, tool_use_id, etc.) - Free-form
- [ ] Line 155-156: `stdout`/`stderr` - Command output
- [ ] Line 204-206: Edit tool strings - User content
- [ ] Line 217: `WriteToolResult.content` - File content
- [ ] Line 232: `TaskToolResult.prompt` - User prompt

---

## 2. Dict[str, Any] to Replace with Models

### Critical (Data Loss Risk)

- [x] **Line 60**: `ToolUseContent.input` - Tool-specific parameters
  - Contains `file_path` and other tool params
  - **Action**: Create discriminated union of tool input types

- [ ] **Line 107**: `Message.usage` - Token usage in nested API responses
  - Should be `TokenUsage` model

- [ ] **Line 175**: `ReadToolResult.file` - File information
  - Contains: `filePath`, `content`, `numLines`, etc.
  - **Action**: Create `FileInfo` model

- [ ] **Line 209, 218**: `structuredPatch` - Git-style patch data
  - **Action**: Create `StructuredPatch` model

- [ ] **Line 224-225**: `TodoToolResult.oldTodos`/`newTodos` - Todo items
  - **Action**: Create `TodoItem` model with status/content/activeForm

- [ ] **Line 233**: `TaskToolResult.content` - Agent response content
  - Should be `list[MessageContent]`

- [ ] **Line 237**: `TaskToolResult.usage` - Token usage
  - Should be `TokenUsage` model

- [ ] **Line 244-245**: `AskUserQuestionToolResult.questions`/`answers`
  - **Action**: Create `Question` and `Answer` models

- [ ] **Line 252**: `WebSearchToolResult.results` - Search results
  - **Action**: Create `SearchResult` model

- [ ] **Line 314**: `UserRecord.skills` - Claude Code skills
  - **Action**: Create `Skill` model

- [ ] **Line 315**: `UserRecord.mcp` - MCP server configuration
  - **Action**: Create `McpConfig` model

- [ ] **Line 375**: `SystemRecord.message` - Can be string or structured
  - Needs analysis of structured form

- [ ] **Line 413**: `CompactBoundarySystemRecord.compactMetadata`
  - Contains: `trigger`, `preTokens`
  - **Action**: Create `CompactMetadata` model

- [ ] **Line 428**: `ApiErrorSystemRecord.error` - API error details
  - Contains: `status`, `headers`, `requestID`
  - **Action**: Create `ApiError` model

- [ ] **Line 471**: `FileHistorySnapshotRecord.snapshot` - File backup data
  - Contains: `messageId`, `trackedFileBackups`, `timestamp`
  - **Action**: Create `FileHistorySnapshot` model

- [ ] **Line 487**: `QueueOperationRecord.data` - Queue metadata
  - Needs analysis

### Medium Priority (Model Quality)

- [ ] **Line 68**: `ToolResultContent.content` - `list[dict[str, Any]]`
  - Already has string alternative
  - Dict form needs analysis

- [ ] **Line 76**: `ImageContent.source` - Image source data
  - **Action**: Create `ImageSource` model

- [ ] **Line 280**: Fallback `dict[str, Any]` in ToolUseResultUnion
  - Keep as fallback for unknown tools

- [ ] **Line 322**: `toolUseResult` - `list[dict[str, Any]]` variant
  - Needs analysis of list form

---

## 3. "Unknown Structure" Comments to Fix

- [ ] **Line 108**: `Message.container: Any` - "Unknown structure, rarely present"
  - **Action**: Analyze actual examples to create model

- [ ] **Line 109**: `Message.context_management: Any` - "Context management information"
  - **Action**: Analyze actual examples to create model

---

## 4. Implementation Strategy

### Phase 1: Literals (Quick Wins)
1. Analyze actual values for each `str` field
2. Replace with `Literal` types
3. Run validation to confirm

### Phase 2: Critical Dict Replacements
1. Start with `TodoItem`, `CompactMetadata`, `ApiError` (simple structures)
2. Move to `FileInfo`, `StructuredPatch` (medium complexity)
3. Tackle `ToolUseContent.input` discriminated union (complex)

### Phase 3: Unknown Structures
1. Find examples of `container` and `context_management`
2. Create proper models
3. Replace `Any` types

### Phase 4: Final Validation
1. Run full validation suite
2. Ensure still at 99.997%+
3. Document any intentional `dict[str, Any]` fallbacks

---

## Notes

- Keep `dict[str, Any]` fallback in ToolUseResultUnion for unknown tools
- Some `str` fields (IDs, content, output) should stay as strings
- Prioritize fields that appear in PATH translation (file_path, etc.)
- Test after each batch of changes to avoid breaking validation

---

## 5. Reserved Field Pattern (IMPLEMENTED)

**Pattern**: Fields that are present in JSON but ALWAYS null should be typed as `None` (not `Any | None`)

**Rationale**:
- Prevents accidental usage
- Makes the schema more accurate
- Documents that these are placeholder/future fields

**Fields typed as None**:
- `AssistantRecord.stopReason` - Always null (0 non-null across all sessions)
- `Message.container` - Reserved field, always null (present in 73 messages)
- `Message.context_management` - Reserved field, always null (present in 413 messages)
- `UserRecord.skills` - Reserved field, always null (0 non-null across all sessions)
- `UserRecord.mcp` - Reserved field, always null (0 non-null across all sessions)
- `QueueOperationRecord.data` - Reserved field, always null (0 non-null across all sessions)

**Documentation**: Added to file-level docstring in models.py explaining this pattern for future reference.

---

## 6. MCP-Only Enforcement for Dict Fallbacks (IMPLEMENTED)

**Requirement**: Only MCP tools (starting with `mcp__`) can use `dict[str, Any]` fallback. All Claude Code built-in tools must be explicitly modeled.

**Implementation**:

1. **MODELED_CLAUDE_TOOLS mapping** (defined at end of models.py):
   ```python
   MODELED_CLAUDE_TOOLS = [
       (BashToolResult, 'Bash'),
       (BashToolResult, 'BashOutput'),
       (ReadToolResult, 'Read'),
       # ... all 15 Claude tools
   ]
   ```
   - Uses list of tuples (not dict) to allow one-to-many mappings
   - Couples tool names directly to model classes (not disconnected strings)

2. **ALLOWED_CLAUDE_TOOL_NAMES** (derived from mapping):
   ```python
   ALLOWED_CLAUDE_TOOL_NAMES = {tool_name for _, tool_name in MODELED_CLAUDE_TOOLS}
   ```
   - Late binding: Defined at end of file after all classes exist
   - Python's late binding allows ToolUseContent validator to reference it

3. **field_validator on ToolUseContent.input**:
   - Checks if input is plain dict (not typed Read/Write/Edit model)
   - Enforces tool name must be in ALLOWED_CLAUDE_TOOL_NAMES or start with `mcp__`
   - Raises clear ValueError if unmodeled Claude tool found

**Result**:
- All 15 Claude Code built-in tools validated successfully
- 64 MCP tools correctly use dict fallback
- Future Claude tools will cause immediate clear error

---

## 7. Validator Error Surfacing (IMPLEMENTED)

**Problem**: With `union_mode='left_to_right'`, validator errors get buried in 78+ union discrimination errors.

**Solution**: Enhanced validation script to detect and surface validator errors clearly.

**Implementation in validate_models.py**:
```python
except ValidationError as e:
    # Check if this is a validator error about unmodeled tools
    for error in e.errors():
        error_msg = str(error.get('ctx', {}).get('error', ''))
        if 'Unmodeled Claude Code' in error_msg:
            # Surface this error clearly!
            error_msg = f'Line {line_num} ({record_type}): ⚠️  VALIDATOR ERROR: {error_msg}'
            results['errors'].append(error_msg)
            break
```

**Result**: Validator errors now display as:
```
⚠️ VALIDATOR ERROR: Unmodeled Claude Code built-in tool: 'FakeTool'.
All Claude Code tools must be explicitly modeled (see MODELED_CLAUDE_TOOLS).
Only MCP tools (starting with 'mcp__') may use the dict fallback.
```

Instead of being buried in union errors.

---

## 8. Final Validation Results (COMPLETED)

**Validation Status**: ✅ **100% SUCCESS**

**Statistics**:
- Total files: 398
- Total records: 34,826
- Valid records: 34,826 (100.0%)
- Invalid records: 0 (0.0%)

**Record Type Breakdown**:
- assistant: 19,315
- user: 14,096
- file-history-snapshot: 576
- system: 373
- summary: 324
- queue-operation: 133

**Models Created**: 18 new Pydantic models
**Literal Conversions**: 13 string fields converted
**Reserved Fields**: 6 fields typed as None
**Discriminated Unions**: 4 unions created
**Any Usage**: Zero (completely eliminated from non-fallback paths)
**Dict Fallbacks**: Only 2 (both for MCP tools, both documented and enforced)

**Claude Code Tools Modeled**: All 15 built-in tools
- Bash, BashOutput, Read, Write, Edit, Grep, Glob, TodoWrite, Task, WebSearch, WebFetch, AskUserQuestion, ExitPlanMode, ListMcpResourcesTool, KillShell

**MCP Tools**: 64 different tools correctly use dict fallback
