# MCPSearch: Claude Code's Dynamic Tool Discovery

**Status:** Active since Claude Code v2.0.73 (Dec 18, 2025), default-on in v2.1.x
**Documentation:** Undocumented - not in official CHANGELOG.md
**Last Updated:** January 11, 2026

## What Is MCPSearch?

MCPSearch is Claude Code's implementation of Anthropic's "Tool Search Tool" - a feature that dynamically discovers MCP tools on-demand instead of loading all tool definitions into context upfront.

**Traditional approach:**
- All MCP tools loaded at session start
- 50+ tools = ~72,000 tokens consumed before any work begins
- Overwhelms Claude's context window

**With MCPSearch:**
- Tools marked as `defer_loading: true`
- Claude searches for tools when needed
- Only matched tools get loaded
- ~85% token reduction in large setups

## Timeline

| Version | Date | Status |
|---------|------|--------|
| v2.0.70 | Early Dec 2025 | Added as "WIP – unusable" |
| **v2.0.73** | **Dec 18, 2025** | **Became active** - Users started seeing MCPSearch invocations |
| v2.1.x | Jan 2026 | Default-on for all users |

**Discovery:** Identified through system prompt analysis by [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts)

## How It Works

### Search Syntax

**Keyword search:**
```json
{"name": "MCPSearch", "input": {"query": "database", "max_results": 5}}
```
Returns up to 5 tools with "database" in name/description.

**Exact selection (select: prefix):**
```json
{"name": "MCPSearch", "input": {"query": "select:mcp__perplexity__perplexity_research"}}
```
Loads a specific tool by exact name.

### Output Format

Results contain `tool_reference` content blocks:
```json
{
  "type": "tool_result",
  "content": [
    {"type": "tool_reference", "tool_name": "mcp__server__tool_name"}
  ]
}
```

The API then expands these references into full tool definitions for Claude to use.

## When Does It Activate?

**Documented thresholds (for benefit evaluation):**
- More than 10 tools configured
- Tool definitions exceed 10,000 tokens
- Multiple MCP servers installed

**Actual behavior (based on user reports):**
- **Activates with ANY MCP server configured** in v2.1.x
- Even with 5 tools and 942 tokens, MCPSearch has been observed
- Appears to be **always-on by default** regardless of threshold

## Known Issues

### Haiku Model Incompatibility

**GitHub Issues:** [#14918](https://github.com/anthropics/claude-code/issues/14918), [#14863](https://github.com/anthropics/claude-code/issues/14863), [#15015](https://github.com/anthropics/claude-code/issues/15015)

**Problem:** Haiku models don't support `tool_reference` blocks. When Claude Code uses MCPSearch with Haiku, the API returns:
```
'claude-haiku-4-5-20251001' does not support tool_reference blocks.
This feature is only available on Claude Sonnet 4+, Opus 4+, and newer models.
```

**Workaround:** Use Sonnet or Opus models when MCP tools are needed.

**Status:** Anthropic is expected to patch Claude Code to detect model type and avoid MCPSearch on Haiku.

### Tool Discovery Failures

**GitHub Issue:** [#14978](https://github.com/anthropics/claude-code/issues/14978)

**Problem:** MCP tools show as "✓ Connected" but don't appear in MCPSearch results. Claude can't find tools even when they're properly configured.

**Causes:**
- Poor tool naming (non-descriptive names)
- Missing/unclear descriptions
- Search algorithm limitations

### Accuracy Concerns

Independent testing by [Arcade](https://blog.arcade.dev/anthropic-tool-search-4000-tools-test) with 4,027 tools showed:

- **BM25 search:** 64% accuracy (16/25 queries)
- **Regex search:** 56% accuracy (14/25 queries)
- **Overall:** ~60% success rate

**Common failures:**
- "send an email" didn't find `Gmail_SendEmail`
- "post a message to Slack" didn't find `Slack_SendMessage`

**Anthropic's internal claims:** 88.1% accuracy for Opus 4.5 on "MCP evaluations" (methodology not published)

**Comparison:**
- GitHub Copilot's embedding-guided routing: **94.5% Tool Use Coverage**
- Cursor's dynamic discovery: 46.9% token reduction with similar accuracy

## How to Disable MCPSearch

### Environment Variable Method

```bash
# Disable tool search before running Claude Code
export ENABLE_TOOL_SEARCH=false
claude
```

To make this permanent, add to your shell profile:
```bash
# ~/.zshrc or ~/.bashrc
export ENABLE_TOOL_SEARCH=false
```

### Alternative: Remove MCP Servers

If the environment variable doesn't work (server-side forced), removing all MCP servers will prevent MCPSearch activation:

```bash
# List current servers
claude mcp list

# Remove each server
claude mcp remove <server-name>
```

**Note:** This eliminates MCP functionality entirely - use only if MCPSearch is causing critical issues.

## Why You Might Want to Disable It

**Reasons to disable:**
1. **Using Haiku models** - MCPSearch breaks Haiku due to tool_reference incompatibility
2. **Accuracy concerns** - 60% real-world retrieval may be insufficient for critical workflows
3. **Small tool set** - With <10 tools, the overhead may exceed benefits
4. **Debugging** - Isolate whether MCPSearch is causing tool invocation issues

**Reasons to keep enabled:**
1. **Large tool library** - 50+ tools significantly benefits from token savings
2. **Context efficiency** - Frees ~85% of MCP tool context in large setups
3. **Multiple MCP servers** - Prevents context overflow
4. **Opus/Sonnet models** - Works well with supported models

## Comparison with Competitors

| Feature | Anthropic MCPSearch | GitHub Copilot | Cursor |
|---------|---------------------|----------------|--------|
| Accuracy | ~60% (real-world) / 88% (internal) | 94.5% coverage | Not published |
| Token savings | 85% reduction | Not published | 46.9% reduction |
| Activation | Always-on (2.1.x) | Manual tool selection | Automatic |
| Model support | Sonnet 4+, Opus 4+ | All models | All models |
| User control | ENV var only | UI toggle | Automatic |

## Recommendations

### If You Have <10 MCP Tools
Consider disabling MCPSearch:
```bash
export ENABLE_TOOL_SEARCH=false
```

With few tools, the token overhead of loading them all (~1-5K) is negligible compared to MCPSearch's ~477 token cost plus search latency.

### If You Have 10-50 MCP Tools
Test both configurations on your typical workflows. Monitor:
- Tool selection accuracy (does Claude find the right tool?)
- Context usage (`/context` command)
- Subjective quality

### If You Have 50+ MCP Tools
Keep MCPSearch enabled. The token savings (50K+ → ~8K) are substantial and necessary for context efficiency.

## Technical Details

**System prompt overhead:** ~477 tokens for MCPSearch tool definition

**Search variants:**
- Regex-based: `tool_search_tool_regex_20251119`
- BM25-based: `tool_search_tool_bm25_20251119`

Claude Code likely defaults to one of these (implementation not documented).

**Model support:**
- ✅ Claude Opus 4.5
- ✅ Claude Sonnet 4.5
- ❌ Claude Haiku 4.5 (tool_reference blocks unsupported)

## Further Reading

- [Anthropic's Advanced Tool Use announcement](https://www.anthropic.com/engineering/advanced-tool-use)
- [Official Tool Search Tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Arcade's 4,000 tool stress test](https://blog.arcade.dev/anthropic-tool-search-4000-tools-test)
- [GitHub Copilot's embedding-guided approach](https://github.blog/ai-and-ml/github-copilot/how-were-making-github-copilot-smarter-with-fewer-tools/)
- [Cursor's dynamic context discovery](https://cursor.com/blog/dynamic-context-discovery)

## Related GitHub Issues

- [#7328](https://github.com/anthropics/claude-code/issues/7328) - Feature request: MCP tool filtering UI
- [#14918](https://github.com/anthropics/claude-code/issues/14918) - MCPSearch breaks Haiku
- [#14978](https://github.com/anthropics/claude-code/issues/14978) - Tools not accessible via MCPSearch
- [#15015](https://github.com/anthropics/claude-code/issues/15015) - tool_reference blocks fail on Haiku
- [#12836](https://github.com/anthropics/claude-code/issues/12836) - Request for Tool Search beta support

## Contributing

Found issues with MCPSearch or have additional insights? Please contribute:
1. File issues at [anthropics/claude-code](https://github.com/anthropics/claude-code/issues)
2. Share configuration discoveries in community forums
3. Report accuracy problems with specific tool/query combinations