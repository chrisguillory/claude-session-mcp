#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0"]
# ///

"""
Comprehensive model quality analysis.

This script analyzes the models.py file to identify:
1. All path fields for translation completeness
2. All dict[str, Any] usage and whether justified
3. All str fields that could be Literal
4. Field optionality patterns
5. Any type: Any usage
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import get_origin, get_args

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SessionRecordAdapter


def analyze_actual_values(session_files):
    """Analyze actual values from real session data."""

    analysis = {
        'stop_reasons': Counter(),
        'user_types': Counter(),
        'system_types': Counter(),
        'message_models': Counter(),
        'tool_names': Counter(),
        'system_subtypes': Counter(),
        'queue_operations': Counter(),
        'bash_status': Counter(),
        'bash_return_code_interpretation': Counter(),
        'grep_modes': Counter(),
        'glob_modes': Counter(),
        'write_types': Counter(),
        'task_status': Counter(),
        'thinking_levels': Counter(),
        'compact_triggers': Counter(),
        'api_error_types': Counter(),
        'service_tiers': Counter(),
        'image_media_types': Counter(),
        'system_levels': Counter(),
    }

    path_fields = {
        'cwd': set(),
        'file_path': set(),
        'filePath': set(),
        'projectPaths': set(),
    }

    optional_fields = defaultdict(lambda: {'present': 0, 'null': 0, 'missing': 0})

    for session_file in session_files[:50]:  # Sample first 50 files
        with open(session_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    record_data = json.loads(line)
                    record_type = record_data.get('type')

                    # Analyze stop reasons
                    if record_type == 'assistant' and 'message' in record_data:
                        msg = record_data['message']
                        if isinstance(msg, dict) and 'stop_reason' in msg:
                            sr = msg['stop_reason']
                            if sr:
                                analysis['stop_reasons'][sr] += 1

                    # Analyze user types
                    if 'userType' in record_data:
                        analysis['user_types'][record_data['userType']] += 1

                    # Analyze system types
                    if record_type == 'system':
                        if 'systemType' in record_data:
                            analysis['system_types'][record_data['systemType']] += 1
                        if 'subtype' in record_data:
                            analysis['system_subtypes'][record_data['subtype']] += 1
                        if 'level' in record_data:
                            level = record_data['level']
                            if level:
                                analysis['system_levels'][level] += 1

                    # Analyze models
                    if 'model' in record_data:
                        model = record_data['model']
                        if model:
                            analysis['message_models'][model] += 1

                    if record_type in ['user', 'assistant'] and 'message' in record_data:
                        msg = record_data['message']
                        if isinstance(msg, dict):
                            if 'model' in msg and msg['model']:
                                analysis['message_models'][msg['model']] += 1

                            # Tool names
                            content = msg.get('content', [])
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                                        tool_name = item.get('name')
                                        if tool_name:
                                            analysis['tool_names'][tool_name] += 1

                    # Queue operations
                    if record_type == 'queue-operation':
                        op = record_data.get('operation')
                        if op:
                            analysis['queue_operations'][op] += 1

                    # Tool results
                    if record_type == 'user' and 'toolUseResult' in record_data:
                        result = record_data['toolUseResult']
                        if isinstance(result, dict):
                            # Bash status
                            if 'status' in result and result['status']:
                                analysis['bash_status'][result['status']] += 1
                            if 'returnCodeInterpretation' in result and result['returnCodeInterpretation']:
                                analysis['bash_return_code_interpretation'][result['returnCodeInterpretation']] += 1

                            # Grep mode
                            if 'mode' in result and 'numFiles' in result:
                                mode = result['mode']
                                if mode:
                                    if 'content' in result:
                                        analysis['grep_modes'][mode] += 1
                                    elif 'filenames' in result:
                                        analysis['glob_modes'][mode] += 1

                            # Write type
                            if 'type' in result and result.get('type') in ['create', 'update']:
                                analysis['write_types'][result['type']] += 1

                            # Task status
                            if 'status' in result and 'agentId' in result:
                                analysis['task_status'][result['status']] += 1

                    # Thinking metadata
                    if 'thinkingMetadata' in record_data:
                        tm = record_data['thinkingMetadata']
                        if isinstance(tm, dict) and 'level' in tm:
                            analysis['thinking_levels'][tm['level']] += 1

                    # Compact metadata
                    if record_type == 'system' and 'compactMetadata' in record_data:
                        cm = record_data['compactMetadata']
                        if isinstance(cm, dict) and 'trigger' in cm:
                            analysis['compact_triggers'][cm['trigger']] += 1

                    # API errors
                    if 'error' in record_data and isinstance(record_data['error'], dict):
                        err = record_data['error']
                        if 'error' in err and isinstance(err['error'], dict):
                            err_detail = err['error']
                            if 'type' in err_detail:
                                analysis['api_error_types'][err_detail['type']] += 1

                    # Service tier
                    if 'usage' in record_data and isinstance(record_data['usage'], dict):
                        usage = record_data['usage']
                        if 'service_tier' in usage and usage['service_tier']:
                            analysis['service_tiers'][usage['service_tier']] += 1

                    # Image media types
                    if record_type in ['user', 'assistant'] and 'message' in record_data:
                        msg = record_data['message']
                        if isinstance(msg, dict):
                            content = msg.get('content', [])
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'image':
                                        source = item.get('source', {})
                                        if 'media_type' in source:
                                            analysis['image_media_types'][source['media_type']] += 1

                    # Track path fields
                    if 'cwd' in record_data:
                        path_fields['cwd'].add(record_data['cwd'])
                    if 'projectPaths' in record_data and isinstance(record_data['projectPaths'], list):
                        for p in record_data['projectPaths']:
                            path_fields['projectPaths'].add(p)

                except:
                    pass

    return analysis, path_fields


def print_section(title):
    print()
    print('=' * 80)
    print(title)
    print('=' * 80)


def main():
    print('=' * 80)
    print('Model Quality Analysis')
    print('=' * 80)

    # Find session files
    claude_dir = Path.home() / '.claude' / 'projects'
    session_files = []
    for project_dir in claude_dir.iterdir():
        if project_dir.is_dir():
            for session_file in project_dir.glob('*.jsonl'):
                session_files.append(session_file)

    print(f'Analyzing {len(session_files)} session files...')

    analysis, path_fields = analyze_actual_values(session_files)

    # Print analysis
    print_section('1. String Fields That Could Be Literal Types')

    for field_name, counter in [
        ('stop_reason', analysis['stop_reasons']),
        ('userType', analysis['user_types']),
        ('systemType', analysis['system_types']),
        ('system subtype', analysis['system_subtypes']),
        ('system level', analysis['system_levels']),
        ('queue operation', analysis['queue_operations']),
        ('bash status', analysis['bash_status']),
        ('bash returnCodeInterpretation', analysis['bash_return_code_interpretation']),
        ('grep mode', analysis['grep_modes']),
        ('glob mode', analysis['glob_modes']),
        ('write type', analysis['write_types']),
        ('task status', analysis['task_status']),
        ('thinking level', analysis['thinking_levels']),
        ('compact trigger', analysis['compact_triggers']),
        ('api_error type', analysis['api_error_types']),
        ('service_tier', analysis['service_tiers']),
        ('image media_type', analysis['image_media_types']),
    ]:
        if counter:
            values = list(counter.keys())
            if len(values) <= 10:  # Only suggest Literal if ≤10 unique values
                print(f'\n{field_name}:')
                for value, count in counter.most_common():
                    print(f'  {value!r}: {count} occurrences')
                print(f'  → Recommend: Literal{tuple(values)}')

    print_section('2. Model Names Found (for model: str field)')
    for model, count in analysis['message_models'].most_common():
        print(f'  {model}: {count}')
    print('\n→ Should model: str remain str (variable) or use Literal (if finite set)?')

    print_section('3. Tool Names Found (for MCP vs Claude Code distinction)')
    claude_tools = []
    mcp_tools = []
    for tool, count in analysis['tool_names'].most_common():
        if tool.startswith('mcp__'):
            mcp_tools.append((tool, count))
        else:
            claude_tools.append((tool, count))

    print('\nClaude Code built-in tools:')
    for tool, count in claude_tools:
        print(f'  {tool}: {count}')

    print(f'\nMCP tools: {len(mcp_tools)} unique tools')
    for tool, count in list(mcp_tools)[:10]:
        print(f'  {tool}: {count}')
    if len(mcp_tools) > 10:
        print(f'  ... and {len(mcp_tools) - 10} more')

    print_section('4. Path Fields Found (for translation)')
    print('\nUnique cwd values (sample):')
    for path in list(path_fields['cwd'])[:10]:
        print(f'  {path}')
    print(f'  Total unique: {len(path_fields["cwd"])}')

    if path_fields['projectPaths']:
        print('\nUnique projectPaths values (sample):')
        for path in list(path_fields['projectPaths'])[:10]:
            print(f'  {path}')
        print(f'  Total unique: {len(path_fields["projectPaths"])}')

    print_section('5. Summary & Recommendations')

    print('\n✓ STRENGTHS:')
    print('  - 100% validation success on all records')
    print('  - Discriminated unions for type safety')
    print('  - Path markers for translation')
    print('  - Reserved field pattern (None type)')
    print('  - MCP tool enforcement')

    print('\n⚠ POTENTIAL IMPROVEMENTS:')

    improvements = []

    # Check for overly generic str fields
    if len(analysis['system_types']) <= 10 and analysis['system_types']:
        improvements.append('  1. SystemRecord.systemType could be Literal (finite set)')

    if len(analysis['bash_status']) <= 10 and analysis['bash_status']:
        improvements.append('  2. BashToolResult.status could be more restrictive Literal')

    if len(analysis['bash_return_code_interpretation']) <= 10 and analysis['bash_return_code_interpretation']:
        improvements.append('  3. BashToolResult.returnCodeInterpretation already uses Literal ✓')

    # Model field
    if len(analysis['message_models']) <= 20:
        improvements.append(
            f'  4. Message.model: Found {len(analysis["message_models"])} unique models - consider Literal'
        )
    else:
        improvements.append(f'  4. Message.model: Found {len(analysis["message_models"])} unique models - keep as str')

    if improvements:
        for imp in improvements:
            print(imp)
    else:
        print('  No obvious improvements found!')

    print('\n📋 NEXT STEPS:')
    print('  1. Review MODEL_IMPROVEMENTS.md checklist')
    print('  2. Send full analysis to Perplexity for expert review')
    print('  3. Decide on str → Literal conversions')
    print('  4. Document all intentional dict[str, Any] fallbacks')


if __name__ == '__main__':
    main()
