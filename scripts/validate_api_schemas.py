#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0", "anthropic>=0.40.0", "lazy-object-proxy>=1.10.0"]
# ///

"""
Validate API schemas against captured Claude Code traffic.

This script validates mitmproxy captures in captures/ against the
cc_internal_api Pydantic schemas to ensure complete schema coverage.

Usage:
    ./scripts/validate_api_schemas.py
    ./scripts/validate_api_schemas.py --captures-dir /path/to/captures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from src.schemas.cc_internal_api.base import PermissiveModel


@dataclass
class EndpointResult:
    """Validation result for a single endpoint type."""

    endpoint: str
    request_schema: str | None
    response_schema: str | None
    requests_validated: int = 0
    requests_failed: int = 0
    responses_validated: int = 0
    responses_failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Summary of all validation results."""

    total_files: int = 0
    total_validated: int = 0
    total_failed: int = 0
    endpoints: dict[str, EndpointResult] = field(default_factory=dict)
    unmatched_files: list[str] = field(default_factory=list)


def get_endpoint_pattern(filename: str) -> str:
    """Extract endpoint pattern from capture filename."""
    # Remove req_/resp_ prefix and .json suffix
    name = filename.replace('.json', '')
    parts = name.split('_')
    if len(parts) >= 3:
        # Skip sequence number and join host + path
        return '_'.join(parts[2:])
    return name


def load_capture_body(filepath: Path, is_request: bool) -> dict[str, Any] | None:
    """Load and extract body from capture file."""
    with open(filepath) as f:
        data = json.load(f)

    body = data.get('body', {})

    # Handle different body formats
    if isinstance(body, dict):
        # Check for nested data (some captures store body as JSON string)
        if 'data' in body:
            payload = body['data']
            if isinstance(payload, str):
                try:
                    result = json.loads(payload)
                    return dict(result) if isinstance(result, dict) else None
                except json.JSONDecodeError:
                    return None
            return dict(payload) if isinstance(payload, dict) else None
        if 'json' in body:
            json_val = body['json']
            return dict(json_val) if isinstance(json_val, dict) else None
        # SSE responses
        if body.get('type') == 'sse':
            return dict(body)
        return dict(body) if body else None

    return None


def get_schema_for_endpoint(pattern: str, is_request: bool) -> tuple[str | None, type[PermissiveModel] | None]:
    """Get the appropriate schema class for an endpoint pattern."""
    # Import schemas lazily to avoid import errors if schemas have issues
    from src.schemas.cc_internal_api import (
        AccountSettingsResponse,
        ClientDataResponse,
        CountTokensRequest,
        CountTokensResponse,
        EvalRequest,
        EvalResponse,
        GroveResponse,
        HelloResponse,
        MessagesRequest,
        MetricsEnabledResponse,
        ModelAccessResponse,
        ReferralEligibilityResponse,
        StatsigInitializeRequest,
        StatsigRegisterRequest,
        StatsigRegisterResponse,
        TelemetryBatchRequest,
    )

    pattern_lower = pattern.lower()

    # Messages API
    if 'v1_messages' in pattern_lower and 'count_tokens' not in pattern_lower:
        if is_request:
            return 'MessagesRequest', MessagesRequest
        # Responses are SSE, handled separately
        return 'MessagesResponse (SSE)', None

    # Count tokens
    if 'count_tokens' in pattern_lower:
        if is_request:
            return 'CountTokensRequest', CountTokensRequest
        return 'CountTokensResponse', CountTokensResponse

    # Telemetry
    if 'event_logging' in pattern_lower:
        if is_request:
            return 'TelemetryBatchRequest', TelemetryBatchRequest
        return None, None  # Response is just acknowledgment

    # Grove
    if 'grove' in pattern_lower:
        if is_request:
            return None, None
        return 'GroveResponse', GroveResponse

    # Metrics enabled
    if 'metrics_enabled' in pattern_lower:
        if is_request:
            return None, None
        return 'MetricsEnabledResponse', MetricsEnabledResponse

    # Feature flags (eval/sdk)
    if 'eval_sdk' in pattern_lower or 'api_eval' in pattern_lower:
        if is_request:
            return 'EvalRequest', EvalRequest
        return 'EvalResponse', EvalResponse

    # Account settings
    if 'account_settings' in pattern_lower:
        if is_request:
            return None, None
        return 'AccountSettingsResponse', AccountSettingsResponse

    # Client data
    if 'client_data' in pattern_lower:
        if is_request:
            return None, None
        return 'ClientDataResponse', ClientDataResponse

    # Hello
    if 'api_hello' in pattern_lower:
        if is_request:
            return None, None
        return 'HelloResponse', HelloResponse

    # Model access (organization claude_c* which is sonnet_1m_access truncated)
    if 'api_organization' in pattern_lower and 'claude_c' in pattern_lower:
        if is_request:
            return None, None
        return 'ModelAccessResponse', ModelAccessResponse

    # Referral (oauth/organizations/.../referral)
    if 'oauth_organizations' in pattern_lower and 'referral' not in pattern_lower:
        # This is the referral eligibility endpoint (truncated to _r in filename)
        if is_request:
            return None, None
        return 'ReferralEligibilityResponse', ReferralEligibilityResponse

    if 'referral' in pattern_lower:
        if is_request:
            return None, None
        return 'ReferralEligibilityResponse', ReferralEligibilityResponse

    # Statsig
    if 'statsig' in pattern_lower:
        if 'initialize' in pattern_lower:
            if is_request:
                return 'StatsigInitializeRequest', StatsigInitializeRequest
            # Response is a union type (empty or full), skip direct validation
            return None, None
        if 'rgstr' in pattern_lower:
            if is_request:
                return 'StatsigRegisterRequest', StatsigRegisterRequest
            return 'StatsigRegisterResponse', StatsigRegisterResponse

    return None, None


def validate_sse_events(body: dict[str, Any]) -> tuple[int, int, list[str]]:
    """Validate SSE events in a streaming response."""
    from pydantic import TypeAdapter

    from src.schemas.cc_internal_api import SSEEvent

    adapter: TypeAdapter[SSEEvent] = TypeAdapter(SSEEvent)
    events = body.get('events', [])

    validated = 0
    failed = 0
    errors: list[str] = []

    for i, event in enumerate(events):
        parsed = event.get('parsed_data', {})
        if not parsed:
            continue

        try:
            adapter.validate_python(parsed)
            validated += 1
        except ValidationError as e:
            failed += 1
            if len(errors) < 3:
                errors.append(f'Event {i} ({parsed.get("type", "?")}): {e.errors()[0]}')

    return validated, failed, errors


def validate_captures(captures_dir: Path) -> ValidationSummary:
    """Validate all captures in directory."""
    summary = ValidationSummary()

    # Group files by endpoint pattern
    files_by_endpoint: dict[str, list[Path]] = {}
    for f in sorted(captures_dir.glob('*.json')):
        pattern = get_endpoint_pattern(f.name)
        if pattern not in files_by_endpoint:
            files_by_endpoint[pattern] = []
        files_by_endpoint[pattern].append(f)

    for pattern, files in files_by_endpoint.items():
        result = EndpointResult(endpoint=pattern, request_schema=None, response_schema=None)

        for filepath in files:
            summary.total_files += 1
            is_request = filepath.name.startswith('req_')

            # Get schema
            schema_name, schema_cls = get_schema_for_endpoint(pattern, is_request)

            if is_request:
                result.request_schema = schema_name
            else:
                result.response_schema = schema_name

            # Load body
            body = load_capture_body(filepath, is_request)
            if body is None:
                continue

            # Handle SSE responses specially
            if isinstance(body, dict) and body.get('type') == 'sse':
                validated, failed, errors = validate_sse_events(body)
                result.responses_validated += validated
                result.responses_failed += failed
                result.errors.extend(errors)
                summary.total_validated += validated
                summary.total_failed += failed
                continue

            # Validate against schema
            if schema_cls is None:
                if schema_name is None:
                    summary.unmatched_files.append(filepath.name)
                continue

            try:
                schema_cls.model_validate(body)
                if is_request:
                    result.requests_validated += 1
                else:
                    result.responses_validated += 1
                summary.total_validated += 1
            except ValidationError as e:
                if is_request:
                    result.requests_failed += 1
                else:
                    result.responses_failed += 1
                summary.total_failed += 1
                if len(result.errors) < 5:
                    result.errors.append(f'{filepath.name}: {e.errors()[0]}')

        summary.endpoints[pattern] = result

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate API schemas against captures')
    parser.add_argument(
        '--captures-dir',
        type=Path,
        default=Path('captures'),
        help='Directory containing capture files',
    )
    args = parser.parse_args()

    if not args.captures_dir.exists():
        print(f'Captures directory not found: {args.captures_dir}')
        sys.exit(1)

    print('=' * 70)
    print('Claude Code API Schema Validation')
    print('=' * 70)
    print()

    summary = validate_captures(args.captures_dir)

    # Print summary
    print(f'Total files: {summary.total_files}')
    print(f'Total validated: {summary.total_validated}')
    print(f'Total failed: {summary.total_failed}')
    if summary.total_validated + summary.total_failed > 0:
        pct = summary.total_validated / (summary.total_validated + summary.total_failed) * 100
        print(f'Success rate: {pct:.1f}%')
    print()

    # Print endpoint results
    print('ENDPOINT VALIDATION')
    print('-' * 70)

    # Group by validation status
    validated_endpoints = []
    failed_endpoints = []
    unmatched_endpoints = []

    for pattern, result in sorted(summary.endpoints.items()):
        total = (
            result.requests_validated + result.requests_failed + result.responses_validated + result.responses_failed
        )
        if total == 0:
            unmatched_endpoints.append((pattern, result))
        elif result.requests_failed + result.responses_failed > 0:
            failed_endpoints.append((pattern, result))
        else:
            validated_endpoints.append((pattern, result))

    # Show validated first
    if validated_endpoints:
        print('\n✓ VALIDATED:')
        for pattern, result in validated_endpoints:
            req_info = f'req:{result.requests_validated}' if result.requests_validated else ''
            resp_info = f'resp:{result.responses_validated}' if result.responses_validated else ''
            info = ', '.join(filter(None, [req_info, resp_info]))
            print(f'  {pattern[:50]:<50} ({info})')

    # Show failed
    if failed_endpoints:
        print('\n✗ FAILED:')
        for pattern, result in failed_endpoints:
            print(f'  {pattern[:50]:<50}')
            print(f'    Requests: {result.requests_validated} ok, {result.requests_failed} failed')
            print(f'    Responses: {result.responses_validated} ok, {result.responses_failed} failed')
            for error in result.errors[:2]:
                print(f'    Error: {error[:80]}')

    # Show unmatched
    if unmatched_endpoints:
        print('\n? UNMATCHED (no schema or empty):')
        for pattern, result in unmatched_endpoints[:10]:
            print(f'  {pattern[:60]}')

    print()

    # Exit code
    if summary.total_failed == 0:
        print('✓ All schemas validated successfully!')
        sys.exit(0)
    else:
        print(f'✗ {summary.total_failed} validations failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
