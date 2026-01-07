"""
Capture type registry and discriminator function.

This module provides the type dispatch logic for the CapturedTraffic
discriminated union. It maps HTTP context (host, path, direction) to
specific capture types.

Design Decision: Callable Discriminator for Type Dispatch

We use a callable discriminator (`get_capture_type()`) rather than a simple
field-based discriminator. This is a deliberate architectural choice:

WHY NOT a simple field discriminator?
- Type dispatch depends on MULTIPLE fields: host, path, direction
- For messages responses, we also inspect body.type (SSE vs JSON)
- Pydantic's simple Discriminator('field') only supports single-field lookup

WHY NOT inject a `capture_type` field in intercept_traffic.py?
- Separation of concerns: intercept_traffic.py is a "dumb" memorializer
- It captures raw HTTP faithfully without semantic interpretation
- All type/schema logic belongs here in the validation layer
- Adding new capture types should only require changes to this file
- Keeps the capture format stable while interpretation can evolve

The callable discriminator is the RIGHT place for this logic - it maintains
clean separation between Layer 1 (capture) and Layer 2 (validation).
"""

from __future__ import annotations

import re
from typing import Any

# Mapping from (host, path_pattern, direction) to discriminator tag
# Path patterns are normalized (no query strings, SDK variants normalized)
# Note: Messages responses require body.type inspection, handled in get_capture_type()
CAPTURE_REGISTRY: dict[tuple[str, str, str], str] = {
    # Anthropic API - Messages (request only; response needs body.type inspection)
    ('api.anthropic.com', '/v1/messages', 'request'): 'messages_request',
    # Anthropic API - Telemetry
    ('api.anthropic.com', '/api/event_logging/batch', 'request'): 'telemetry_request',
    ('api.anthropic.com', '/api/event_logging/batch', 'response'): 'telemetry_response',
    # Anthropic API - Count Tokens
    ('api.anthropic.com', '/v1/messages/count_tokens', 'request'): 'count_tokens_request',
    ('api.anthropic.com', '/v1/messages/count_tokens', 'response'): 'count_tokens_response',
    # Anthropic API - Feature Flags (Eval)
    ('api.anthropic.com', '/api/eval/sdk', 'request'): 'eval_request',
    ('api.anthropic.com', '/api/eval/sdk', 'response'): 'eval_response',
    # Anthropic API - Metrics
    ('api.anthropic.com', '/api/claude_code/metrics', 'request'): 'metrics_request',
    ('api.anthropic.com', '/api/claude_code/metrics', 'response'): 'metrics_response',
    ('api.anthropic.com', '/api/claude_code/organizations/metrics_enabled', 'request'): 'metrics_enabled_request',
    ('api.anthropic.com', '/api/claude_code/organizations/metrics_enabled', 'response'): 'metrics_enabled_response',
    # Anthropic API - Health
    ('api.anthropic.com', '/api/hello', 'request'): 'hello_request',
    ('api.anthropic.com', '/api/hello', 'response'): 'hello_response',
    # Anthropic API - Grove
    ('api.anthropic.com', '/api/claude_code_grove', 'request'): 'grove_request',
    ('api.anthropic.com', '/api/claude_code_grove', 'response'): 'grove_response',
    # Anthropic API - Settings
    ('api.anthropic.com', '/api/claude_code/settings', 'request'): 'settings_request',
    ('api.anthropic.com', '/api/claude_code/settings', 'response'): 'settings_response',
    # Anthropic API - OAuth endpoints
    ('api.anthropic.com', '/api/oauth/claude_cli/client_data', 'request'): 'client_data_request',
    ('api.anthropic.com', '/api/oauth/claude_cli/client_data', 'response'): 'client_data_response',
    ('api.anthropic.com', '/api/oauth/profile', 'request'): 'profile_request',
    ('api.anthropic.com', '/api/oauth/profile', 'response'): 'profile_response',
    ('api.anthropic.com', '/api/oauth/claude_cli/roles', 'request'): 'roles_request',
    ('api.anthropic.com', '/api/oauth/claude_cli/roles', 'response'): 'roles_response',
    ('api.anthropic.com', '/api/oauth/account/settings', 'request'): 'account_settings_request',
    ('api.anthropic.com', '/api/oauth/account/settings', 'response'): 'account_settings_response',
    ('api.anthropic.com', '/api/oauth/claude_cli/create_api_key', 'request'): 'create_api_key_request',
    ('api.anthropic.com', '/api/oauth/claude_cli/create_api_key', 'response'): 'create_api_key_response',
    # Anthropic API - Referral endpoints (path normalized to remove UUID)
    ('api.anthropic.com', '/api/oauth/organizations/referral/eligibility', 'request'): 'referral_eligibility_request',
    ('api.anthropic.com', '/api/oauth/organizations/referral/eligibility', 'response'): 'referral_eligibility_response',
    ('api.anthropic.com', '/api/oauth/organizations/referral/redemptions', 'request'): 'referral_redemptions_request',
    ('api.anthropic.com', '/api/oauth/organizations/referral/redemptions', 'response'): 'referral_redemptions_response',
    # Anthropic API - Model access (path normalized to remove UUID)
    ('api.anthropic.com', '/api/organization/claude_code_sonnet_1m_access', 'request'): 'model_access_request',
    ('api.anthropic.com', '/api/organization/claude_code_sonnet_1m_access', 'response'): 'model_access_response',
    # Statsig - Register
    ('statsig.anthropic.com', '/v1/rgstr', 'request'): 'statsig_register_request',
    ('statsig.anthropic.com', '/v1/rgstr', 'response'): 'statsig_register_response',
    # Statsig - Initialize
    ('statsig.anthropic.com', '/v1/initialize', 'request'): 'statsig_initialize_request',
    ('statsig.anthropic.com', '/v1/initialize', 'response'): 'statsig_initialize_response',
    # External - Datadog
    ('http-intake.logs.us5.datadoghq.com', '/api/v2/logs', 'request'): 'datadog_request',
    ('http-intake.logs.us5.datadoghq.com', '/api/v2/logs', 'response'): 'datadog_response',
    # External - GCS (version check)
    ('storage.googleapis.com', '/claude-code-dist/latest', 'request'): 'gcs_version_request',
    ('storage.googleapis.com', '/claude-code-dist/latest', 'response'): 'gcs_version_response',
    # External - Console OAuth
    ('console.anthropic.com', '/v1/oauth/token', 'request'): 'oauth_token_request',
    ('console.anthropic.com', '/v1/oauth/token', 'response'): 'oauth_token_response',
    # External - Segment Analytics
    ('api.segment.io', '/v1/batch', 'request'): 'segment_batch_request',
    ('api.segment.io', '/v1/batch', 'response'): 'segment_batch_response',
    # External - Claude.ai Domain Info
    ('claude.ai', '/api/web/domain_info', 'request'): 'domain_info_request',
    ('claude.ai', '/api/web/domain_info', 'response'): 'domain_info_response',
    # External - Documentation Fetches
    ('code.claude.com', '/docs', 'request'): 'code_claude_doc_request',
    ('code.claude.com', '/docs', 'response'): 'code_claude_doc_response',
    ('platform.claude.com', '/docs', 'request'): 'platform_claude_doc_request',
    ('platform.claude.com', '/docs', 'response'): 'platform_claude_doc_response',
    ('platform.claude.com', '/llms.txt', 'request'): 'platform_claude_doc_request',
    ('platform.claude.com', '/llms.txt', 'response'): 'platform_claude_doc_response',
}


def normalize_path(path: str) -> str:
    """
    Normalize path for registry lookup.

    Handles:
    - Query string removal: /path?query=value -> /path
    - SDK variant normalization: /api/eval/sdk-ABC123 -> /api/eval/sdk
    - GCS path normalization: /claude-code-dist-UUID/claude-code-releases/latest -> /claude-code-dist/latest
    - Organization UUID normalization: /api/oauth/organizations/{uuid}/... -> /api/oauth/organizations/...
    - Model access UUID normalization: /api/organization/{uuid}/... -> /api/organization/...
    """
    # Remove query string
    path = path.split('?')[0]

    # Normalize dynamic SDK paths
    path = re.sub(r'/api/eval/sdk-[a-zA-Z0-9]+', '/api/eval/sdk', path)

    # Normalize GCS version check paths (strip UUID and intermediate dir)
    path = re.sub(
        r'/claude-code-dist-[a-f0-9-]+/claude-code-releases/latest',
        '/claude-code-dist/latest',
        path,
    )

    # Normalize organization UUIDs in OAuth paths
    # /api/oauth/organizations/{uuid}/referral/... -> /api/oauth/organizations/referral/...
    path = re.sub(
        r'/api/oauth/organizations/[a-f0-9-]+/',
        '/api/oauth/organizations/',
        path,
    )

    # Normalize organization UUIDs in model access paths
    # /api/organization/{uuid}/claude_code_sonnet_1m_access -> /api/organization/claude_code_sonnet_1m_access
    path = re.sub(
        r'/api/organization/[a-f0-9-]+/',
        '/api/organization/',
        path,
    )

    # Normalize doc fetch paths
    # /docs/en/whatever.md -> /docs
    if path.startswith('/docs/'):
        path = '/docs'

    return path


def extract_endpoint_from_filename(filename: str) -> tuple[str, str]:
    """
    Extract host and path hint from capture filename.

    Filenames are like: req_001_api_anthropic_com_v1_messages_beta_true.json

    Returns (host_hint, path_hint) where:
    - host_hint: 'api.anthropic.com' or 'statsig.anthropic.com' or ''
    - path_hint: '/v1/messages' or '/api/event_logging' or ''
    """
    # Remove prefix and extension
    stem = filename.replace('.json', '')
    parts = stem.split('_', 2)  # ['req', '001', 'api_anthropic_com_...']

    if len(parts) < 3:
        return '', ''

    endpoint_info = parts[2]  # 'api_anthropic_com_v1_messages_beta_true'

    # Detect host
    host = ''
    if endpoint_info.startswith('api_anthropic_com'):
        host = 'api.anthropic.com'
    elif endpoint_info.startswith('statsig_anthropic_com'):
        host = 'statsig.anthropic.com'

    # Try to detect path pattern
    path = ''
    if 'v1_messages' in endpoint_info and 'count_tokens' not in endpoint_info:
        path = '/v1/messages'
    elif 'event_logging' in endpoint_info:
        path = '/api/event_logging/batch'
    elif 'v1_rgstr' in endpoint_info:
        path = '/v1/rgstr'

    return host, path


def get_capture_type(v: Any) -> str:
    """
    Callable discriminator for CapturedTraffic union.

    Reads HTTP context (host, path, direction) from input data
    and returns discriminator tag for union dispatch.

    Special handling:
    - Messages responses: inspects body.type to distinguish SSE vs JSON
    - Other endpoints: uses registry lookup

    Must handle both dict (deserialization) and model instances
    (serialization/re-validation).
    """
    # Extract fields from either dict or model instance
    if isinstance(v, dict):
        host = v.get('host', '')
        raw_path = v.get('path', '')
        direction = v.get('direction', '')
        events = v.get('events')
    else:
        # During serialization, input is model instance
        host = getattr(v, 'host', '')
        raw_path = getattr(v, 'path', '')
        direction = getattr(v, 'direction', '')
        events = getattr(v, 'events', None)

    # Normalize path for lookup
    path = normalize_path(raw_path)

    # Special handling for Messages API responses - need to distinguish SSE vs JSON
    # After _preprocess_capture():
    #   - SSE responses: body deleted, events populated from body.events
    #   - JSON responses: body.data extracted to body field
    # Discriminate based on presence of events field (set by preprocessing)
    if host == 'api.anthropic.com' and path == '/v1/messages' and direction == 'response':
        if events is not None:
            return 'messages_stream_response'
        return 'messages_json_response'

    # Registry lookup: (host, path, direction)
    key = (host, path, direction)
    if key in CAPTURE_REGISTRY:
        return CAPTURE_REGISTRY[key]

    # Fallback for unknown endpoints - dispatch based on direction
    direction_to_tag = {
        'request': 'unknown_request',
        'response': 'unknown_response',
        'error': 'proxy_error',
    }
    return direction_to_tag.get(direction, 'unknown_request')
