# Capture Schema System

This directory contains Pydantic v2 models for capturing, validating, and analyzing
HTTP traffic from the Claude Code observability platform.

## Philosophy: Maximum Strictness

This codebase follows a **maximum strictness** philosophy for type safety.
The goal is complete observability with zero escape hatches.

### Core Principles

1. **FULLY SPECIFIED** - Every field has an exact type, no escape hatches
2. **FULL DISCRIMINATION** - Different shapes = different classes (even 1000+ classes)
3. **NO LAZY DEFAULTS** - Bifurcate types instead of `| None = None`
4. **FAIL-FAST EVERYWHERE** - Unknown data = immediate validation error
5. **LINTING ENFORCED** - Scripts prevent regression to loose patterns

### Banned Patterns

| Pattern                  | Why It's Wrong      | Replacement                          |
|--------------------------|---------------------|--------------------------------------|
| Pure `Mapping[str, Any]` | Hides structure     | Typed + Fallback pattern (see below) |
| `Sequence[Any]`          | Hides element types | Typed elements                       |
| `                        | None = None`        | Lazy optionality                     | Bifurcate into separate types |
| `Any` anywhere           | Total escape hatch  | Never use (except validators)        |
| `dict[str, Any]`         | Mutable + untyped   | `Mapping` + typed values             |

Note: `Mapping[str, Any]` is acceptable **only** as a fallback in the Typed + Fallback
pattern (see below), not as a primary type annotation.

### The Bifurcation Pattern

When a field is "sometimes present", create separate types:

```python
# BAD - Lazy optionality
class Record(StrictModel):
    maybe_field: str | None = None

# GOOD - Bifurcated types
class RecordWithField(StrictModel):
    field: str

class RecordWithoutField(StrictModel):
    pass

Record = RecordWithField | RecordWithoutField
```

### Exceptions: The noqa Pattern

When a loose pattern is genuinely necessary (e.g., JSON Schema meta-schema),
add an explicit noqa comment with a reason:

```python
# noqa: loose-typing - JSON Schema meta-schema; typing would require typing JSON Schema itself
properties: Mapping[str, Any]
```

### The Typed + Fallback Pattern

For **extensible APIs** where unknown data can appear, use a discriminated union
with typed models for known patterns and a **PermissiveModel subclass** as fallback.

#### The PermissiveModel / StrictModel Symmetry

```python
# In src/schemas/types.py - Foundation types
class BaseStrictModel(BaseModel):
    model_config = {'extra': 'forbid'}  # Rejects unknown fields

class PermissiveModel(BaseModel):
    model_config = {'extra': 'allow'}   # Accepts unknown fields
```

| Model             | Behavior               | Use Case                        |
|-------------------|------------------------|---------------------------------|
| `StrictModel`     | Rejects unknown fields | Known, fixed structures         |
| `PermissiveModel` | Accepts unknown fields | Fallback for unknown structures |

#### Creating Domain-Specific Fallbacks

Subclass names describe the **domain**, not the mechanism (inheritance conveys that):

```python
# In statsig.py - domain-specific fallback
class UnknownConfigValue(PermissiveModel):
    """Unknown Statsig config value structure."""
    pass

# In segment.py - domain-specific fallbacks
class UnknownSegmentTraits(PermissiveModel):
    """Unknown Segment event traits."""
    pass

# Usage in union - NO NOQA NEEDED! It's a proper type.
DynamicConfigValue = Annotated[
    FeedbackTimingConfigValue
    | EnabledConfigValue
    | VariantConfigValue
    | UnknownConfigValue,  # Proper type, detectable via isinstance()
    pydantic.Field(union_mode='left_to_right'),
]
```

#### Detection and Observability

Because `PermissiveModel` subclasses are proper types, detection is trivial:

```python
# In validate_captures.py
if isinstance(config.value, PermissiveModel):
    logger.warning(f"Unknown structure: {config.value.get_structure().keys()}")
```

This enables tracking fallback usage without hard failures.

#### Key Points

1. **Type what you observe** - Create models for EVERY observed structure
2. **Subclass names describe domain** - `UnknownConfigValue`, not `PermissiveConfigValue`
3. **No noqa needed** - `PermissiveModel` subclasses are proper types!
4. **Detection via isinstance()** - `isinstance(x, PermissiveModel)` catches all fallbacks
5. **Order matters** - Put specific types first, fallback last (union_mode='left_to_right')

#### Typed + Fallback vs True Unknown Captures

There are TWO different patterns - don't confuse them:

| Pattern              | Location               | Body Type                            | noqa? | Goal                            |
|----------------------|------------------------|--------------------------------------|-------|---------------------------------|
| **Typed + Fallback** | statsig.py, segment.py | `PermissiveModel` subclass           | NO    | Type safety + graceful fallback |
| **True Unknown**     | gcs.py                 | `Mapping[str, Any] \| Sequence[Any]` | YES   | Catch unmapped endpoints        |

**Typed + Fallback**: We have typed models, need fallback for unknown structures.
The fallback is a proper type (`UnknownConfigValue`), detectable, no noqa needed.

**True Unknown** (`UnknownRequestCapture`): Entire endpoint is unmapped. Body can
be dict OR list. Uses `Mapping[str, Any]` with noqa. Goal is to shrink as we type
more endpoints.

#### Anti-pattern: Premature Type Deletion

Just as we don't prematurely catch exceptions (see CLAUDE.md), we don't
prematurely delete types. If a typed pattern seems "not useful", the correct
response is to **keep it** unless there's evidence it's wrong. Future patterns
might reuse it, and the maintenance cost is near-zero.

#### When to Use Typed + Fallback

- API is explicitly designed to be extensible (e.g., Statsig dynamic configs)
- New variants can appear at any time
- You have observed examples to type
- You want detection/tracking of unknown structures

#### When NOT to Use

- API has a fixed schema (use strict typing only)
- You haven't observed any data yet (capture first, then type)
- Entire endpoint is unmapped (use `UnknownRequestCapture` in gcs.py)

## Adding New Capture Types

1. Examine actual capture data in `captures/<session>/`
2. Create typed model for EACH observed shape
3. Use discriminated unions for variants
4. Bifurcate instead of optional defaults
5. Add to registry in `registry.py`
6. Run: `./scripts/validate_captures.py`
7. Run: `./scripts/check_schema_typing.py` (mutable + loose typing)

## Validation Scripts

- `validate_captures.py` - Validates all captures against schemas
- `check_schema_typing.py` - Enforces immutable types AND strict typing patterns

## Module Structure

| Module         | Contents                                                                  |
|----------------|---------------------------------------------------------------------------|
| `base.py`      | `RequestCapture`, `ResponseCapture`, service bases, connection metadata   |
| `anthropic.py` | Anthropic API: Messages, Telemetry, CountTokens, Eval, internal endpoints |
| `statsig.py`   | Statsig feature flags: Register, Initialize                               |
| `datadog.py`   | Datadog telemetry: log ingestion                                          |
| `segment.py`   | Segment analytics: discriminated union of 6 event types                   |
| `external.py`  | Other services: OAuth, domain checks, documentation                       |
| `gcs.py`       | GCS version check, fallback/unknown captures                              |
| `registry.py`  | `CAPTURE_REGISTRY`, `get_capture_type()`, path normalization              |
| `loader.py`    | `CapturedTraffic` union, `load_capture()`, preprocessing                  |
| `__init__.py`  | Public API exports                                                        |

### Module Organization Rationale

Modules are organized by **service host**, not by function:

- **`anthropic.py`** - `api.anthropic.com` (~38 classes)
- **`statsig.py`** - `statsig.anthropic.com` (~4 classes)
- **`segment.py`** - `api.segment.io` (~18 classes) - discriminated union of event types
- **`datadog.py`** - `http-intake.logs.*.datadoghq.com` (~7 classes)
- **`external.py`** - Multiple low-volume services (~15 classes):
  - `console.anthropic.com` - OAuth token exchange
  - `claude.ai` - Domain info checks
  - `code.claude.com`, `platform.claude.com` - Documentation fetches

Services get their own module when they have sufficient volume and complexity.
`external.py` aggregates services with few endpoints (1-2 each) that don't
warrant dedicated files. If a service grows, it can be extracted to its own module.

## Architecture

### Level 1: Decomposed Request/Response Bases

HTTP requests and responses are fundamentally different entities. Using a
shared base class with `status_code: int | None` loses type precision.

- `RequestCapture` - Base for all HTTP requests (no status_code field)
- `ResponseCapture` - Base for all HTTP responses (status_code is `int`, not `int | None`)

### Level 2: Service-Specific Bases

Each external service gets its own base with `host` constrained by `Literal`:

- `AnthropicRequestCapture` / `AnthropicResponseCapture` - `host: Literal['api.anthropic.com']`
- `StatsigRequestCapture` / `StatsigResponseCapture` - `host: Literal['statsig.anthropic.com']`

### Level 3: Endpoint-Specific Captures

Each API endpoint gets its own capture class with typed body:

```python
class MessagesRequestCapture(AnthropicRequestCapture):
    method: Literal['POST']
    body: MessagesRequest

class MessagesStreamResponseCapture(AnthropicResponseCapture):
    events: Sequence[SSEEvent]
```

### Callable Discriminator

We use a callable discriminator (`get_capture_type()`) rather than a simple
field-based discriminator because:

1. Type dispatch depends on MULTIPLE fields: host, path, direction
2. Messages responses need body.type inspection (SSE vs JSON)
3. Pydantic's simple `Discriminator('field')` only supports single-field lookup

The callable discriminator maintains clean separation between capture (Layer 1)
and validation (Layer 2).

## Design Decisions

### Why Not Inject a `capture_type` Field?

The intercept script (`scripts/intercept_traffic.py`) is a "dumb" memorializer.
It captures raw HTTP faithfully without semantic interpretation. All type/schema
logic belongs here in the validation layer. This keeps:

- Separation of concerns clear
- Capture format stable while interpretation evolves
- New capture types requiring only schema changes

### Why Decomposed Bases?

The ~12 duplicated fields between `RequestCapture` and `ResponseCapture` are
acceptable because:

- Each base is self-contained and readable
- No hidden inheritance to trace
- Type precision: `status_code` is `int` on responses, not `int | None`
- Fail-fast: Missing `status_code` on response fails immediately
