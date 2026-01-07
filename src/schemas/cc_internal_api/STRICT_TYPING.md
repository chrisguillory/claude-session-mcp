# Maximum Strictness: A Philosophy of Type System Design

This document establishes the typing philosophy for this codebase and provides operational guidance for maintaining type system integrity. It addresses both the "why" (cognitive and architectural foundations) and the "how" (specific patterns and anti-patterns).

## The Central Insight: Where Complexity Lives

Every type system makes a choice about where complexity resides:

**Loose typing** distributes complexity to every consumption site. Each function that receives a loosely-typed value must:
- Check for None before using optional fields
- Validate that the shape matches expectations
- Handle cases that "shouldn't happen but might"
- Maintain mental models of which field combinations are valid

**Strict typing** concentrates complexity at the definition site. The type definition absorbs the work of modeling all valid shapes, and every consumption site receives:
- Values that are guaranteed to have the expected structure
- Compile-time or validation-time errors for invalid data
- Self-documenting contracts about what's possible

This is not a symmetric trade-off. A type is defined **once** and consumed **thousands of times**. Front-loading complexity at the definition site pays dividends at every consumption site thereafter.

## The Cognitive Science Foundation

Research on cognitive load in programming reveals why strict typing improves developer experience despite appearing more complex upfront.

### Working Memory and Type Systems

Human working memory is severely limited—we can hold roughly 4-7 chunks of information simultaneously. When consuming a loosely-typed value, developers must:

1. Remember which fields might be None
2. Track which field combinations are valid together
3. Mentally simulate what happens when optional fields are absent
4. Maintain awareness of edge cases throughout the code path

Each of these consumes working memory capacity that could otherwise be applied to the actual problem. This is **extraneous cognitive load**—effort spent understanding the tool rather than solving the problem.

Discriminated unions and bifurcated types eliminate this burden:

```python
# Loose typing: Developer must remember budget_tokens is only valid when enabled
class ThinkingConfig(StrictModel):
    type: Literal['enabled', 'disabled']
    budget_tokens: int | None = None

def configure_thinking(config: ThinkingConfig):
    if config.type == 'enabled':
        if config.budget_tokens is not None:  # Must check - type system doesn't help
            # Use budget_tokens
        else:
            # What do we do here? Invalid state the type system allowed
            pass

# Strict typing: Type system encodes the valid states
class EnabledThinking(StrictModel):
    type: Literal['enabled']
    budget_tokens: int  # Always present when enabled

class DisabledThinking(StrictModel):
    type: Literal['disabled']

ThinkingConfig = EnabledThinking | DisabledThinking

def configure_thinking(config: ThinkingConfig):
    match config:
        case EnabledThinking():
            # budget_tokens is guaranteed present - no check needed
            use_budget(config.budget_tokens)
        case DisabledThinking():
            # budget_tokens doesn't exist - can't accidentally use it
            pass
```

The strict version doesn't require the developer to remember anything—the type system makes invalid states unrepresentable.

### The Paradox of Apparent Complexity

Developers initially perceive discriminated unions as "more complex" because there are more type definitions to read. This perception inverts with experience:

- **Day 1**: "Why are there five classes instead of one? This is over-engineered."
- **Day 30**: "I can pattern match on the type and know exactly what fields exist. The code practically writes itself."

The initial complexity is **visible but bounded**—it exists in the type definitions and nowhere else. The alternative complexity is **invisible but pervasive**—it lurks in every function that consumes the type, in every code review that must verify None-checks, in every runtime error from invalid field combinations.

### Transferring Burden to the Machine

The fundamental value proposition of type systems is transferring verification burden from human minds to automated checkers. Loose typing partially defeats this transfer:

- The type checker verifies that `budget_tokens` is `int | None`
- But humans must verify that it's only accessed when `type == 'enabled'`

Strict typing completes the transfer:

- The type checker verifies that `EnabledThinking.budget_tokens` is `int`
- No human verification required—invalid access is a compile-time error

This is not merely convenience. Human verification is fallible, inconsistent, and doesn't scale. Machine verification is deterministic, tireless, and improves as the codebase grows.

## Infrastructure That Compounds

Strict typing is infrastructure—a one-time investment that yields returns on every subsequent interaction with the codebase.

### The Compounding Effect

Consider the lifecycle of a type definition:

1. **Definition cost**: Higher for strict typing (more models, more thought about valid states)
2. **Consumption cost**: Lower for strict typing (no defensive checks, self-documenting)
3. **Modification cost**: Lower for strict typing (type checker catches invalid changes)
4. **Debugging cost**: Lower for strict typing (invalid states caught at boundaries, not deep in logic)

The definition cost is paid once. The consumption, modification, and debugging costs are paid repeatedly—potentially thousands of times over the life of the codebase. This asymmetry means strict typing becomes more valuable over time, not less.

### Enabling Fearless Iteration

Loosely-typed codebases develop a kind of paralysis. Developers become afraid to change types because they can't know what might break. The type checker says the change is fine, but runtime errors might lurk anywhere a None-check was forgotten or a field combination was misunderstood.

Strictly-typed codebases enable confident refactoring. When you modify a discriminated union—add a variant, change a field, remove an option—the type checker identifies every location that needs updating. You can make sweeping changes with confidence that the compiler will catch oversights.

This confidence compounds. Teams with strict typing iterate faster because they're not afraid of their own codebase. They can respond to changing requirements without the fear that accompanies changes in loosely-typed systems.

### The AI Complexity Trade-off

It is genuinely harder to create proper discriminated unions and bifurcated types than to use `Any` or optional fields. This complexity is real and shouldn't be dismissed.

But consider who bears the complexity:

- **Loose typing**: Simple for the author, complex for every consumer forever
- **Strict typing**: Complex for the author once, simple for every consumer forever

When AI assists with type definition, it should absorb the complexity of proper modeling so that humans (and future AI) consuming those types receive simplicity. This is the correct allocation of effort—front-load the thinking at the definition site where it's done once, rather than distributing it to every consumption site where it's done repeatedly.

The AI's job is not to minimize its own effort. It's to create artifacts that minimize total effort across the system's lifetime.

## The Investigation Gap

A dangerous pattern emerges when AI assistants encounter complex-looking data:

```
1. Observe something that appears variable or polymorphic
2. Reach for loose typing (Any, Mapping[str, Any], | None = None)
3. Rationalize the choice ("genuinely polymorphic", "too complex to model")
4. Move on without verification
5. Later discover: the "polymorphism" was discriminable all along
```

This pattern—which we call the **investigation gap**—represents a failure to complete the cognitive work of understanding the data before modeling it. The rationalization feels true because the data genuinely looks complex. But the appearance of complexity usually dissolves under investigation.

### Why the Investigation Gap Occurs

Several factors drive this pattern:

**Path of least resistance**: `Any` and optional fields require less thought than proper discrimination. When facing time pressure or uncertainty, the easy path is tempting.

**Acceptance of surface appearances**: Comments in existing code ("this field can be X or Y") are accepted as constraints without verification. But comments can be wrong, outdated, or imprecise.

**Premature generalization**: Seeing two slightly different shapes triggers the instinct to create a union or optional fields, when investigation might reveal a discriminator that makes them separate types.

**Optimization for immediate completion**: Loose typing "solves" the immediate problem of parsing the data. The costs manifest later, distributed across every consumption site.

### Breaking the Pattern

The investigation gap closes when verification becomes habitual:

1. **Before typing, examine data**: Not specifications or comments, but actual instances from the system
2. **Look for discriminators**: Fields whose values partition the data into distinct groups
3. **Question assumed flexibility**: "Can this really be X or Y, or is there a field that determines which?"
4. **Verify comments against reality**: Existing assertions about polymorphism might be wrong

When investigation becomes routine, the apparent polymorphism that justifies loose typing frequently resolves into discriminable structure.

## Operational Guidelines

### The Verification Checklist

Before using `Any`, `Mapping[str, Any]`, `Sequence[Any]`, or `| None = None`:

- [ ] I have examined 5+ actual instances of this data
- [ ] I have checked every field for potential discriminator behavior
- [ ] I have verified that no field value determines which other fields are present
- [ ] I can articulate exactly why discrimination is impossible, with evidence
- [ ] I have considered whether bifurcation would simplify consumer code

If any checkbox is unchecked, **stop and investigate further**.

### Escalation to Deep Investigation

Some typing decisions genuinely require more investigation than quick examination provides. Indicators that deeper investigation is warranted:

- Large models with many optional fields (suggests under-bifurcation)
- Unions without explicit discriminators (might have hidden discriminators)
- Comments asserting complexity without evidence (might be wrong)
- Fields that are "sometimes present" (might be discriminable by another field)
- Any temptation to use `Any` for structured data

For these cases, dedicate focused investigation time or delegate to a specialized agent that will:
- Gather comprehensive data samples
- Systematically check every field for discriminator behavior
- Map the actual shapes that occur in practice
- Design discriminated unions based on observed patterns

### Anti-Patterns and Their Corrections

| Anti-Pattern                | Why It's Wrong                       | Correction                                     |
|-----------------------------|--------------------------------------|------------------------------------------------|
| `Any`                       | Complete escape from type safety     | Discriminated union of observed shapes         |
| `Mapping[str, Any]`         | Hides structure in plain sight       | TypedDict, StrictModel, or discriminated union |
| `Sequence[Any]`             | Loses element type information       | Typed sequence with discriminated elements     |
| `T \| None = None`          | Distributes None-checking everywhere | Bifurcate into types with/without the field    |
| Union without Discriminator | Forces smart-mode guessing           | Explicit `Discriminator('field')`              |
| Accepting comments as truth | Comments can be wrong                | Verify against actual data                     |

### Empty Value Markers

For values that are always empty, use explicit marker types:

```python
# Always-empty dict: {}
class EmptyDict(StrictModel):
    """With extra='forbid', only {} validates."""
    pass

# Always-empty sequence: []
EmptySequence = Annotated[Sequence[Any], Field(max_length=0)]
```

These markers:
- Document the semantic meaning ("this is always empty")
- Fail validation if the API starts sending data
- Avoid the confusion of typed containers that never contain their type

## The Meta-Lesson

Perceived complexity that justifies loose typing almost always masks incomplete investigation.

When investigation is done properly—examining real data, finding discriminators, understanding the actual shapes—strict typing proves both possible and preferable. The polymorphism that seemed to require `Any` resolves into a discriminated union. The optionality that seemed to require `| None` resolves into bifurcated types.

The instinct toward loose typing is an instinct to avoid work. But the work avoided at the definition site merely transfers to every consumption site, multiplied by the number of consumers and extended across the lifetime of the code.

Strict typing is not perfectionism. It is the recognition that type definitions are leverage points—places where concentrated effort yields distributed benefit. The discipline of maximum strictness is the discipline of investing where returns compound.