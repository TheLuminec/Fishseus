# Config Module

Stores prompt and runtime configuration assets used by higher-level services.

## Files
- `personality_prompt.txt` – system personality/instruction prompt loaded by `AssistantService`.

## How it is used
- `AssistantService` reads this file at startup and injects it into LLM system context.
- Adjusting this file changes assistant tone/behavior without code changes.
