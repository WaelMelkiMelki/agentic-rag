from dataclasses import dataclass, field

#nice
@dataclass
class UsageTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    calls_by_provider: dict = field(default_factory=dict)
    call_log: list = field(default_factory=list)

    def record(self, response, provider: str, model: str):
        """Extract token usage from an LLM response."""

        usage_meta = getattr(response, "usage_metadata", None)

        if usage_meta:
            prompt_tokens = usage_meta.get("input_tokens", 0)
            completion_tokens = usage_meta.get("output_tokens", 0)
        else:
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get(
                "prompt_token_count",
                usage.get("prompt_tokens", 0),
            )
            completion_tokens = usage.get(
                "candidates_token_count",
                usage.get("completion_tokens", 0),
            )

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1

        if provider not in self.calls_by_provider:
            self.calls_by_provider[provider] = 0
        self.calls_by_provider[provider] += 1

        self.call_log.append(
            {
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def estimate_cost_usd(self) -> float:
        """Return a rough estimate based on Gemini Flash pricing."""
        input_cost = self.total_prompt_tokens * 0.10 / 1_000_000
        output_cost = self.total_completion_tokens * 0.40 / 1_000_000
        return input_cost + output_cost

    def summary(self) -> str:
        providers = ", ".join(
            f"{provider}: {calls} calls"
            for provider, calls in self.calls_by_provider.items()
        )
        return (
            f"📊 Total calls: {self.total_calls} ({providers})\n"
            f"📊 Tokens: {self.total_prompt_tokens} in + "
            f"{self.total_completion_tokens} out\n"
            f"📊 Est. cost: ${self.estimate_cost_usd():.6f}"
        )
