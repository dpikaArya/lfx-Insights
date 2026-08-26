# Security Policy

## Reporting a vulnerability

Please report security issues privately to the maintainers (louisfelix.nothias@gmail.com)
rather than opening a public issue.

## Notes

- lfx Insights calls external services: a Perspicacité MCP endpoint and an LLM provider
  (via litellm). All HTTP calls use timeouts and bounded retries.
- LLM-generated text is untrusted: it is grounded against the corpus (`verify_quote`)
  and never executed. Output is written to files only.
- No secrets are committed; provider keys are read from the environment.
