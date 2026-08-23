<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# Agent Security Guidelines

## Credentials & API Secrets
- **DO NOT read, inspect, or modify `kaggle.json`:** Under no circumstances should any AI agent read, output, display, or alter the `kaggle.json` file, as it contains private Kaggle API authentication credentials.
- **DO NOT output API secrets in chat or logs:** Never request the user to paste raw API credentials in the chat, and never print raw token values to terminal outputs or files. Direct the user to configure them through standard system environment variables or local ignored files.
