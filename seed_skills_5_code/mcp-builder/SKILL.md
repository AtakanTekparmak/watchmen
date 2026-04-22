---
name: mcp-builder
description: "Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK)."
version: 1.0.0
author: Anthropic (anthropics/skills)
license: Apache-2.0
source_url: https://github.com/anthropics/skills/tree/main/skills/mcp-builder
source: upstream
---

# MCP Server Development Guide

## Overview

Create MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. The quality of an MCP server is measured by how well it enables LLMs to accomplish real-world tasks.

---

# Process

## High-Level Workflow

Creating a high-quality MCP server involves four main phases:

### Phase 1: Deep Research and Planning

#### 1.1 Understand Modern MCP Design

**API Coverage vs. Workflow Tools:**
Balance comprehensive API endpoint coverage with specialized workflow tools. Workflow tools can be more convenient for specific tasks, while comprehensive coverage gives agents flexibility to compose operations. Performance varies by client — some clients benefit from code execution that combines basic tools, while others work better with higher-level workflows. When uncertain, prioritize comprehensive API coverage.

**Tool Naming and Discoverability:**
Clear, descriptive tool names help agents find the right tools quickly. Use consistent prefixes (e.g., `github_create_issue`, `github_list_repos`) and action-oriented naming.

**Context Management:**
Agents benefit from concise tool descriptions and the ability to filter/paginate results. Design tools that return focused, relevant data. Some clients support code execution which can help agents filter and process data efficiently.

**Actionable Error Messages:**
Error messages should guide agents toward solutions with specific suggestions and next steps.

#### 1.2 Study MCP Protocol Documentation

Start with the sitemap to find relevant pages: `https://modelcontextprotocol.io/sitemap.xml`

Then fetch specific pages with `.md` suffix for markdown format (e.g., `https://modelcontextprotocol.io/specification/draft.md`).

Key pages to review:
- Specification overview and architecture
- Transport mechanisms (streamable HTTP, stdio)
- Tool, resource, and prompt definitions

#### 1.3 Study Framework Documentation

**Recommended stack:**
- **Language**: TypeScript (high-quality SDK support, good compatibility in execution environments like MCPB, LLMs generate TypeScript well, benefits from static typing and linting).
- **Transport**: Streamable HTTP for remote servers (stateless JSON — simpler to scale than stateful sessions). stdio for local servers.

**Load framework documentation:**
- **MCP Best Practices**: `references/mcp_best_practices.md` — core guidelines in this skill's references dir.
- **TypeScript SDK**: fetch `https://raw.githubusercontent.com/modelcontextprotocol/typescript-sdk/main/README.md`
- **Python SDK**: fetch `https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/README.md`

#### 1.4 Plan Your Implementation

Review the service's API docs to identify key endpoints, authentication requirements, and data models. Prioritize comprehensive API coverage. List endpoints to implement, starting with the most common operations.

---

### Phase 2: Implementation

#### 2.1 Set Up Project Structure

Pick TypeScript (MCP SDK) or Python (FastMCP). Either is fine. Use `references/mcp_best_practices.md` as the style guide.

#### 2.2 Implement Core Infrastructure

Create shared utilities:
- API client with authentication
- Error handling helpers
- Response formatting (JSON/Markdown)
- Pagination support

#### 2.3 Implement Tools

For each tool:

**Input Schema** — use Zod (TypeScript) or Pydantic (Python). Include constraints and clear descriptions. Add examples in field descriptions.

**Output Schema** — define `outputSchema` where possible for structured data. Use `structuredContent` in tool responses (TypeScript SDK feature).

**Tool Description** — concise summary, parameter descriptions, return type schema.

**Implementation** — async/await for I/O, proper error handling with actionable messages, support pagination where applicable, return both text content and structured data when using modern SDKs.

**Annotations** — `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`.

---

### Phase 3: Review and Test

#### 3.1 Code Quality

Review for no duplicated code, consistent error handling, full type coverage, clear tool descriptions.

#### 3.2 Build and Test

**TypeScript:**
- `npm run build` to verify compilation
- Test with MCP Inspector: `npx @modelcontextprotocol/inspector`

**Python:**
- `python -m py_compile your_server.py` to verify syntax
- Test with MCP Inspector

---

### Phase 4: Create Evaluations

After implementing your MCP server, create comprehensive evaluations to test its effectiveness. Use the scripts in this skill to connect and run graded questions:

- `scripts/connections.py` — unified MCP client wrapper over stdio/SSE/streamable-HTTP transports.
- `scripts/evaluation.py` — driver that loads a QA XML file and scores a model's tool-using answers against it.
- `scripts/example_evaluation.xml` — reference format.
- `scripts/requirements.txt` — runtime deps (`mcp`, `anthropic`).

#### Create 10 Evaluation Questions

1. **Tool Inspection**: List available tools and understand their capabilities.
2. **Content Exploration**: Use READ-ONLY operations to explore available data.
3. **Question Generation**: Create 10 complex, realistic questions.
4. **Answer Verification**: Solve each question yourself to verify answers.

#### Evaluation Requirements

Each question must be:
- **Independent** — not dependent on other questions.
- **Read-only** — only non-destructive operations required.
- **Complex** — requires multiple tool calls and deep exploration.
- **Realistic** — based on real use cases humans would care about.
- **Verifiable** — single, clear answer that can be verified by string comparison.
- **Stable** — answer won't change over time.

#### XML Format

```xml
<evaluation>
  <qa_pair>
    <question>Find discussions about AI model launches with animal codenames. One model needed a specific safety designation that uses the format ASL-X. What number X was being determined for the model named after a spotted wild cat?</question>
    <answer>3</answer>
  </qa_pair>
</evaluation>
```

Run:

```bash
python3 scripts/evaluation.py --help
```
