# Available Claude Code Skills

Personal skills installed in `~/.claude/skills/`, available across all projects.

## Anthropic Skills

Source: `~/.claude/skills-repos/anthropic-skills/`

| Skill | Description |
|-------|-------------|
| **algorithmic-art** | Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. For generative art, flow fields, or particle systems. |
| **brand-guidelines** | Applies Anthropic's official brand colors and typography to artifacts that benefit from Anthropic's look-and-feel. |
| **canvas-design** | Create visual art in .png and .pdf documents using design philosophy. For posters, art, designs, or other static pieces. |
| **claude-api** | Build apps with the Claude API or Anthropic SDK. Triggers when code imports `anthropic`/`@anthropic-ai/sdk`/`claude_agent_sdk`. |
| **doc-coauthoring** | Structured workflow for co-authoring documentation, proposals, technical specs, and decision docs. |
| **docx** | Create, read, edit, or manipulate Word documents (.docx files). Tables of contents, headings, page numbers, letterheads, tracked changes. |
| **frontend-design** | Create distinctive, production-grade frontend interfaces. For web components, pages, dashboards, React components, HTML/CSS layouts. |
| **internal-comms** | Write internal communications: status reports, leadership updates, newsletters, FAQs, incident reports, project updates. |
| **mcp-builder** | Guide for creating MCP (Model Context Protocol) servers in Python (FastMCP) or Node/TypeScript (MCP SDK). |
| **pdf** | Read, extract, combine, merge, split, rotate, watermark, create, fill forms, encrypt/decrypt, OCR — anything with PDF files. |
| **pptx** | Create, read, edit, combine, or split PowerPoint presentations (.pptx). Templates, layouts, speaker notes, comments. |
| **skill-creator** | Create new skills, modify existing skills, run evals, benchmark performance, optimize descriptions for better triggering. |
| **slack-gif-creator** | Create animated GIFs optimized for Slack with constraints, validation tools, and animation concepts. |
| **theme-factory** | Style artifacts (slides, docs, reports, HTML pages) with 10 pre-set themes or generate new themes on-the-fly. |
| **web-artifacts-builder** | Create multi-component claude.ai HTML artifacts using React, Tailwind CSS, and shadcn/ui. |
| **webapp-testing** | Test local web applications using Playwright. Verify frontend functionality, debug UI, capture screenshots, view logs. |
| **xlsx** | Open, read, edit, create, or convert spreadsheet files (.xlsx, .xlsm, .csv, .tsv). Formulas, formatting, charting, data cleaning. |

## Streamlit Skills

Source: `~/.claude/skills-repos/streamlit-agent-skills/`

### Umbrella Skill

| Skill | Description |
|-------|-------------|
| **developing-with-streamlit** | Routing skill for ALL Streamlit tasks: creating, editing, debugging, styling, theming, optimizing, or deploying Streamlit apps. Auto-triggers on any Streamlit-related work. |

### Sub-Skills (loaded on demand by the umbrella skill)

| Skill | Description |
|-------|-------------|
| **building-streamlit-chat-ui** | Chat interfaces, conversational UIs, chatbots, AI assistants. Covers `st.chat_message`, `st.chat_input`, message history, streaming. |
| **building-streamlit-custom-components-v2** | Bidirectional Custom Components v2 using `st.components.v2.component`. Inline HTML/CSS/JS, packaged components, theming. |
| **building-streamlit-dashboards** | KPI displays, metric cards, data-heavy layouts. Covers borders, cards, responsive layouts, dashboard composition. |
| **building-streamlit-multipage-apps** | Multi-page apps, navigation setup, state management across pages. |
| **choosing-streamlit-selection-widgets** | Choosing between radio buttons, selectbox, segmented control, pills, and other selection widgets. |
| **connecting-streamlit-to-snowflake** | Database connections, secrets management, querying Snowflake from Streamlit. |
| **creating-streamlit-themes** | Customizing app colors, fonts, appearance. Covers `config.toml`, design principles, CSS avoidance. |
| **displaying-streamlit-data** | Charts, dataframes, metrics visualization. Native charts, Altair, column configuration, sparklines. |
| **improving-streamlit-design** | Polishing apps with icons, badges, spacing, text styling. Material icons, badge syntax, dividers, text casing. |
| **optimizing-streamlit-performance** | Caching, fragments, static vs dynamic widget choices. For slow apps or excessive reruns. |
| **organizing-streamlit-code** | Code structure, separation of concerns, clean UI code, import patterns. |
| **setting-up-streamlit-environment** | Python environments, dependency management with uv, running apps. |
| **using-streamlit-cli** | CLI commands for running apps, configuration, and diagnostics. |
| **using-streamlit-custom-components** | Third-party community components: installation, popular packages, when to use them. |
| **using-streamlit-layouts** | Sidebars, columns, containers, dialogs, bordered cards, horizontal containers. |
| **using-streamlit-markdown** | GitHub-flavored Markdown plus Streamlit extensions: colored text, badges, Material icons, LaTeX. |
| **using-streamlit-session-state** | `st.session_state` for persisting data, widget state, callbacks, and debugging state issues. |

## Updating Skills

```bash
cd ~/.claude/skills-repos/anthropic-skills && git pull
cd ~/.claude/skills-repos/streamlit-agent-skills && git pull
```
