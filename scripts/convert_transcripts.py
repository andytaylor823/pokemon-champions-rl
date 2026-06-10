"""
Convert Cursor agent transcript JSONL files into readable markdown conversations.

Usage:
    # Convert specific transcripts
    python scripts/convert_transcripts.py <path1.jsonl> <path2.jsonl> ...

    # Convert all transcripts in the agent-transcripts directory
    python scripts/convert_transcripts.py --all

    # Specify output directory (default: docs/transcripts/)
    python scripts/convert_transcripts.py --all --output-dir docs/transcripts/

Output is one .md file per transcript, named by a slugified title derived from
the first user message, with the UUID as a suffix for uniqueness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Titles for known transcripts (UUID -> (slug, display_title))
KNOWN_TITLES: dict[str, tuple[str, str]] = {
    # b52b2ad3 is venv/pyproject setup, not RL theory -- excluded from relevant-only
    "c48b0545-8a08-4ea5-ab5d-8e5eafb02452": ("mcts-vs-alphazero-mechanics", "MCTS vs AlphaZero Mechanics"),
    "dc2d3bfe-1734-478b-9bd1-3a0fccd0e6f4": ("mcts-strategy-for-pokemon", "MCTS Strategy for Pokemon"),
    "28098d4e-f344-4ce5-a05e-0c5d2a4e9348": ("state-encoding-and-gt-cfr-north-star", "State Encoding & GT-CFR North Star"),
    "fd219ba0-c42a-48e6-8fc2-d58f7f206e6d": ("pog-paper-parameters", "Player of Games Paper Parameters"),
    "f112bffd-a0b3-4d0c-83b5-a5d14dccf778": ("gt-cfr-search-nn-interface", "GT-CFR Search–NN Interface"),
    "686af863-60a5-4675-bb0f-30b405bc0cea": ("kuhn-poker-gt-cfr-build", "Kuhn Poker GT-CFR Build"),
    "4490bb54-bbfe-449d-b95c-2bea9c097e1b": ("leduc-poker-gt-cfr-build", "Leduc Poker GT-CFR Build"),
    "a7f6c182-1fda-4b02-a1e4-ecc48a5da659": ("correlated-meta-priors", "Correlated Meta Priors"),
}

# Regex patterns for stripping system/metadata XML tags from user messages
STRIP_PATTERNS = [
    # Remove entire tagged blocks (including content) for system noise
    re.compile(r"<open_and_recently_viewed_files>.*?</open_and_recently_viewed_files>", re.DOTALL),
    re.compile(r"<git_status>.*?</git_status>", re.DOTALL),
    re.compile(r"<attached_files>.*?</attached_files>", re.DOTALL),
    re.compile(r"<manually_attached_skills>.*?</manually_attached_skills>", re.DOTALL),
    re.compile(r"<system_reminder>.*?</system_reminder>", re.DOTALL),
    re.compile(r"<system_notification>.*?</system_notification>", re.DOTALL),
    re.compile(r"<agent_transcripts>.*?</agent_transcripts>", re.DOTALL),
    re.compile(r"<rules>.*?</rules>", re.DOTALL),
    re.compile(r"<user_info>.*?</user_info>", re.DOTALL),
    re.compile(r"<agent_skills>.*?</agent_skills>", re.DOTALL),
    re.compile(r"<available_skills.*?>.*?</available_skills>", re.DOTALL),
    re.compile(r"<code_selection.*?>.*?</code_selection>", re.DOTALL),
]

# Tags to unwrap (keep content, strip the tags themselves)
UNWRAP_PATTERNS = [
    re.compile(r"<user_query>\s*", re.DOTALL),
    re.compile(r"\s*</user_query>", re.DOTALL),
    re.compile(r"<timestamp>.*?</timestamp>\s*", re.DOTALL),
]

# Heuristic patterns that indicate "thinking/reasoning" text blocks in assistant messages.
# These are internal chain-of-thought that aren't part of the user-facing response.
THINKING_STARTERS = [
    "The user wants",
    "The user is asking",
    "The user has",
    "The user's",
    "The user made",
    "OK, so",
    "OK so",
    "OK now",
    "Let me think",
    "Let me also think",
    "Let me re-read",
    "Let me check",
    "Now I have a thorough",
    "Now I understand",
    "Now let me",
    "I need to",
    "I think the",
    "I should",
    "I'm looking at",
    "I'm thinking",
    "I'll ",
    "Actually, let me",
    "Actually, looking",
    "Actually, the",
    "Actually wait",
    "Actually re-reading",
    "Hmm,",
    "Wait,",
    "Looking at",
    "Looking more",
    "So the flow is:",
    "Good — ",
    "Good, ",
    "This is the key design",
    "This is actually",
    "This works for",
    "Now looking at",
    "For the converter",
    "The challenge is",
    "The problem is",
    "The distinction is",
    "The crucial distinction",
]


def clean_user_text(raw: str) -> str:
    """Strip system metadata from a user message, keeping only the actual query."""
    text = raw
    # Remove entire tagged blocks of system noise
    for pattern in STRIP_PATTERNS:
        text = pattern.sub("", text)
    # Unwrap user_query and timestamp tags (keep content)
    for pattern in UNWRAP_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def is_thinking_block(text: str) -> bool:
    """
    Heuristic: detect assistant text blocks that are internal reasoning
    rather than user-facing response content.

    A block is likely "thinking" if:
    - It starts with a known thinking-starter phrase
    - It doesn't contain markdown headers (which indicate structured response)
    - It doesn't contain code blocks (which indicate technical content)
    """
    stripped = text.strip()
    if not stripped:
        return True

    has_headers = "## " in stripped or "### " in stripped
    has_code_blocks = "```" in stripped
    has_tables = "| " in stripped and " | " in stripped

    # If it has structured content, it's probably a response
    if has_headers or has_code_blocks or has_tables:
        return False

    # Check for thinking-starter patterns
    for starter in THINKING_STARTERS:
        if stripped.startswith(starter):
            return True

    return False


def strip_trailing_thinking(text: str) -> str:
    """
    Remove trailing thinking/reasoning paragraphs from a text block.

    The common pattern is: a structured response (with ## headers, code blocks,
    tables) followed by unstructured first-person reasoning paragraphs. We find
    the last piece of structured content and truncate after it.
    """
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) <= 1:
        return text

    # Walk backwards from the end, dropping paragraphs that look like thinking
    last_good = len(paragraphs) - 1
    for i in range(len(paragraphs) - 1, -1, -1):
        p = paragraphs[i].strip()
        if not p:
            last_good = i - 1
            continue
        # Check if this paragraph is structured content (headers, code, tables, lists)
        has_structure = (
            p.startswith("#")
            or p.startswith("```")
            or p.startswith("| ")
            or p.startswith("- ")
            or p.startswith("1. ")
            or p.startswith("> ")
            or p.startswith("$$")
            or "```" in p
            or "| " in p
        )
        if has_structure:
            last_good = i
            break
        # Check if it starts with a thinking pattern
        is_thinking = False
        for starter in THINKING_STARTERS:
            if p.startswith(starter):
                is_thinking = True
                break
        if is_thinking:
            last_good = i - 1
            continue
        # It's an unstructured paragraph that doesn't match thinking starters -- keep it
        last_good = i
        break

    if last_good < 0:
        return text
    return "\n\n".join(paragraphs[: last_good + 1])


def extract_assistant_text(content_blocks: list[dict]) -> str:
    """
    Extract user-facing text from assistant message content blocks.
    Skips tool_use blocks and filters out likely thinking/reasoning blocks.
    """
    text_parts: list[str] = []

    for block in content_blocks:
        if block.get("type") != "text":
            continue
        text = block["text"].strip()
        if not text:
            continue
        # Skip pure thinking blocks
        if is_thinking_block(text):
            continue
        # Strip trailing thinking from mixed blocks
        cleaned = strip_trailing_thinking(text)
        if cleaned.strip():
            text_parts.append(cleaned)

    return "\n\n".join(text_parts)


def extract_tool_summary(content_blocks: list[dict]) -> str | None:
    """Produce a brief one-line summary of tools called, if any."""
    tools: list[str] = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            # Summarize the tool call briefly
            if name == "Read":
                path = inp.get("path", "")
                short = Path(path).name if path else "?"
                tools.append(f"Read({short})")
            elif name == "Write":
                path = inp.get("path", "")
                short = Path(path).name if path else "?"
                tools.append(f"Write({short})")
            elif name == "Shell":
                cmd = inp.get("command", "")
                # Truncate long commands
                short_cmd = cmd[:60] + "..." if len(cmd) > 60 else cmd
                tools.append(f"Shell(`{short_cmd}`)")
            elif name == "Glob":
                tools.append(f"Glob({inp.get('glob_pattern', '?')})")
            elif name in ("StrReplace", "Grep", "Task", "TodoWrite"):
                tools.append(name)
            else:
                tools.append(name)

    if not tools:
        return None
    return "*[Tools: " + ", ".join(tools) + "]*"


def derive_title(uuid: str, first_user_msg: str) -> tuple[str, str]:
    """
    Derive a slug and display title from the UUID or first user message.
    Returns (slug, display_title).
    """
    if uuid in KNOWN_TITLES:
        return KNOWN_TITLES[uuid]
    # Slugify from the first user message
    words = re.sub(r"[^a-zA-Z0-9\s]", "", first_user_msg).lower().split()
    slug = "-".join(words[:6]) if words else "untitled"
    display = slug.replace("-", " ").title()
    return slug, display


def convert_transcript(jsonl_path: Path) -> tuple[str, str]:
    """
    Convert a single JSONL transcript into a markdown string.
    Returns (title_slug, markdown_content).
    """
    lines: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))

    uuid = jsonl_path.stem
    turns: list[dict[str, str]] = []
    first_user_msg = ""

    for entry in lines:
        role = entry.get("role", "")
        content_blocks = entry.get("message", {}).get("content", [])

        if role == "user":
            # Extract and clean user text
            raw_texts = [b["text"] for b in content_blocks if b.get("type") == "text"]
            raw = "\n\n".join(raw_texts)
            cleaned = clean_user_text(raw)
            if not cleaned:
                continue
            if not first_user_msg:
                first_user_msg = cleaned
            turns.append({"role": "user", "text": cleaned})

        elif role == "assistant":
            # Extract user-facing text
            text = extract_assistant_text(content_blocks)
            tool_summary = extract_tool_summary(content_blocks)

            # Skip turns that are only tool calls with no substantive text
            if not text and tool_summary:
                # Include a brief note about what tools were used
                turns.append({"role": "assistant_tools", "text": tool_summary})
                continue
            if not text:
                continue

            # Append tool summary if there were also tool calls
            if tool_summary:
                text = tool_summary + "\n\n" + text

            turns.append({"role": "assistant", "text": text})

    # Merge consecutive assistant turns (tool-only followed by response)
    merged: list[dict[str, str]] = []
    for turn in turns:
        if (
            merged
            and merged[-1]["role"] in ("assistant", "assistant_tools")
            and turn["role"] in ("assistant", "assistant_tools")
        ):
            # Merge into previous assistant turn
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + turn["text"]
            merged[-1]["role"] = "assistant"
        else:
            merged.append(turn)

    # Build markdown
    title_slug, title_display = derive_title(uuid, first_user_msg)

    md_parts: list[str] = []
    md_parts.append(f"# {title_display}")
    md_parts.append("")
    md_parts.append(f"**Source:** `agent-transcripts/{uuid}/{uuid}.jsonl`")
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")

    turn_num = 0
    for turn in merged:
        if turn["role"] == "user":
            turn_num += 1
            md_parts.append(f"## User (Turn {turn_num})")
            md_parts.append("")
            md_parts.append(turn["text"])
            md_parts.append("")
            md_parts.append("---")
            md_parts.append("")
        elif turn["role"] in ("assistant", "assistant_tools"):
            md_parts.append(f"## Assistant (Turn {turn_num})")
            md_parts.append("")
            md_parts.append(turn["text"])
            md_parts.append("")
            md_parts.append("---")
            md_parts.append("")

    return title_slug, "\n".join(md_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Cursor agent transcript JSONL files into readable markdown."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths to .jsonl transcript files to convert.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all transcripts in the default agent-transcripts directory.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Path to agent-transcripts directory (auto-detected if not set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/transcripts"),
        help="Output directory for markdown files (default: docs/transcripts/).",
    )
    parser.add_argument(
        "--relevant-only",
        action="store_true",
        help="Only convert the known theory-relevant transcripts.",
    )

    args = parser.parse_args()

    # Collect input files
    input_files: list[Path] = []

    if args.files:
        input_files = [Path(f) for f in args.files]
    elif args.all or args.relevant_only:
        # Auto-detect transcripts directory
        transcripts_dir = args.transcripts_dir
        if transcripts_dir is None:
            # Try the standard Cursor project path
            home = Path.home()
            workspace = Path.cwd()
            # Build the Cursor projects path from workspace
            project_key = str(workspace).replace("/", "-").lstrip("-")
            candidate = home / ".cursor" / "projects" / project_key / "agent-transcripts"
            if candidate.exists():
                transcripts_dir = candidate
            else:
                print(f"Could not auto-detect transcripts dir. Tried: {candidate}", file=sys.stderr)
                print("Use --transcripts-dir to specify the path.", file=sys.stderr)
                sys.exit(1)

        if args.relevant_only:
            # Only convert known theory-relevant transcripts
            for uuid in KNOWN_TITLES:
                jsonl = transcripts_dir / uuid / f"{uuid}.jsonl"
                if jsonl.exists():
                    input_files.append(jsonl)
                else:
                    print(f"Warning: {jsonl} not found, skipping.", file=sys.stderr)
        else:
            # Convert all
            for subdir in sorted(transcripts_dir.iterdir()):
                if subdir.is_dir():
                    jsonl = subdir / f"{subdir.name}.jsonl"
                    if jsonl.exists():
                        input_files.append(jsonl)

    if not input_files:
        parser.print_help()
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each transcript
    for jsonl_path in input_files:
        try:
            title_slug, markdown = convert_transcript(jsonl_path)
            uuid_short = jsonl_path.stem[:8]
            output_name = f"{title_slug}_{uuid_short}.md"
            output_path = args.output_dir / output_name
            output_path.write_text(markdown, encoding="utf-8")
            print(f"  {jsonl_path.stem} -> {output_path}")
        except Exception as e:
            print(f"  ERROR converting {jsonl_path}: {e}", file=sys.stderr)

    print(f"\nDone. {len(input_files)} transcript(s) converted to {args.output_dir}/")


if __name__ == "__main__":
    main()
