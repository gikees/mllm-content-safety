"""Chain-of-Thought annotation templates for content safety reasoning.

Provides structured CoT templates that guide the model through
step-by-step analysis of multimodal content.
"""

import re

COT_SYSTEM_PROMPT = (
    "You are a content safety analyst. When given an image and text, "
    "analyze the content step-by-step for potential policy violations. "
    "Be precise and objective in your reasoning.\n\n"
    "Do NOT use markdown formatting. Use plain text only. No bold, no headers.\n\n"
    "Here is an example of the exact output format you must follow:\n\n"
    "Step 1 - Visual Description: The image shows a cartoon dog sitting on a couch.\n"
    "Step 2 - Text Analysis: The text says \"when Monday hits.\" The tone is humorous.\n"
    "Step 3 - Cross-Modal Reasoning: The image and text together make a relatable joke "
    "about dreading Mondays. No harmful subtext.\n"
    "Step 4 - Policy Assessment: This content does not violate any policies. "
    "It is a benign humor meme.\n"
    "Step 5 - Classification: safe\n"
    "Risk Severity: 1/5\n\n"
    "You MUST end with exactly:\n"
    "Step 5 - Classification: safe  OR  Step 5 - Classification: unsafe\n"
    "Risk Severity: X/5"
)

COT_TEMPLATE = """Step 1 - Visual Description: {visual_description}
Step 2 - Text Analysis: {text_analysis}
Step 3 - Cross-Modal Reasoning: {cross_modal}
Step 4 - Policy Assessment: {policy_assessment}
Step 5 - Classification: {classification}
Risk Severity: {severity}/5"""

ANNOTATION_PROMPT = (
    "Analyze this image and its text for content safety.\n\n"
    "Follow these steps exactly using plain text (no markdown, no bold):\n"
    "Step 1 - Visual Description: Describe what the image depicts objectively.\n"
    "Step 2 - Text Analysis: What does the text say? What is its tone?\n"
    "Step 3 - Cross-Modal Reasoning: How do the image and text interact? "
    "Does their combination create a meaning different from either alone?\n"
    "Step 4 - Policy Assessment: Does this content violate policies against "
    "hate speech, harassment, violence, or discrimination? Be specific.\n"
    "Step 5 - Classification: safe or unsafe\n"
    "Risk Severity: Rate from 1 (completely benign) to 5 (severely harmful)\n"
)


def build_cot_prompt(text_overlay: str) -> str:
    """Build a CoT annotation prompt for a given meme/post.

    Args:
        text_overlay: The text content of the meme or post.

    Returns:
        Formatted prompt string for CoT generation.
    """
    return ANNOTATION_PROMPT + f'\nThe text overlay/caption reads: "{text_overlay}"'


def parse_cot_response(response: str) -> dict:
    """Parse a CoT response into structured fields.

    Handles both plain-text format (Step 1 - Visual Description:)
    and markdown bold format (**Step 1: Visual Description**) that
    models sometimes produce.

    Args:
        response: Raw model response following the CoT template.

    Returns:
        Dict with keys: visual_description, text_analysis, cross_modal,
        policy_assessment, classification, severity.
    """
    result = {
        "visual_description": "",
        "text_analysis": "",
        "cross_modal": "",
        "policy_assessment": "",
        "classification": "",
        "severity": 0,
        "raw": response,
    }

    # Regex patterns that match both formats:
    #   "Step 1 - Visual Description:"  (plain text)
    #   "**Step 1: Visual Description**" (markdown bold)
    #   "**Step 1 - Visual Description:**" (hybrid)
    step_patterns = [
        (re.compile(r'^\**\s*Step\s*1[\s:\-]+Visual\s+Description\**:?\s*', re.IGNORECASE), "visual_description"),
        (re.compile(r'^\**\s*Step\s*2[\s:\-]+Text\s+Analysis\**:?\s*', re.IGNORECASE), "text_analysis"),
        (re.compile(r'^\**\s*Step\s*3[\s:\-]+Cross[\s\-]?Modal\s+Reasoning\**:?\s*', re.IGNORECASE), "cross_modal"),
        (re.compile(r'^\**\s*Step\s*4[\s:\-]+Policy\s+Assessment\**:?\s*', re.IGNORECASE), "policy_assessment"),
        (re.compile(r'^\**\s*Step\s*5[\s:\-]+Classification\**:?\s*', re.IGNORECASE), "classification"),
    ]
    severity_pattern = re.compile(r'^\**\s*Risk\s+Severity\**:?\s*', re.IGNORECASE)

    lines = response.strip().split("\n")
    current_key = None

    for line in lines:
        stripped = line.strip()
        matched = False

        for pattern, key in step_patterns:
            m = pattern.match(stripped)
            if m:
                result[key] = stripped[m.end():].strip()
                current_key = key
                matched = True
                break

        if not matched:
            m = severity_pattern.match(stripped)
            if m:
                severity_str = stripped[m.end():].strip()
                try:
                    result["severity"] = int(severity_str.split("/")[0])
                except ValueError:
                    result["severity"] = 0
                current_key = None
                matched = True

        if not matched and current_key:
            result[current_key] += " " + stripped

    # Fallback: extract classification from last lines if still empty
    if not result["classification"]:
        for line in reversed(lines[-5:]):
            lower = line.strip().lower()
            if "unsafe" in lower:
                result["classification"] = "unsafe"
                break
            elif "safe" in lower:
                result["classification"] = "safe"
                break

    # Fallback: extract severity from any X/5 pattern if still 0
    if result["severity"] == 0:
        for line in reversed(lines[-5:]):
            m = re.search(r'(\d)\s*/\s*5', line)
            if m:
                result["severity"] = int(m.group(1))
                break

    # Normalize classification to just "safe" or "unsafe"
    if result["classification"]:
        lower_cls = result["classification"].lower().strip().rstrip(".")
        if "unsafe" in lower_cls:
            result["classification"] = "unsafe"
        elif "safe" in lower_cls:
            result["classification"] = "safe"

    return result
