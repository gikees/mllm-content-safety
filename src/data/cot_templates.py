"""Chain-of-Thought annotation templates for content safety reasoning.

Provides structured CoT templates that guide the model through
step-by-step analysis of multimodal content.
"""

COT_SYSTEM_PROMPT = (
    "You are a content safety analyst. When given an image and text, "
    "analyze the content step-by-step for potential policy violations. "
    "Be precise and objective in your reasoning."
)

COT_TEMPLATE = """Step 1 - Visual Description: {visual_description}
Step 2 - Text Analysis: {text_analysis}
Step 3 - Cross-Modal Reasoning: {cross_modal}
Step 4 - Policy Assessment: {policy_assessment}
Step 5 - Classification: {classification}
Risk Severity: {severity}/5"""

ANNOTATION_PROMPT = (
    "Analyze this image and its text for content safety.\n\n"
    "Follow these steps exactly:\n"
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

    step_prefixes = {
        "Step 1 - Visual Description:": "visual_description",
        "Step 2 - Text Analysis:": "text_analysis",
        "Step 3 - Cross-Modal Reasoning:": "cross_modal",
        "Step 4 - Policy Assessment:": "policy_assessment",
        "Step 5 - Classification:": "classification",
    }

    lines = response.strip().split("\n")
    current_key = None

    for line in lines:
        matched = False
        for prefix, key in step_prefixes.items():
            if line.strip().startswith(prefix):
                result[key] = line.strip().removeprefix(prefix).strip()
                current_key = key
                matched = True
                break

        if not matched and current_key:
            if line.strip().startswith("Risk Severity:"):
                severity_str = line.strip().removeprefix("Risk Severity:").strip()
                try:
                    result["severity"] = int(severity_str.split("/")[0])
                except ValueError:
                    result["severity"] = 0
                current_key = None
            else:
                result[current_key] += " " + line.strip()

    return result
