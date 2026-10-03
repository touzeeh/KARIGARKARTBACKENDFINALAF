import json
import os

import anthropic


class ClaudeService:
    """
    Central Claude AI service for KarigarKart.

    Claude Haiku 4.5 is used for text-generation features:
    - AI catalog generation
    - AI product titles
    - AI product descriptions

    Image enhancement is handled by local ML/Real-ESRGAN.
    Audio transcription remains on the existing Gemini path because
    Claude is not being used as a speech-to-text replacement here.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured. "
                "Add it to the environment."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5"

    def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)

        text = "".join(parts).strip()

        if not text:
            raise RuntimeError("Claude returned an empty response.")

        return text

    def test(self) -> str:
        return self.generate_text(
            "Reply with exactly: Claude connection successful",
            max_tokens=50,
        )

    def generate_catalog(
        self,
        description: str,
        category: str = "",
        language: str = "en",
    ) -> dict:
        language_instruction = (
            "Write the output in Hindi."
            if language.lower().startswith("hi")
            else "Write the output in English."
        )

        prompt = f"""
You are an AI catalog assistant for KarigarKart,
an Indian marketplace for handmade artisan products.

Create a professional product listing from the information below.

Product description:
{description}

Category:
{category}

{language_instruction}

Return ONLY valid JSON in exactly this structure:

{{
  "title": "short attractive product title",
  "description": "clear detailed marketplace description",
  "category": "appropriate product category",
  "tags": [
    "tag1",
    "tag2",
    "tag3",
    "tag4",
    "tag5"
  ]
}}

Rules:
- Do not invent materials that are not reasonably supported.
- Do not invent certifications.
- Do not claim the product is handmade unless the input supports it.
- Keep the title concise.
- Make the description suitable for an Indian handicraft marketplace.
- Tags should be useful search keywords.
- Return ONLY JSON.
"""

        raw = self.generate_text(prompt, max_tokens=800)
        cleaned = raw.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Claude returned invalid catalog JSON: " + raw
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError("Claude catalog response was not a JSON object.")

        return result

    def generate_product_title(
        self,
        description: str,
        category: str = "",
    ) -> str:
        prompt = f"""
Create ONE concise and attractive product title
for an Indian handicraft marketplace.

Product description:
{description}

Category:
{category}

Rules:
- Maximum around 80 characters.
- Do not use emojis.
- Do not use quotation marks.
- Do not invent facts.
- Do not add unsupported materials.
- Do not add unsupported certifications.
- Return ONLY the title.
"""

        return self.generate_text(prompt, max_tokens=120)

    def generate_product_description(
        self,
        description: str,
        category: str = "",
    ) -> str:
        prompt = f"""
Write a professional marketplace description for
an Indian artisan or handicraft product.

Product information:
{description}

Category:
{category}

Rules:
- 80-150 words.
- Clear and easy to understand.
- Highlight craftsmanship and product characteristics
  only when supported by the input.
- Do not invent certifications.
- Do not invent materials.
- Do not make unsupported claims.
- Do not exaggerate.
- Return ONLY the description.
"""

        return self.generate_text(prompt, max_tokens=350)
