from google import genai
from dotenv import load_dotenv
import os


def call_llm(prompt, schema):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("API key not found")

    client = genai.Client(api_key=api_key)
    model = "gemini-3-flash-preview"

    print("Calling Gemini API..")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
        },
    )

    if response.usage_metadata:
        usage = response.usage_metadata.candidates_token_count
        print(f"Output tokens: {usage}")

    validated = schema.model_validate_json(response.text)
    return validated
