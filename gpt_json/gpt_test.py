import asyncio
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.models import GPTModelVersion

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

class CriteriaSchema(BaseModel):
    satisfactory: bool

SYSTEM_PROMPT = """
Determine whether the user's input text satisfies this criteria.
Criteria: {criteria}

Respond with the following JSON schema:

{json_schema}
"""

async def criteria_semantic(text: str, criteria: str) -> bool:
    """Returns whether the text satisfies the criteria."""
    gpt_json = GPTJSON[CriteriaSchema](API_KEY, model= GPTModelVersion.GPT_3_5)
    response, _ = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"Text: {text}",
            )
        ],
        format_variables={
            "criteria": criteria
        }
    )

    return response.satisfactory
