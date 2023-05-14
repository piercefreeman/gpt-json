import json
from os import getenv

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.gpt_test import criteria_semantic
from gpt_json.models import GPTModelVersion

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

USER_PROMPT = """You are an expert math tutor. You respond only in JSON in the specified format. Do not print anything else.

Your task is to take the transcript of the student-tutor conversation above, the problem, and the list of possible actions with descriptions and return the following in JSON:
student_model: based on the problem and the conversation, describe what the student understands about the problem
why_action: based on the student model and the action descriptions, describe why you've selected the following action
action: ID of action to take
tutor_response: response to the student

Problem: {problem} 
Conversation History:
{conversation}

Actions:
{actions}
"""

ACTIONS = [{'code': 'S2', 'name': 'Provide a hint', 'use_when': 'The student is struggling, seems to have no intuition for what to do next or this problem is just too difficult for them, and they would benefit from a suggestion.', 'instructions': 'Give the student something small that could make solving this problem easier. Do not solve the problem for them.'}, {'code': 'C1', 'name': 'Challenge wrong intuition', 'use_when': 'The student supplies an incorrect response or reveals an incorrect assumption, but you believe that simply challenging it will be enough to help them fix it themselves.', 'instructions': 'Challenge their thinking without revealing precisely what’s wrong with it. The hope is for them to uncover the error independently.'}, {'code': 'C2', 'name': 'Redirect to correct concept/reframe problem', 'use_when': 'The student supplies an incorrect response or reveals an incorrect assumption, and you either have challenged it already, or you believe they need redirection from you, since they’re unable to fix it independently.', 'instructions': 'The student is veering off-track. Help them get back on track with a hint or a light reframing.'}, {'code': 'M1', 'name': 'Prompt to explain work', 'use_when': 'The student has supplied an answer and you’re at a loss as to how they arrived to it.', 'instructions': 'Ask them to explain your reasoning or show you their step-by-step process of having arrived to that answer.'}, {'code': 'N', 'name': 'Do Nothing', 'use_when': 'The student appears to be struggling, but you are convinced they have what they need to solve it.', 'instructions': "Validate what the learner already knows, and encourage them to spend more time trying, rather than intervening. If you've already tried this, you can allude to something specific already covered in the conversation as extra validation or a light hint."}, {'code': 'D', 'name': 'Success', 'use_when': 'A learner completes the problem correctly, explaining their reasoning through the conversation.', 'instructions': 'Congratulate the learner on their success and their learnings through their unique problem-solving journey.'}]

SYSTEM_PROMPT = """Respond with the following JSON schema:

{json_schema}
"""


def render_conversation(msgs):
    out = ""
    for i, msg in enumerate(msgs):
        if i % 2 == 0:
            out += "Student: " + msg
        else:
            out += "Teacher: " + msg
    return out

def render_actions(routines):
    actions = [{
       "code" : routine["code"],
       "name" : routine["name"],
       "use_when" : routine["use_when"], 
    } for routine in routines]
    return json.dumps(actions, indent=0)


class TutorSchema(BaseModel):
    student_model: str
    why_action: str
    action: str
    tutor_response: str

gpt_json = GPTJSON[TutorSchema](API_KEY, model=GPTModelVersion.GPT_4, temperature=0)
teacher_clf = gpt_json.to_function(messages=[
    GPTMessage(
        role=GPTMessageRole.SYSTEM,
        content=SYSTEM_PROMPT,
    ),
    GPTMessage(
        role=GPTMessageRole.USER,
        content=USER_PROMPT,
    ),
], transforms={
    "conversation": render_conversation,
    "actions": render_actions
})

@pytest.mark.asyncio
async def test_teacher_gives_hint():
    teacher_output = await teacher_clf(problem="x^2 + 3 = 12", conversation=["How do you undo a square?"], actions=ACTIONS)
    
    print("Teacher response: ", teacher_output.tutor_response)
    assert teacher_output.action == "S2"
    assert not await criteria_semantic(teacher_output.tutor_response, criteria="The teacher mentions a very large gorilla.")
    assert not await criteria_semantic(teacher_output.tutor_response, criteria="The teacher undoes the square for the student.")
    assert await criteria_semantic(teacher_output.tutor_response, criteria="The teacher gives a small hint to the student.")

    
