import asyncio
import sys
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel

from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from gpt_json.models import GPTModelVersion
from gpt_json.types_streaming import StreamEventEnum

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a phenomenal and stubborn math tutor. Given the following problem, determine the best way to respond to the student by:
1) Describing what the student does and does not understand about the problem in {student_model_key}
2) Responding to the student's question in a way that does not give away the answer but helps them understand the problem better

Problem: x^2 + 3 = 12 

Respond with the following JSON schema:

{json_schema}
"""

class TutorSchema(BaseModel):
    student_model: str
    tutor_response: str

gpt_json = GPTJSON[TutorSchema](API_KEY, model=GPTModelVersion.GPT_4)

async def main():
    problem = "x^2 + 3 = 12"
    print("\nProblem:", problem)

    question = input("Ask question: ")
    messages = [
        GPTMessage(
            role=GPTMessageRole.SYSTEM,
            content=SYSTEM_PROMPT,
        ),
        GPTMessage(
            role=GPTMessageRole.USER,
            content=f"Student: {question}",
        ),
    ]

    print("\nTeacher's thought process:")
    teacher_generator = gpt_json.stream(messages=messages, format_variables={"student_model_key": "student_model", "problem" : problem}) 
    seen_keys = set()
    async for partial_teacher in teacher_generator:
        if partial_teacher.event == StreamEventEnum.OBJECT_CREATED:
            continue

        if partial_teacher.event == StreamEventEnum.KEY_UPDATED and partial_teacher.updated_key not in seen_keys:
            key_readable = {
            "student_model": "Thought: ",
            "tutor_response": "Response to student: "
            }[partial_teacher.updated_key]
            print(key_readable, end="")
            seen_keys.add(partial_teacher.updated_key)
        
        print(partial_teacher.value_change, end="")
        if partial_teacher.event == StreamEventEnum.KEY_COMPLETED:
            print()
        sys.stdout.flush()
    
if __name__ == "__main__":
    asyncio.run(main())
