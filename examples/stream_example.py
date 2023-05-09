import sys

import openai

from gpt_json.stream_json import stream_json

# DO NOT COMMIT THIS KEY
openai.api_key = "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"

def stream_tutor_response(problem, question):
    rendered_prompt = user_prompt_template.replace("{question}", question)
    rendered_prompt = rendered_prompt.replace("{problem}", problem)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered_prompt},
            ], stream=True, max_tokens=700, temperature=0)

    return stream_json(completion)


system_prompt = "You are an expert math tutor. You respond only in JSON in the specified format."
user_prompt_template = """You are an expert math tutor. You respond only in JSON in the specified format. Do not print anything else.

Problem: {problem} 
Student Question: {question}

Your task is to take answer the student question by responding with a JSON object in the following format:
{
    "internal_student_model": "{based on the student question, describe what the student does and does not understand about the problem}",
    "internal_step_by_step": "{step by step solution to the problem}",
    "tutor_response": "{provide your best response to the question}"
}"""
    
if __name__ == "__main__":
    problem = "x^2 + 3 = 12"
    print("\nProblem:", problem)

    while True:
        question = input("Ask question: ")

        print("\nTeacher's thought process:")
        teacher_generator = stream_tutor_response(problem, question)
        for teacher_data, token, event in teacher_generator:
            if event == "new_key":
                print({
                "internal_student_model": "Thought: ",
                "internal_step_by_step": "Problem Solution: ",
                "tutor_response": "\nTeacher's Response: "
                }[token], end="")
            elif event == "value_changed":
                print(token, end="")
            sys.stdout.flush()
        print()