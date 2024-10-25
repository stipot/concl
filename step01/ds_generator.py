import openai
import json
import argparse
import datetime
import toml
import re

data_folder = "./step01/data/"


# Read the API key from .secrets.toml
def read_api_key():
    secrets = toml.load(".secrets.toml")
    return secrets.get("OPENAI_API_KEY")


# Load field data
def load_field_data(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


# Generate questions
def generate_questions(client, language, num_questions, field_data, model):
    prompt_template = """
Objective: In the {subfield_of_knowledge} section of {field_of_knowledge}, find {num_questions} questions where the answer depends on the context or set of assumptions.
Instructions:
Create questions that can have different answers based on different mathematical contexts.
For each question, provide at least two contexts that lead to different answers.
Provide the answer for each context.
Ensure all answers are correct.
Answers should be concise, not more than two words or numbers.
Provide {num_questions} answers is array. Format output in JSON:
[
  {{
    "field": "Your field of knowledge",
    "subfield": "Your subfield",
    "q": "Question",
    "vars": [
      {{"ctx": "Context", "ans": "Answer"}},
    ]
  }},
]
""".strip()

    field_of_knowledge = field_data.get("field_of_knowledge", "Mathematics")
    subfield_of_knowledge = ", ".join(field_data.get("subfields", ["number systems"]))
    prompt = prompt_template.format(field_of_knowledge=field_of_knowledge, subfield_of_knowledge=subfield_of_knowledge, num_questions=num_questions)

    print(f"Prompt to model:\n{prompt}\n")

    # Inside generate_questions function
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an AI language model that generates questions."}, {"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
        # response_format={"type": "json_object"},
    )

    # Extract the assistant's reply
    print(response.to_dict()["choices"][0]["message"]["content"].strip())
    assistant_reply = response.to_dict()["choices"][0]["message"]["content"].strip()
    if assistant_reply[:7] == "```json":
        assistant_reply = assistant_reply[10:-6]
    if assistant_reply[0] != "[":
        assistant_reply = f"[{assistant_reply}]"
    print("Response:\n", assistant_reply)

    try:
        generated_items = json.loads(assistant_reply) if assistant_reply else []
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return []

    timestamp = datetime.datetime.now().isoformat()
    for item in generated_items:
        item["generated_at"] = timestamp

    return generated_items


# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate questions using OpenAI API.")
    parser.add_argument("--field_data_file", type=str, default="field_data.json", help="Path to the field data JSON file.")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, ru).")
    parser.add_argument("--num_questions", type=int, default=2, help="Number of questions to generate.")
    parser.add_argument("--model_number", type=int, default=0, help="Model number from the supported models list.")

    args = parser.parse_args()
    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = openai.OpenAI(api_key=api_key)
    field_data_list = load_field_data(data_folder + args.field_data_file)
    supported_models = ["gpt-4o", "gpt-4o-2024-05-13"]
    model = supported_models[args.model_number]

    for field_data in field_data_list:
        generated_questions = generate_questions(client, args.language, args.num_questions, field_data, model)

        output_filename = f"{data_folder}{model}_generated_questions_{args.language}.jsonl"
        with open(output_filename, "a", encoding="utf-8") as outfile:
            for item in generated_questions:
                json_line = json.dumps(item, ensure_ascii=False)
                outfile.write(json_line + "\n")

        print(f"\nSaved to '{output_filename}'.")


if __name__ == "__main__":
    main()
