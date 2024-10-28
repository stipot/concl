# Question generation part
import openai
import json
import argparse
import datetime
import toml
import os
import re

data_folder = "./step01/data/"


# Read the API key from .secrets.toml
def read_api_key():
    secrets = toml.load(".secrets.toml")
    return secrets.get("OPENAI_API_KEY")


# Load data from field_data.json
def load_field_data(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


# Generate questions for a given subject
def generate_questions(client, language, num_questions, field_of_knowledge, subfield, subject_data, model):
    subject = subject_data.get("subject", "")
    perspectives = subject_data.get("perspectives", [])

    # Create a string of perspectives
    perspectives_str = ", ".join(perspectives)

    # Modify the prompt to exclude field, subfield, and subject from the output
    # Also, instruct the model to use short keys for brevity
    prompt_template = f"""
Objective: In the subject "{subject}" of the subfield "{subfield}" in "{field_of_knowledge}", generate {num_questions} questions where the answer depends on the context or set of assumptions.

Instructions:
- Create questions that can have different answers based on different contexts or perspectives. The questions must allow a finite number of possible answers.
- For each question, provide at least two contexts, that lead to different answers.
- Question without context should also have (obvious or default) answer (don't require information from the contexts).
- Provide the answer for each context.
- Ensure all answers are correct.
- Answers should be concise, not more than two words or numbers.
- Provide the output in JSON format with the following structure:
[
  {{
    "q": "Your question",
    "v": [
      {{"c": "Context 1", "a": "Answer 1"}},
      {{"c": "Context 2", "a": "Answer 2"}}
    ]
  }}
]
Please use only the keys "q" for question, "v" for variations, "c" for context, and "a" for answer.
""".strip()

    print(f"Prompt to model:\n{prompt_template}\n")

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an AI language model that generates questions."}, {"role": "user", "content": prompt_template}],
        max_tokens=3000,
        temperature=0.7,
    )

    # Extract the assistant's reply
    assistant_reply = response.to_dict()["choices"][0]["message"]["content"].strip()

    # Remove possible code markers and extract JSON content
    assistant_reply = extract_json_content(assistant_reply)

    # Attempt to parse the assistant's reply as JSON
    generated_items = parse_json_items(assistant_reply)

    # If parsing failed, return an empty list
    if not generated_items:
        return []

    timestamp = datetime.datetime.now().isoformat()

    # Add back the field, subfield, and subject to each item
    for item in generated_items:
        item["f"] = field_of_knowledge
        item["s"] = subfield
        item["j"] = subject
        item["d"] = timestamp  # Add timestamp with key "d"

    return generated_items


# Function to extract JSON content from the assistant's reply
def extract_json_content(reply):
    # Remove code block markers
    reply = reply.strip()
    if reply.startswith("```json"):
        reply = reply[7:]
        if reply.endswith("```"):
            reply = reply[:-3]
    elif reply.startswith("```"):
        reply = reply[3:]
        if reply.endswith("```"):
            reply = reply[:-3]
    # Remove any text before or after the JSON array
    json_match = re.search(r"\[.*\]", reply, re.DOTALL)
    if json_match:
        reply = json_match.group(0)
    else:
        # Attempt to fix incomplete JSON by adding closing brackets
        if not reply.startswith("["):
            reply = "[" + reply
        if not reply.endswith("]"):
            reply = reply + "]"
    return reply.strip()


# Function to parse JSON items, handling incomplete JSON
def parse_json_items(reply):
    try:
        # Attempt to parse the entire reply
        return json.loads(reply)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Attempting to parse individual JSON objects...")
        # Remove the outer brackets if present
        reply = reply.strip()
        if reply.startswith("["):
            reply = reply[1:]
        if reply.endswith("]"):
            reply = reply[:-1]
        # Split the reply into potential JSON objects
        items = re.findall(r"\{.*?\}(?=,\s*\{|\s*$)", reply, re.DOTALL)
        generated_items = []
        for i, item_str in enumerate(items):
            try:
                obj = json.loads(item_str)
                generated_items.append(obj)
            except json.JSONDecodeError as e:
                print(f"Skipping incomplete item {i}: {e}")
                continue
        return generated_items


# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate questions using OpenAI API.")
    parser.add_argument("--field_data_file", type=str, default="field_data.json", help="Path to the field data JSON file.")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, ru).")
    parser.add_argument("--num_questions", type=int, default=30, help="Number of questions to generate per subject.")
    parser.add_argument("--model_number", type=int, default=0, help="Model number from the supported models list.")

    args = parser.parse_args()
    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = openai.OpenAI(api_key=api_key)
    supported_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    model = supported_models[args.model_number]

    field_data_list = load_field_data(os.path.join(data_folder, args.field_data_file))

    # Iterate over each field of knowledge (fok)
    for field_data in field_data_list:
        field_of_knowledge = field_data.get("fok", "Unknown Field")
        sfoks = field_data.get("sfoks", [])

        # Iterate over each subfield
        for sfok_data in sfoks:
            subfield = sfok_data.get("name", "Unknown Subfield")
            subjects = sfok_data.get("subjects", [])

            # Iterate over each subject
            for subject_data in subjects:
                # Generate questions for the current subject
                generated_questions = generate_questions(client, args.language, args.num_questions, field_of_knowledge, subfield, subject_data, model)

                # Save the generated questions to a file
                output_filename = f"{data_folder}{model}_generated_questions_{args.language}.jsonl"
                with open(output_filename, "a", encoding="utf-8") as outfile:
                    for item in generated_questions:
                        json_line = json.dumps(item, ensure_ascii=False)
                        outfile.write(json_line + "\n")

                subject_name = subject_data.get("subject", "")
                print(f"\nSaved to '{output_filename}' for subject '{subject_name}'.\n")


if __name__ == "__main__":
    main()
