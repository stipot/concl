# Questions batch test mode
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


# Load generated questions
def load_generated_questions(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# Function to test the model
def test_model(client, questions_data, model, temperature, session_id, batch_size):
    results = []
    num_questions = len(questions_data)  # TODO Ограничить выборку
    for i in range(0, num_questions, batch_size):
        batch_items = questions_data[i : i + batch_size]

        # Prepare prompts without context
        prompts_no_context = []
        for item in batch_items:
            question = item.get("q", "")
            prompts_no_context.append(question)

        # Test questions without context
        answers_no_context = get_model_answers(prompts_no_context, model, temperature)

        # Prepare prompts with contexts
        prompts_with_context = []
        contexts_list = []
        for idx, item in enumerate(batch_items):
            question = item.get("q", "")
            variations = item.get("v", [])
            # For each variation (context), create a prompt
            for variation in variations:
                context = variation.get("c", "")
                context = context.strip()
                if not context.endswith("."):
                    context += "."
                prompt = f"{context} {question}"
                prompts_with_context.append(prompt)
                contexts_list.append({"index": idx, "expected_answer": variation.get("a", ""), "context": variation.get("c", "")})

        # Test questions with contexts
        answers_with_context = get_model_answers(prompts_with_context, model, temperature, output_json=True)

        # Process and record results
        timestamp = datetime.datetime.now().isoformat()
        for idx, item in enumerate(batch_items):
            field = item.get("f", "")
            subfield = item.get("s", "")
            subject = item.get("j", "")
            question = item.get("q", "")

            # Answer without context
            model_answer_no_context = answers_no_context[idx]

            # Variations with context
            variation_results = []
            for ctx_info in [c for c in contexts_list if c["index"] == idx]:
                expected_answer = ctx_info["expected_answer"]
                context = ctx_info["context"]
                model_answer = answers_with_context.pop(0)
                # Compare answers (case-insensitive)
                correct = model_answer.strip().lower() == expected_answer.strip().lower()
                variation_result = {"c": context, "e": expected_answer, "m": model_answer, "k": correct}
                variation_results.append(variation_result)

            # Compile result item
            result_item = {"f": field, "s": subfield, "j": subject, "q": question, "n": model_answer_no_context, "v": variation_results, "d": timestamp, "t": session_id}

            results.append(result_item)

    return results


# Function to get model answers for a list of prompts
def get_model_answers(prompts, model, temperature, output_json=False):
    # Prepare the system message and the prompt
    if output_json:
        # Instruct the model to output answers in JSON format with single-token keys
        system_message = "You are a helpful assistant that answers questions. Please provide the answers in JSON format with keys 'a' for answer."
    else:
        system_message = "You are a helpful assistant that answers questions."

    # Combine all prompts into a single message
    combined_prompt = "\n\n".join([f"Q: {prompt}" for prompt in prompts])

    # Send the batch request
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": combined_prompt}],
        max_tokens=1500,
        temperature=temperature,
    )

    # Extract the assistant's reply
    assistant_reply = response["choices"][0]["message"]["content"].strip()

    # Split the assistant's reply into individual answers
    answers = parse_model_output(assistant_reply, len(prompts), output_json)

    return answers


# Function to parse the model's output into individual answers
def parse_model_output(assistant_reply, num_prompts, output_json=False):
    answers = []
    if output_json:
        # Parse JSON outputs
        # Split the reply into JSON objects
        json_objects = re.findall(r"\{.*?\}", assistant_reply, re.DOTALL)
        for obj_str in json_objects:
            try:
                obj = json.loads(obj_str)
                answer = obj.get("a", "").strip()
                answers.append(answer)
            except json.JSONDecodeError:
                answers.append("")  # Append empty string if parsing fails
        # If the number of answers is less than expected, pad with empty strings
        while len(answers) < num_prompts:
            answers.append("")
    else:
        # Split the replies based on "A:" or numbering
        reply_parts = re.split(r"\nQ:", assistant_reply)
        for part in reply_parts:
            answer_match = re.search(r"A:\s*(.*)", part, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
                answers.append(answer)
        # If the number of answers is less than expected, pad with empty strings
        while len(answers) < num_prompts:
            answers.append("")
    return answers


# Main function
def main():
    parser = argparse.ArgumentParser(description="Test model on generated questions.")
    parser.add_argument("--questions_file", type=str, required=True, help="Path to the generated questions JSONL file.")
    parser.add_argument("--model_number", type=int, default=0, help="Model number from the supported models list.")
    parser.add_argument("--output_file", type=str, default="test_results.jsonl", help="Output file for test results.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Inference temperature.")
    parser.add_argument("--session_id", type=str, required=True, help="Identifier for the test session.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of questions per batch.")
    args = parser.parse_args()

    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    openai.api_key = api_key
    supported_models = ["gpt-3.5-turbo", "gpt-4"]
    model = supported_models[args.model_number]

    questions_data = load_generated_questions(args.questions_file)

    test_results = test_model(questions_data, model, args.temperature, args.session_id, args.batch_size)

    # Save test results
    output_filepath = os.path.join(data_folder, args.output_file)
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for result in test_results:
            json_line = json.dumps(result, ensure_ascii=False)
            outfile.write(json_line + "\n")

    print(f"\nTest results saved to '{output_filepath}'.")


if __name__ == "__main__":
    main()
