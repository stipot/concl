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


# Function to read output_file and collect processed questions
def load_processed_questions(output_filepath):
    processed_questions = set()
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as outfile:
            for line in outfile:
                try:
                    result = json.loads(line)
                    question_text = result.get("q", "").strip()
                    if question_text:
                        processed_questions.add(question_text)
                except json.JSONDecodeError:
                    continue
    return processed_questions


# Function to test the model
def test_model(client, questions_data, model_name, temperature, session_id, batch_size, output_filepath):
    num_questions = len(questions_data)  # TODO Ограничить выборку

    # Initialize an empty list for results
    results = []

    for i in range(0, num_questions, batch_size):
        batch_items = questions_data[i : i + batch_size]

        # Prepare prompts without context, numbering them
        prompts_no_context = []
        for idx, item in enumerate(batch_items):
            question = item.get("q", "")
            prompts_no_context.append({"q": idx + 1, "question": question})

        # Test questions without context
        answers_no_context, model_used = get_model_answers(client, prompts_no_context, model_name, temperature)

        # Prepare prompts with contexts, numbering them
        prompts_with_context = []
        contexts_list = []
        context_counter = 1  # Counter for numbering context questions
        for idx, item in enumerate(batch_items):
            question = item.get("q", "")
            variations = item.get("v", [])
            for variation in variations:
                context = variation.get("c", "").strip()
                if not context.endswith("."):
                    context += "."
                prompt = f'{{"q": {context_counter}, "question": "{context} {question}"}}'
                prompts_with_context.append(prompt)
                contexts_list.append(
                    {"batch_index": idx, "context_index": context_counter - 1, "expected_answer": variation.get("a", ""), "context": variation.get("c", "")}  # Zero-based index
                )
                context_counter += 1

        # Test questions with contexts
        answers_with_context, _ = get_model_answers(client, prompts_with_context, model_name, temperature)

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
            for ctx_info in [c for c in contexts_list if c["batch_index"] == idx]:
                expected_answer = ctx_info["expected_answer"]
                context = ctx_info["context"]
                context_index = ctx_info["context_index"]
                model_answer = answers_with_context[context_index]
                # Compare answers (case-insensitive)
                correct = model_answer.strip().lower() == expected_answer.strip().lower()
                variation_result = {"c": context, "e": expected_answer, "m": model_answer, "k": correct}
                variation_results.append(variation_result)

            # Compile result item
            result_item = {
                "f": field,
                "s": subfield,
                "j": subject,
                "q": question,
                "n": model_answer_no_context,
                "v": variation_results,
                "d": timestamp,
                "t": session_id,
                "model": model_used,
            }

            results.append(result_item)

        # Write results to file after processing the batch
        with open(output_filepath, "a", encoding="utf-8") as outfile:
            for result in results:
                json_line = json.dumps(result, ensure_ascii=False)
                outfile.write(json_line + "\n")

        # Clear results for the next batch
        results = []

    print(f"\nTest results saved to '{output_filepath}'.")


# Function to get model answers for a list of prompts
def get_model_answers(client, prompts, model_name, temperature):
    # Prepare the system message and the prompt
    system_message = "You are a helpful assistant that answers questions."

    # Instruction before the questions
    instruction = "This is a test. Please answer each question concisely, in no more than two words."
    # Prepare the JSON template for the model's response
    prompt_json = json.dumps(prompts, ensure_ascii=False)

    # Combine prompts into a single message
    combined_prompt = f"""{instruction}
Answer the following questions and provide the answers in JSON format with the structure:
[{{"q": question number, "a": "your answer"}}]
Questions:
{prompt_json}
"""

    # Send the batch request
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": combined_prompt}],
        max_tokens=1500,
        temperature=temperature,
    )

    # Get the model name from the response
    model_used = response.model

    # Extract the assistant's reply
    assistant_reply = response.choices[0].message.content.strip()

    # Parse the numbered answers
    answers = parse_numbered_answers(assistant_reply, len(prompts))

    return answers, model_used


# Function to parse numbered answers from the assistant's reply
def parse_numbered_answers(assistant_reply, num_prompts):
    answers = [""] * num_prompts  # Initialize with empty strings
    # Remove code block markers if present
    assistant_reply = assistant_reply.strip()
    if assistant_reply.startswith("```json"):
        assistant_reply = assistant_reply[7:]
        if assistant_reply.endswith("```"):
            assistant_reply = assistant_reply[:-3]
    elif assistant_reply.startswith("```"):
        assistant_reply = assistant_reply[3:]
        if assistant_reply.endswith("```"):
            assistant_reply = assistant_reply[:-3]
    # Try to parse the reply as JSON
    try:
        reply_data = json.loads(assistant_reply)
        for item in reply_data:
            idx = int(item.get("q", 0)) - 1  # Zero-based index
            if 0 <= idx < num_prompts:
                answer = item.get("a", "").strip()
                answers[idx] = answer
    except json.JSONDecodeError:
        print("Error parsing JSON from assistant's reply.")
        # If parsing fails, leave answers as empty strings
    return answers


# Main function
def main():
    parser = argparse.ArgumentParser(description="Test model on generated questions.")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, ru).")
    parser.add_argument("--questions_file", type=str, default="questions_data_en.jsonl", help="Path to the generated questions JSONL file.")
    parser.add_argument("--model", type=str, default="gpt-4o", help='Model name: "gpt-4o", "gpt-4o-2024-05-13", "gpt-3.5-turbo", "gpt-4"')
    parser.add_argument("--output_file", type=str, default="test_results.jsonl", help="Output file for test results.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Inference temperature.")
    parser.add_argument("--session_id", type=str, required=True, help="Identifier for the test session.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of questions per batch.")
    args = parser.parse_args()

    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = openai.OpenAI(api_key=api_key)

    # Path to the output file
    output_filepath = os.path.join(data_folder, args.output_file)

    # Load processed questions
    processed_questions = load_processed_questions(output_filepath)

    # Path to the questions_file
    input_filepath = os.path.join(data_folder, args.questions_file)

    # Load questions data
    questions_data = load_generated_questions(input_filepath)

    # Filter out already processed questions
    questions_data = [q for q in questions_data if q.get("q", "").strip() not in processed_questions]

    if not questions_data:
        print("All questions have been processed.")
        return

    # Run the model testing
    test_model(client, questions_data, args.model, args.temperature, args.session_id, args.batch_size, output_filepath)


if __name__ == "__main__":
    main()
