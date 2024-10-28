# Question generation part
import openai
import json
import argparse
import datetime
import toml
import os

data_folder = "./step01/data/"


# Чтение API ключа из .secrets.toml
def read_api_key():
    secrets = toml.load(".secrets.toml")
    return secrets.get("OPENAI_API_KEY")


# Загрузка данных из field_data.json
def load_field_data(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


# Генерация вопросов для заданного subject
def generate_questions(client, language, num_questions, field_of_knowledge, subfield, subject_data, model):
    subject = subject_data.get("subject", "")
    perspectives = subject_data.get("perspectives", [])

    # Формируем строку с перечислением перспектив
    perspectives_str = ", ".join(perspectives)

    prompt_template = f"""
Objective: In the subject "{subject}" of the subfield "{subfield}" in "{field_of_knowledge}", generate {num_questions} questions where the answer depends on the context or set of assumptions.
Instructions:
- Create questions that can have different answers based on different contexts or perspectives.
- For each question, provide at least two contexts from the following perspectives: {perspectives_str}, that lead to different answers.
- Provide the answer for each context.
- Ensure all answers are correct.
- Answers should be concise, not more than two words or numbers.
- Provide the output in JSON format with the following structure:
[
  {{
    "field": "{field_of_knowledge}",
    "subfield": "{subfield}",
    "subject": "{subject}",
    "question": "Your question",
    "variations": [
      {{"context": "Context 1", "answer": "Answer 1"}},
      {{"context": "Context 2", "answer": "Answer 2"}},
      ...
    ]
  }},
  ...
]
""".strip()

    print(f"Prompt to model:\n{prompt_template}\n")

    # Вызов OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an AI language model that generates questions."}, {"role": "user", "content": prompt_template}],
        max_tokens=3000,
        temperature=0.7,
    )

    # Извлечение ответа от модели
    assistant_reply = response.to_dict()["choices"][0]["message"]["content"].strip()

    # Обработка ответа, удаление маркеров кода, если они присутствуют
    assistant_reply = response.to_dict()["choices"][0]["message"]["content"].strip()
    if assistant_reply[:7] == "```json":
        assistant_reply = assistant_reply[10:-6]
    if assistant_reply[0] != "[":
        assistant_reply = f"[{assistant_reply}]"

    print("Assistant's reply:\n", assistant_reply)

    # Попытка разобрать ответ как JSON
    try:
        generated_items = json.loads(assistant_reply) if assistant_reply else []
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return []

    timestamp = datetime.datetime.now().isoformat()
    for item in generated_items:
        item["generated_at"] = timestamp

    return generated_items


# Основная функция
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
    supported_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4"]
    model = supported_models[args.model_number]

    field_data_list = load_field_data(os.path.join(data_folder, args.field_data_file))

    # Проходим по каждому fok в field_data_list
    for field_data in field_data_list:
        field_of_knowledge = field_data.get("fok", "Unknown Field")
        sfoks = field_data.get("sfoks", [])

        # Проходим по каждому sfok
        for sfok_data in sfoks:
            subfield = sfok_data.get("name", "Unknown Subfield")
            subjects = sfok_data.get("subjects", [])

            # Проходим по каждому subject
            for subject_data in subjects:
                # Генерируем вопросы для текущего subject
                generated_questions = generate_questions(client, args.language, args.num_questions, field_of_knowledge, subfield, subject_data, model)

                # Сохраняем сгенерированные вопросы в файл
                output_filename = f"{data_folder}{model}_generated_questions_{args.language}.jsonl"
                with open(output_filename, "a", encoding="utf-8") as outfile:
                    for item in generated_questions:
                        json_line = json.dumps(item, ensure_ascii=False)
                        outfile.write(json_line + "\n")

                subject_name = subject_data.get("subject", "")
                print(f"\nSaved to '{output_filename}' for subject '{subject_name}'.\n")


if __name__ == "__main__":
    main()
