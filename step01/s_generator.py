import openai
import json
import argparse
import datetime
import toml

data_folder = "./step01/data/"


# Чтение API ключа из .secrets.toml
def read_api_key():
    secrets = toml.load(".secrets.toml")
    return secrets.get("OPENAI_API_KEY")


# Загрузка данных из sfoks.json
def load_sfoks(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


# Генерация subjects для каждого sfok
def generate_subjects(client, field_of_knowledge, sfok_name, num_subjects, model):
    prompt_template = """
Objective: In the "{sfok_name}" section of "{field_of_knowledge}", generate {num_subjects} subjects with perspectives that can lead to different interpretations.
Instructions:
- Create a list of subjects with at least two different perspectives for each subject.
- Perspectives should be diverse, allowing for different contexts or assumptions.
- Provide the result in JSON format with the structure:
[
  {{
    "subject": "Subject name",
    "perspectives": ["Perspective 1", "Perspective 2", "..."]
  }},
  ...
]
    """.strip()

    prompt = prompt_template.format(field_of_knowledge=field_of_knowledge, sfok_name=sfok_name, num_subjects=num_subjects)

    print(f"Prompt to model:\n{prompt}\n")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an AI assistant that generates subjects with perspectives."}, {"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
    )

    # Извлечение ответа от модели
    assistant_reply = response.to_dict()["choices"][0]["message"]["content"].strip()
    # TODO Get exact response.to_dict()["model"]

    # Удаление возможных маркеров кода
    if assistant_reply.startswith("```json"):
        assistant_reply = assistant_reply[7:].strip()
        if assistant_reply.endswith("```"):
            assistant_reply = assistant_reply[:-3].strip()

    print("Response:\n", assistant_reply)

    try:
        generated_items = json.loads(assistant_reply)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return []

    timestamp = datetime.datetime.now().isoformat()
    for item in generated_items:
        item["generated_at"] = timestamp

    return generated_items


# Основная функция
def main():
    parser = argparse.ArgumentParser(description="Generate subjects using OpenAI API.")
    parser.add_argument("--sfoks_file", type=str, default="sfoks.json", help="Path to the sfoks JSON file.")
    parser.add_argument("--num_subjects", type=int, default=30, help="Number of subjects to generate for each sfok.")
    parser.add_argument("--model_number", type=int, default=0, help="Model number from the supported models list.")

    args = parser.parse_args()
    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    openai.api_key = api_key
    supported_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4"]
    model = supported_models[args.model_number]

    sfoks_data = load_sfoks(data_folder + args.sfoks_file)

    all_generated_data = []

    # Проходим по каждой комбинации fok и sfok для генерации subjects
    for entry in sfoks_data[:2]:
        field_of_knowledge = entry["fok"]
        sfok_list = entry["sfok"]
        sfoks = []
        for sfok_name in sfok_list:
            generated_subjects = generate_subjects(openai, field_of_knowledge, sfok_name, args.num_subjects, model)
            sfok_data = {"name": sfok_name, "subjects": generated_subjects}
            sfoks.append(sfok_data)
        fok_data = {"fok": field_of_knowledge, "sfoks": sfoks}
        all_generated_data.append(fok_data)

    # Сохранение в field_data.json
    output_filename = data_folder + "field_data.json"
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(all_generated_data, outfile, ensure_ascii=False, indent=2)

    print(f"\nSaved to '{output_filename}'.")


if __name__ == "__main__":
    main()
