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


# Загрузка сгенерированных вопросов
def load_generated_questions(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# Функция для тестирования модели
def test_model(client, questions_data, model, temperature, session_id):
    results = []
    for item in questions_data[:2]:  # TODO Ограничить выборку
        field = item.get("field", "")
        subfield = item.get("subfield", "")
        subject = item.get("subject", "")
        question = item.get("question", "")
        variations = item.get("variations", [])

        # Временная метка для текущего теста
        timestamp = datetime.datetime.now().isoformat()

        # Тестирование вопроса без контекста
        prompt_template = """This is test question. Answer with maximum of two words or numbers. %s"""
        prompt_no_context = prompt_template % question

        response_no_context = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_no_context}],
            max_tokens=50,
            temperature=temperature,
        )

        model_answer_no_context = response_no_context.to_dict()["choices"][0]["message"]["content"].strip()

        # Тестирование вопроса с каждым контекстом
        variation_results = []
        for variation in variations:
            context = variation.get("context", "")
            expected_answer = variation.get("answer", "")

            prompt_with_context = prompt_template % f"{context}\n{question}"

            response_with_context = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_with_context}],
                max_tokens=50,
                temperature=temperature,
            )

            model_answer = response_with_context.to_dict()["choices"][0]["message"]["content"].strip()

            # Сравнение ответа модели с ожидаемым ответом
            correct = model_answer.lower() == expected_answer.lower()
            print(f"Expected: {expected_answer}. Replied: {model_answer}")

            # Уменьшение длины ключей до 1 токена
            variation_result = {"c": context, "e": expected_answer, "m": model_answer, "k": correct}

            variation_results.append(variation_result)

        # Составление результата для текущего вопроса
        result_item = {"f": field, "s": subfield, "j": subject, "q": question, "n": model_answer_no_context, "v": variation_results, "d": timestamp, "t": session_id}

        results.append(result_item)

    return results


# Главная функция
def main():
    parser = argparse.ArgumentParser(description="Test model on generated questions.")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, ru).")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help='Model name: "gpt-4o", "gpt-4o-2024-05-13", "gpt-3.5-turbo", "gpt-4"')
    parser.add_argument("--output_file", type=str, default="test_results.jsonl", help="Output file for test results.")
    parser.add_argument("--temperature", type=str, default=0.5, help="Inference temperature")
    parser.add_argument("--session_id", type=str, required=True, help="Identifier for the test session.")
    args = parser.parse_args()

    args = parser.parse_args()
    api_key = read_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = openai.OpenAI(api_key=api_key)
    questions_data = load_generated_questions(f"{data_folder}questions_data_{args.language}.jsonl")

    test_results = test_model(client, questions_data, args.model, args.temperature, args.session_id)

    # Сохранение результатов тестирования
    output_filepath = os.path.join(data_folder, args.output_file)
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for result in test_results:
            json_line = json.dumps(result, ensure_ascii=False)
            outfile.write(json_line + "\n")

    print(f"\nTest results saved to '{output_filepath}'.")


if __name__ == "__main__":
    main()
