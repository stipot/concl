# ConCl (Context Clarity)

Цель проекта ConCl (Context Clarity) в разработке методики и алгоритма сжатия контекста для раскрытия неопределенности последующего текста, т.е. сохранении тех элементов контекста, которые максимально способствуют уменьшению неоднозначности выводов. При этом важно не столько минимизировать объем контекста, сколько стратегически отбирать его части, способные прояснить смысл последующего утверждения.

Результаты проекта планируется применять в следующих ситуациях:
- Улучшение векторизации текста с обогащением уплотненным контекстом для повышения релевантности поиска.
- Повышение эффективности переноса контекста в малое окно внимания языковой модели в длинных цепочках обсуждений.
- Решение задачи уплотнения текста без потери многообразия смыслов.
- В перспективе возможен выход на плотный нечеловекочитаемый метаязык для обмена информацией с языковыми моделями.

# Дорожная карта

1. [Оценка предубеждений моделей](./step01/step01.ipynb)
* Устойчивость выбора ответа в условиях концептуальной неопределенности вопроса.
2. [Проверка моделей на генерацию дастасета на проверку адаптивности](./step02/step02.ipynb) (поведенческой гибкости)
* Способность модели изменять ответ в зависимости от контр предубежденного ответа
3. [Сопротивление зашумленному контенту](./step03/)
* Влияние зашумления контекста
* Влияние зашумления вариантов ответа
4. [Разработка метода извлечения значимой части контекста ](./step04/step04.ipynb)
Сжатие контекста
* Сжатие зашуленного контекста с сохранением смещения ответа в от предубежденного к контекстуализированному
5. [Разработка метода обогащения векторного представления](./step05/step05.ipynb)
* Поиск по векторным представлениям: утверждение, утверждение + контекст, утверждение + сжатый контекст
* Работа с зашумлением данных



## Создание Датасета
#### Введение

Современные большие языковые модели (БЯМ) обладают значительными знаниями, но часто полагаются на заранее известную информацию, что затрудняет оценку их способности использовать контекст для снятия неоднозначностей. В данной работе предлагается новый подход к созданию кастомизированного датасета, который нацелен на проверку и улучшение способности моделей использовать контекст для правильного выбора ответа среди конкурирующих альтернатив.

#### Цель

Целью данного подхода является создание верифицируемого и унифицированного набора данных, который позволяет моделям демонстрировать способность к анализу и применению контекста в ситуациях, где ответ на вопрос неочевиден или изменяется в зависимости от дополнительных сведений. Это достигается путем использования методов конкурирующих концепций и контрочевидного контекста.

#### Методология

[Модель 1](./step01.ipynb)
Создание датасета опирается на два ключевых метода: **метод конкурирующих концепций** и **метод контрочевидного контекста**. 

1. **Метод Конкурирующих Концепций:**
   - **Описание:** Вопросы формулируются таким образом, чтобы включать конкурирующие концепции (например, капитализм vs. социализм, правда vs. ложь). Без контекста ответ на такой вопрос может быть дискуссионным или очевидным с точки зрения общих знаний.
   - **Генерация:** Контекст добавляется к каждому вопросу таким образом, чтобы он менял восприятие и подталкивал к выбору конкретной концепции. Это достигается путем добавления цитат, исторических фактов или ситуационных описаний, которые конкретизируют вопрос в сторону одного из вариантов.
```
Создай фрагмент датасета в формате JSON, следуя этим инструкциям:

1. Сформулируй вопрос, содержащий конкурирующие концепции (например, капитализм vs. социализм, правда vs. ложь).
2. Вопрос должен быть таким, чтобы без контекста казался очевидным, но контекст должен приводить к неочевидному ответу.
3. Ответ должен быть одним словом.
4. Структура JSON должна содержать поля:
   - "question": текст вопроса.
   - "competitors": текст с названиями конкурирующих концепций, разделенных "vs.".
   - "options": массив с двумя противоположными вариантами ответов.
   - "context": текст, который изменяет очевидность ответа.
   - "correct_answer": правильный ответ на основе контекста.

Пример:

[{
  "question": "Какая система лучше поддерживает равенство?",
  "options": ["капитализм", "социализм"],
  "competitors": "капитализм vs. социализм",
  "context": "Известный капиталистический лидер заявил, что рынок создает равные возможности для всех участников, давая каждому шанс на успех через конкуренцию.",
  "correct_answer": "капитализм"
}]

Сгенерируй 10 записей, следуя этим требованиям.
```

   - **Пример:** 
     ```json
     {
       "question": "Какая система лучше поддерживает равенство?",
       "options": ["капитализм", "социализм"],
       "competitors": "капитализм vs. социализм",
       "context": "Известный капиталистический лидер заявил, что рынок создает равные возможности для всех участников, давая каждому шанс на успех через конкуренцию.",
       "correct_answer": "капитализм"
     }
     ```
    - **TODO** 
        1. Ввести классификатор областей: экономика, культуру, физика, математика, психология и т.п.

2. **Метод Контрочевидного Контекста:**
   - **Описание:** Вопросы формулируются с очевидными ответами, но контекст изменяет восприятие и ведет к неочевидному ответу. Этот метод направлен на создание ситуаций, в которых интуитивный ответ оказывается неверным из-за дополнительных обстоятельств.
   - **Генерация:** Контекст добавляется таким образом, чтобы опровергнуть интуитивно правильный ответ и привести к противоположному. Это может включать неожиданные факты, изменения условий или особые случаи.
```
Создай фрагмент датасета в формате JSON, используя метод контрочевидного контекста, следуя этим инструкциям:

1. Сформулируй вопрос, на который без контекста ответ очевиден.
2. Добавь контекст, который изменяет восприятие вопроса, приводя к неожиданному или неочевидному ответу.
3. Ответ должен быть одним словом.
4. Структура JSON должна содержать поля:
   - "question": текст вопроса.
   - "options": массив с двумя возможными ответами ("да", "нет").
   - "context": текст, который меняет очевидный ответ на неочевидный.
   - "correct_answer": правильный ответ на основе контекста.

Пример:

[{
  "question": "Спутник Земли больше Эвереста?",
  "options": ["да", "нет"],
  "context": "США запустили новый космический спутник Земли.",
  "correct_answer": "нет"
}]

Сгенерируй 10 записей, следуя этим требованиям.
```


   - **Пример:** 
     ```json
     {
       "question": "Спутник Земли больше Эвереста?",
       "options": ["да", "нет"],
       "context": "США запустили новый космический спутник Земли.",
       "correct_answer": "нет"
     }
     ```

#### Проверяемость и Надежность

- **Проверяемость:** Предоставленный контекст и структура вопросов позволяют точно проверять, насколько модель полагается на контекст для принятия решения. Очевидные ответы без контекста становятся некорректными, если контекст не был учтен.
- **Надежность:** За счет строгого контроля над генерацией вопросов и контекстов, данный подход исключает возможность угадывания ответов моделями за счет предобученных знаний, фокусируя их на анализе и интерпретации.


#### Публичные датасеты
- SQuAD (Stanford Question Answering Dataset): Датасет вопросов и ответов на основе текстов. Можно адаптировать для задачи сжатия контекста.
- NarrativeQA: Датасет для генерации ответов на вопросы по длинным текстам. Содержит богатый контекст и может быть использован для оценки способности модели работать с длинными последовательностями.
- HotpotQA: Содержит сложные вопросы, требующие интеграции информации из нескольких частей контекста.

## Варианты архитектуры

Выбор архитектуры с латентным кодированием контекста действительно может быть особенно перспективным, поскольку она позволяет учитывать не только явную текстовую информацию, но и скрытые ассоциации и контекстуальные связи, которые не всегда напрямую представлены в тексте. Вот более детализированное предложение по реализации этого подхода, ориентированное на стимулирование редукции информации на основе ассоциаций ближайших порядков:

### Архитектура с Латентным Кодированием Контекста

#### Основные элементы:

1. **Латентное Кодирование через Автоэнкодер:**
   - Используется автоэнкодер или вариационный автоэнкодер (VAE), чтобы преобразовать контекст в компактное латентное представление.
   - Энкодер обучается выделять ключевые особенности контекста, которые наиболее влияют на устранение неопределенности в последующем тексте.

2. **Ассоциативное Латентное Пространство:**
   - Латентное пространство кодирует не только явные элементы текста, но и ассоциативные связи ближайших порядков. Это достигается через использование механизмов внимания и специальных потерь, стимулирующих модель сохранять критически важные ассоциации.
   - Можно использовать ассоциативные потери (например, косинусная близость или более сложные метрики), чтобы модель обучалась выделять не только семантически релевантные фрагменты, но и контексты, связанные через скрытые ассоциации.

3. **Уплотнитель для Контекстуальной Релевантности:**
   - Включение плотного слоя после автоэнкодера, который будет дополнительно фильтровать и подстраивать латентное представление для максимальной релевантности и минимизации избыточности.
   - Этот слой можно обучить на сопоставлении предсказаний с правильными ответами, используя сигналы обратной связи от задачи предсказания вывода.

4. **Декодер для Генерации Сжатого Контекста:**
   - Декодер преобразует латентное представление обратно в текстовую форму, обеспечивая, что восстановленный контекст сохраняет ключевые элементы, необходимые для предсказания вывода.
   - Здесь можно ввести дополнительные ограничения на количество восстановленной информации, чтобы стимулировать сжатие.

5. **Интеграция с БЯМ:**
   - Латентные представления могут быть интегрированы в БЯМ, предоставляя модель дополнительным входом, который обогащает предсказания за счет ассоциативного контекста.
   - Это позволяет основному корпусу модели использовать сжатый и обогащенный контекст для улучшения точности предсказаний.

#### Потенциальные преимущества:

- **Улучшение сжатия:** Модель будет способна эффективно уменьшать объем контекста, сохраняя все ассоциации и связи, которые могут быть важны для понимания последующих утверждений.
- **Гибкость:** Благодаря использованию латентного пространства, архитектура сможет адаптироваться к различным видам текстов и задач, что делает её универсальной для различных приложений.
- **Улучшение точности предсказаний:** Сохранение ассоциативных связей ближайших порядков может помочь в более глубоком понимании текста и улучшении результатов предсказания.
