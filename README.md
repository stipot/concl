# ConCl (Context Clarity)

Цель проекта ConCl (Context Clarity) в разработке методики и алгоритма сжатия контекста для раскрытия неопределенности последующего текста, т.е. сохранении тех элементов контекста, которые максимально способствуют уменьшению неоднозначности выводов. При этом важно не столько минимизировать объем контекста, сколько стратегически отбирать его части, способные прояснить смысл последующего утверждения.

Результаты проекта планируется применять в следующих ситуациях:
- Улучшение векторизации текста с обогащением уплотненным контекстом для повышения релевантности поиска.
- Повышение эффективности переноса контекста в малое окно внимания языковой модели в длинных цепочках обсуждений.
- Решение задачи уплотнения текста без потери многообразия смыслов.
- В перспективе возможен выход на плотный нечеловекочитаемый метаязык для обмена информацией с языковыми моделями.

# Дорожная карта

1. [Создание датасета контекстно вариативных вопросов. Его верификация.](./step01/step01.ipynb)
* Устойчивость выбора ответа в условиях концептуальной неопределенности вопроса.
2. [Оценка предубеждений моделей. Проверка моделей на генерацию дастасета на проверку адаптивности](./step02/step02.ipynb) (поведенческой гибкости)
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


Ниже приведены **только изменения**, которые необходимо внести в ваш код. Пояснения даны в комментариях к изменениям. Не переписывайте весь код, а только внесите указанные изменения.

### Изменения

1. **Добавить вывод графиков потерь автокодировщика, совмещённых потерь и точности бинарных классификаторов, итоговых confusion matrix, потерь и точности дообучения энкодера, оценку confusion matrix по всем классам в одном месте:**

   В классе `Trainer`, вместо `plot_metrics` и `plot_confusion_matrix` в конце добавьте новый метод `plot_all_results`, который будет строить все необходимые графики. Для этого измените существующий `plot_metrics` и `plot_confusion_matrix` на один метод, например `plot_all_results`:

   ```python
   # В классе Trainer удалите или переименуйте plot_metrics и plot_confusion_matrix.
   # Вместо этого добавьте новый метод plot_all_results.
   
   def plot_all_results(self):
       # 1. Потери автокодировщика
       # 2. Потери и точность всех бинарных классификаторов на одном графике (например, средние значения по всем классам)
       # 3. Итоговые confusion matrix по каждому классу после обучения и дообучения
       # 4. Потери и точность дообучения энкодера
       # 5. Confusion matrix по всем классам после дообучения

       # Пример: построим фигуру с несколькими subplot
       fig = make_subplots(rows=3, cols=2, subplot_titles=(
           "AE Loss", 
           "Avg Binary Classifiers Loss & Acc",
           "Fine-tune Loss & Acc",
           "Confusion Matrix Before Fine-tuning (Class 0)",
           "Confusion Matrix After Fine-tuning (Class 0)",
           "Other Confusion Matrices..."
       ))

       # Пример: добавляем потери AE
       fig.add_trace(go.Scatter(y=self.ae_loss_log, mode="lines+markers", name="AE Loss"), row=1, col=1)

       # Пример: усреднённые потери и точность бинарных классификаторов
       avg_loss = []
       avg_acc = []
       for cls in range(self.num_classes):
           if self.classifier_loss_log[cls]:
               avg_loss.append(self.classifier_loss_log[cls][-1])
               avg_acc.append(self.classifier_acc_log[cls][-1])
       if avg_loss and avg_acc:
           fig.add_trace(go.Bar(y=avg_loss, x=list(range(self.num_classes)), name="Binary Avg Loss"), row=1, col=2)
           fig.add_trace(go.Bar(y=avg_acc, x=list(range(self.num_classes)), name="Binary Avg Acc"), row=1, col=2)

       # Пример: потери и точность дообучения энкодера
       if self.fine_tune_loss_log and self.fine_tune_acc_log:
           fig.add_trace(go.Scatter(y=self.fine_tune_loss_log, mode="lines+markers", name="Fine-tune Loss"), row=2, col=1)
           fig.add_trace(go.Scatter(y=self.fine_tune_acc_log, mode="lines+markers", name="Fine-tune Acc"), row=2, col=1)

       # Пример: Confusion matrix для класса 0 до финетюнинга
       # Предполагается, что у вас сохранены pred_all_before/target_all_before 
       # и pred_all_after/target_all_after для построения confusion matrix.
       # Используйте confusion_matrix из sklearn.

       if self.pred_all_before and self.target_all_before:
           from sklearn.metrics import confusion_matrix
           cm_before = confusion_matrix(self.target_all_before[0], self.pred_all_before[0], labels=[0,1])
           fig.add_trace(go.Heatmap(z=cm_before, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale='Blues'), row=2, col=2)

       # Аналогично для after_finetune
       if self.pred_all_after and self.target_all_after:
           cm_after = confusion_matrix(self.target_all_after[0], self.pred_all_after[0], labels=[0,1])
           fig.add_trace(go.Heatmap(z=cm_after, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale='Blues'), row=3, col=1)

       # Можно добавить остальные confusion matrix как дополнительный шаг или сменить разметку subplot.

       fig.update_layout(height=1200, width=1000, title_text="Все результаты обучения")
       fig.show()
   ```

   **Изменения**:  
   - Удалите вызовы старых `plot_metrics()` и `plot_confusion_matrix()` и замените их одним вызовом `plot_all_results()` после всех этапов обучения.
   - Метод `plot_all_results` берет данные из существующих логов и массивов, ничего нового не создавайте, просто используйте уже имеющуюся информацию.
   
2. **Отдельный блок, который после полного цикла обучения начинает сначала, но вместо обучения автокодировщика загружает улучшенный автокодировщик с прошлого этапа, и поставить этот процесс в цикл:**

   В основном скрипте после полного цикла добавьте что-то вроде:
   ```python
   # После завершения одного цикла
   # Выполним повторный цикл, где вместо обучения автокодировщика загрузим улучшенный автокодировщик
   num_iterations = 3  # к примеру
   for i in range(num_iterations):
       # Загрузка улучшенного автокодировщика
       trainer.load_autoencoder(suffix="after_finetune")

       # Повторяем обучение бинарных классификаторов с загруженным автокодировщиком
       trainer.train_binary_classifiers(epochs=epochs, lr=0.001)

       # Оцениваем, дообучаем энкодер и снова сохраняем
       trainer.evaluate_combined_model(trainer.test_loader)
       trainer.fine_tune_encoder(fine_tune_loader, epochs=3, lr=0.0005)
       trainer.save_autoencoder(suffix=f"iter_{i}_after_finetune")
       # Можно также сохранить классификаторы
       for cls in range(num_classes):
           trainer.save_classifier(cls=cls, suffix=f"iter_{i}_after_finetune")
   ```

   **Изменения**:
   - Добавьте этот цикл после основного процесса обучения и дообучения.
   - Не переписывайте весь код, только вставьте этот блок после основной логики.
   
3. **Проверьте, что автокодировщик и бинарные классификаторы замораживаются на соответствующих этапах:**

   В вашем коде `train_binary_classifier` уже добавлен код для заморозки автокодировщика:
   ```python
   self.autoencoder.eval()
   for param in self.autoencoder.parameters():
       param.requires_grad = False
   ```
   
   Убедитесь, что при дообучении энкодера вы возвращаете `requires_grad = True`:
   ```python
   # После обучения бинарных классификаторов:
   self.autoencoder.train()
   for param in self.autoencoder.parameters():
       param.requires_grad = True
   ```
   
   **Изменения**:
   - Добавьте после обучения каждого бинарного классификатора (или после цикла обучения всех классификаторов) возврат автокодировщика в train и восстановление `requires_grad`:
     ```python
     # После train_binary_classifiers(...)
     self.autoencoder.train()
     for param in self.autoencoder.parameters():
         param.requires_grad = True
     ```
   
   При дообучении энкодера (`fine_tune_encoder`) автокодировщик уже должен быть `train()` и `requires_grad=True`, убедитесь, что это так.

---

**Итого:**

- **Изменение 1:**  
  Заменить старые методы вывода графиков одним методом `plot_all_results()`, который отобразит все необходимые графики.  
  Потом в основном коде вызвать `trainer.plot_all_results()` вместо отдельных `plot_metrics()` и `plot_confusion_matrix()`.

- **Изменение 2:**  
  После основного процесса обучения и дообучения добавить цикл, который будет загружать улучшенный автокодировщик и снова обучать бинарные классификаторы, затем дообучать энкодер и сохранять результаты.

- **Изменение 3:**  
  После обучения бинарных классификаторов убедиться в возврате автокодировщика в `train()` режим и `requires_grad=True` для всех параметров.

Не переписывайте весь код, только внесите указанные изменения в существующие части.