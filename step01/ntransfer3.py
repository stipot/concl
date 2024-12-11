# %% [markdown]
# # Модульная архитектура

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from collections import defaultdict, Counter
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  # Для альтернативной визуализации (опционально)

# %% [markdown]
# ## Определение Моделей
# ### Autoencoder


# %%
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Слои энкодера
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2
        )
        self.fc_enc = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 2 * 2)
        self.decoder = nn.Sequential(
            # Слои декодера
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.fc_enc(z)

        h = self.fc_dec(z)
        h = h.view(z.size(0), 512, 2, 2)
        x_recon = self.decoder(h)
        return x_recon, z

    def encode(self, x, require_grad=True):
        if not require_grad:
            with torch.no_grad():
                z = self.encoder(x)
                z = z.view(z.size(0), -1)
                z = self.fc_enc(z)
        else:
            z = self.encoder(x)
            z = z.view(z.size(0), -1)
            z = self.fc_enc(z)
        return z


# %% [markdown]
# ### BinaryClassifier


# %%
class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim=512):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, z):
        out = self.fc(z)
        return out  # Без активации, т.к. используем BCEWithLogitsLoss


# %% [markdown]
# ### 1.3. CombinedBinaryClassifier


# %%
class CombinedBinaryClassifier(nn.Module):
    def __init__(self, binary_classifiers, num_classes):
        super(CombinedBinaryClassifier, self).__init__()
        self.binary_classifiers = nn.ModuleList(binary_classifiers)
        self.num_classes = num_classes

    def forward(self, z):
        logits = []
        for classifier in self.binary_classifiers:
            out = classifier(z)
            logits.append(out)
        logits = torch.cat(logits, dim=1)  # [batch, num_classes]
        return logits  # Без активации, будем применять sigmoid позже


# %% [markdown]
# ### 1.4. Определение Класса MulticlassClassifier

# %%
import torch.nn as nn


class MulticlassClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=10):
        super(MulticlassClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, z):
        out = self.fc(z)
        return out  # Без softmax, так как используем CrossEntropyLoss


# %% [markdown]
# ## 2. Определение Класса Trainer
# Класс Trainer будет управлять процессами обучения автокодировщика, бинарных классификаторов, объединённой модели и дообучения энкодера.

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from collections import defaultdict, Counter
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Trainer:
    def __init__(self, num_classes=10, latent_dim=512, batch_size=128, device="cuda" if torch.cuda.is_available() else "cpu", save_dir="./saved_models"):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Инициализация моделей
        self.autoencoder = Autoencoder(latent_dim=self.latent_dim).to(self.device)
        self.binary_classifiers = [BinaryClassifier(embedding_dim=self.latent_dim).to(self.device) for _ in range(self.num_classes)]
        self.combined_model = CombinedBinaryClassifier(self.binary_classifiers, self.num_classes).to(self.device)

        # Потери и точности
        self.ae_loss_log = []
        self.binary_loss_logs = defaultdict(list)
        self.binary_acc_logs = defaultdict(list)
        self.fine_tune_loss_log = []
        self.fine_tune_acc_log = []

        # Точности до и после финетюнинга
        self.acc_combined_before = 0.0
        self.error_rate_before = 1.0
        self.acc_combined_after = 0.0
        self.error_rate_after = 1.0

    def prepare_dataloaders(self, train_subset, test_subset):
        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def train_autoencoder(self, epochs=10, lr=0.001):
        print("\nИнициализация и обучение автокодировщика...")
        optimizer_ae = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion_ae = nn.MSELoss()

        for ep in range(epochs):
            epoch_start = time.time()
            self.autoencoder.train()
            running_loss = 0.0
            for d, _ in self.train_loader:
                d = d.to(self.device)
                optimizer_ae.zero_grad()
                x_recon, z = self.autoencoder(d)
                loss = criterion_ae(x_recon, d)
                loss.backward()
                optimizer_ae.step()
                running_loss += loss.item() * d.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.ae_loss_log.append(epoch_loss)
            epoch_end = time.time()
            print(f"Эпоха {ep+1}/{epochs}, Потери AE: {epoch_loss:.6f}, Время: {epoch_end - epoch_start:.2f} сек.")

    def train_binary_classifiers(self, epochs=10, lr=0.001):
        print("\nОбучение бинарных классификаторов для каждого класса...")
        criterion_cls = nn.BCEWithLogitsLoss()

        for cls in range(self.num_classes):
            print(f"\nОбучение классификатора для класса {cls} vs все остальные")

            # Создаём бинарный датасет: 50% класс, 50% остальные
            pos_samples = [i for i, (_, target) in enumerate(train_subset) if target == cls]
            neg_samples = random.sample([i for i, (_, target) in enumerate(train_subset) if target != cls], len(pos_samples))

            # Объединяем индексы
            binary_indices = pos_samples + neg_samples

            # Собираем данные и переименовываем метки
            binary_data = torch.stack([train_subset[i][0] for i in binary_indices], dim=0)
            binary_targets = torch.cat([torch.ones(len(pos_samples)), torch.zeros(len(neg_samples))], dim=0).float()

            # Создаём TensorDataset с правильными метками
            binary_dataset = TensorDataset(binary_data, binary_targets)

            # Создаём DataLoader
            binary_loader = DataLoader(binary_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

            # Инициализируем оптимизатор
            optimizer_cls = optim.Adam(self.binary_classifiers[cls].parameters(), lr=lr)

            # Обучение
            for ep in range(epochs):
                loss, acc = self._train_binary_classifier(self.binary_classifiers[cls], optimizer_cls, criterion_cls, binary_loader)
                self.binary_loss_logs[cls].append(loss)
                self.binary_acc_logs[cls].append(acc)
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    def _train_binary_classifier(self, model, optimizer, criterion, loader):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device).unsqueeze(1).float()
            optimizer.zero_grad()
            z = self.autoencoder.encode(data)
            outputs = model(z)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds.float() == target).sum().item()
            total += target.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate_combined_model(self, loader):
        self.combined_model.eval()
        self.autoencoder.eval()
        correct = 0
        total = 0
        pred_all = []
        target_all = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                z = self.autoencoder.encode(data)
                logits = self.combined_model(z)  # [batch, num_classes]
                probs = torch.sigmoid(logits)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
                pred_all.append(preds.cpu())
                target_all.append(target.cpu())
        if total > 0:
            pred_all = torch.cat(pred_all)
            target_all = torch.cat(target_all)
            accuracy = correct / total
            error_rate = 1 - accuracy
            print(f"Combined model accuracy on all classes (0-{self.num_classes-1}): {accuracy*100:.2f}%")
            print(f"Error rate: {error_rate*100:.2f}%")
            return accuracy, error_rate, pred_all, target_all
        else:
            return 0.0, 1.0, None, None

    def accuracy_per_class(self, pred_all, target_all):
        print("\nAccuracy per class:")
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        for p, t in zip(pred_all, target_all):
            class_total[t.item()] += 1
            if p.item() == t.item():
                class_correct[t.item()] += 1
        for cls in range(self.num_classes):
            if class_total[cls] > 0:
                acc = class_correct[cls] / class_total[cls] * 100
                print(f"Class {cls}: {acc:.2f}%")
            else:
                print(f"Class {cls}: No samples in test set.")

    def save_models(self, suffix="before_finetune"):
        print("\nСохранение моделей...")
        autoenc_save_path = os.path.join(self.save_dir, f"autoencoder_{suffix}.pth")
        torch.save(self.autoencoder.state_dict(), autoenc_save_path)
        print(f"Autoencoder сохранён по пути: {autoenc_save_path}")

        combined_model_save_path = os.path.join(self.save_dir, f"combined_model_{suffix}.pth")
        torch.save(self.combined_model.state_dict(), combined_model_save_path)
        print(f"CombinedBinaryClassifier сохранён по пути: {combined_model_save_path}")

        # Сохранение бинарных классификаторов отдельно
        for cls in range(self.num_classes):
            classifier_save_path = os.path.join(self.save_dir, f"binary_classifier_class_{cls}_{suffix}.pth")
            torch.save(self.binary_classifiers[cls].state_dict(), classifier_save_path)
            print(f"BinaryClassifier {cls} сохранён по пути: {classifier_save_path}")

    def load_models(self, suffix="before_finetune"):
        print("\nЗагрузка моделей...")
        autoenc_save_path = os.path.join(self.save_dir, f"autoencoder_{suffix}.pth")
        self.autoencoder.load_state_dict(torch.load(autoenc_save_path))
        self.autoencoder.to(self.device)
        print(f"Autoencoder загружен из {autoenc_save_path}")

        combined_model_save_path = os.path.join(self.save_dir, f"combined_model_{suffix}.pth")
        self.combined_model.load_state_dict(torch.load(combined_model_save_path))
        self.combined_model.to(self.device)
        print(f"CombinedBinaryClassifier загружен из {combined_model_save_path}")

        # Загрузка бинарных классификаторов отдельно
        for cls in range(self.num_classes):
            classifier_save_path = os.path.join(self.save_dir, f"binary_classifier_class_{cls}_{suffix}.pth")
            self.binary_classifiers[cls].load_state_dict(torch.load(classifier_save_path))
            self.binary_classifiers[cls].to(self.device)
            print(f"BinaryClassifier {cls} загружен из {classifier_save_path}")

    def fine_tune_encoder(self, fine_tune_loader, epochs=3, lr=0.0005):
        print("\nДообучение энкодера на небольшой выборке с фиксированными классификаторами...")
        criterion_fine = nn.BCEWithLogitsLoss()
        optimizer_enc = optim.Adam([p for p in self.autoencoder.parameters() if p.requires_grad], lr=lr)

        for ep in range(epochs):
            ep_start = time.time()  # Определяем ep_start перед обучением
            loss_fine, acc_fine = self._train_encoder_fixed_classifiers(criterion_fine, fine_tune_loader, optimizer_enc)
            self.fine_tune_loss_log.append(loss_fine)
            self.fine_tune_acc_log.append(acc_fine)
            ep_end = time.time()
            print(f"Fine-tuning Epoch {ep+1}/{epochs}, Loss: {loss_fine:.4f}, Accuracy: {acc_fine*100:.2f}%, Время: {ep_end - ep_start:.2f} сек.")

    def _train_encoder_fixed_classifiers(self, criterion, loader, optimizer):
        self.combined_model.eval()  # Классификаторы фиксированы
        self.autoencoder.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device).long()
            batch_size = data.size(0)
            optimizer.zero_grad()
            z = self.autoencoder.encode(data, require_grad=True)  # Позволяем градиентам проходить

            # Получаем логиты из всех классификаторов
            logits_all = []
            for cls in range(self.num_classes):
                classifier = self.combined_model.binary_classifiers[cls]
                logit = classifier(z)  # [batch_size,1]
                logits_all.append(logit)
            logits_all = torch.cat(logits_all, dim=1)  # [batch_size, num_classes]

            # Создаём целевые метки: 1 для правильного класса, 0 для остальных
            targets = torch.zeros(batch_size, self.num_classes).to(self.device)
            targets[torch.arange(batch_size), target] = 1.0

            # Вычисляем loss
            loss = criterion(logits_all, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size

            # Предсказания: sigmoid and argmax
            preds = torch.sigmoid(logits_all)
            preds = torch.argmax(preds, dim=1)
            correct += (preds == target).sum().item()
            total += batch_size
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def plot_results(self):
        print("\nГрафическое отображение результатов...")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Потери Автокодировщика", "Точность Обучения Бинарных Классификаторов", "Потери и Точность Дообучения Энкодера", "Итоговая Точность Модели"),
        )

        # Потери Автокодировщика
        fig.add_trace(go.Scatter(y=self.ae_loss_log, mode="lines+markers", name="Потери AE"), row=1, col=1)

        # Точность Обучения Бинарных Классификаторов для всех классов
        for cls in range(self.num_classes):
            fig.add_trace(go.Scatter(y=self.binary_acc_logs[cls], mode="lines+markers", name=f"Class {cls} Train Acc"), row=1, col=2)

        # Потери и Точность Дообучения Энкодера
        fig.add_trace(go.Scatter(y=self.fine_tune_loss_log, mode="lines+markers", name="Fine-tune Loss"), row=2, col=1)
        fig.add_trace(go.Scatter(y=self.fine_tune_acc_log, mode="lines+markers", name="Fine-tune Accuracy"), row=2, col=1)

        # Итоговая Точность Модели до и после Дообучения
        fig.add_trace(
            go.Bar(
                x=["Before Fine-tuning", "After Fine-tuning"], y=[self.acc_combined_before, self.acc_combined_after], name="Combined Model Accuracy", marker_color=["blue", "green"]
            ),
            row=2,
            col=2,
        )

        fig.update_layout(height=800, width=1200, title_text="Анализ Обучения", showlegend=True)
        fig.show()

    def perform_full_cycle(self, train_subset, test_subset, fine_tune_epochs=3, fine_tune_lr=0.0005):
        """
        Выполняет полный цикл:
        1. Обучение автокодировщика
        2. Обучение бинарных классификаторов
        3. Оценка модели
        4. Сохранение моделей перед финетюнингом
        5. Финетюнинг энкодера
        6. Оценка модели после финетюнинга
        7. Сохранение моделей после финетюнинга
        """
        self.prepare_dataloaders(train_subset, test_subset)
        self.train_autoencoder(epochs=epochs, lr=0.001)
        self.train_binary_classifiers(epochs=epochs, lr=0.001)
        self.evaluate_and_log()
        self.save_models(suffix="before_finetune")
        self.fine_tune_encoder_after_cycle(fine_tune_epochs, fine_tune_lr)
        self.evaluate_and_log(after_finetune=True)
        self.save_models(suffix="after_finetune")

    def evaluate_and_log(self, after_finetune=False):
        print("\nОценка объединённой модели...")
        if after_finetune:
            prefix = "После финетюнинга"
        else:
            prefix = "До финетюнинга"
        print(f"{prefix} оценки модели на тестовом наборе:")
        acc, error_rate, pred_all, target_all = self.evaluate_combined_model(self.test_loader)
        self.accuracy_per_class(pred_all, target_all)

    def fine_tune_encoder_after_cycle(self, fine_tune_epochs=3, fine_tune_lr=0.0005):
        # Создаём выборку для дообучения
        fine_tune_subset = create_fine_tuning_subset(train_dataset, self.num_classes, samples_per_class)
        fine_tune_loader = DataLoader(fine_tune_subset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.fine_tune_encoder(fine_tune_loader, epochs=fine_tune_epochs, lr=fine_tune_lr)


# %% [markdown]
# ## 3. Подготовка Данных
# Создадим утилитные функции для подготовки датасетов и загрузчиков.

# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict, Counter
import random


def filter_dataset(dataset, num_classes, min_samples):
    """
    Фильтрует датасет, оставляя только `num_classes` классов и минимум `min_samples` образцов на класс.
    """
    class_counts = Counter()
    class_indices = defaultdict(list)

    # Собираем индексы для каждого класса
    for idx, (_, target) in enumerate(dataset):
        if target < num_classes:
            class_indices[target].append(idx)
            class_counts[target] += 1

    # Проверяем, что каждый класс имеет минимум образцов
    for cls in range(num_classes):
        if class_counts[cls] < min_samples:
            raise ValueError(f"Класс {cls} имеет только {class_counts[cls]} образцов, требуется минимум {min_samples}.")

    # Ограничиваем количество образцов до min_samples для каждого класса
    selected_indices = []
    for cls in range(num_classes):
        selected_indices.extend(class_indices[cls][:min_samples])

    return Subset(dataset, selected_indices)


# %% [markdown]
# ## 4. Основной Скрипт
# Теперь объединим всё вместе в основном скрипте, который будет использовать класс Trainer для выполнения всех этапов обучения, сохранения и загрузки моделей.
#
# ### 4.1. Ячейка 1: Обучение Автокодировщика и Бинарных Классификаторов, Сохранение Моделей

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict, Counter

# Предполагается, что классы Autoencoder, MulticlassClassifier и Trainer уже определены выше

# Настройки
num_classes = 10
latent_dim = 512
batch_size = 128
epochs = 10
min_samples_per_class = 200  # Увеличено до 200
max_test_samples = 10000
save_dir = "./saved_models"

# Подготовка трансформаций с аугментацией данных
transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

# Загрузка датасетов
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Фильтрация датасетов
train_subset = filter_dataset(train_dataset, num_classes, min_samples_per_class)
test_subset = filter_dataset(test_dataset, num_classes, min_samples_per_class)

# Ограничение тестового набора
if len(test_subset) > max_test_samples:
    test_indices = random.sample(range(len(test_subset)), max_test_samples)
    test_subset = Subset(test_subset, test_indices)

# Инициализация тренера
trainer = Trainer(num_classes=num_classes, latent_dim=latent_dim, batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu", save_dir=save_dir)

# Подготовка загрузчиков данных
trainer.prepare_dataloaders(train_subset, test_subset)

# Обучение автокодировщика
trainer.train_autoencoder(epochs=epochs, lr=0.001)

# Обучение многоклассового классификатора
trainer.train_binary_classifiers(epochs=epochs, lr=0.001)

# Оценка модели до финетюнинга
trainer.acc_combined_before, trainer.error_rate_before, trainer.pred_all_before, trainer.target_all_before = trainer.evaluate_combined_model(trainer.test_loader)

# Подсчёт точности по каждому классу до финетюнинга
trainer.accuracy_per_class(trainer.pred_all_before, trainer.target_all_before)

# Сохранение моделей перед финетюнингом
trainer.save_models(suffix="before_finetune")

# %% [markdown]
# ### 4.2. Ячейка 2: Загрузка Моделей и Дообучение Энкодера

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict, Counter

# Определите классы Autoencoder, BinaryClassifier, CombinedBinaryClassifier и Trainer здесь или импортируйте из другого файла

# Настройки
num_classes = 10
latent_dim = 512
batch_size = 64  # Изменено пользователем
fine_tuning_epochs = 20  # Изменено пользователем
samples_per_class = 200
save_dir = "./saved_models"

# Инициализация тренера
trainer = Trainer(num_classes=num_classes, latent_dim=latent_dim, batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu", save_dir=save_dir)

# Определение трансформаций
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# Загрузка датасета для дообучения
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)


def create_fine_tuning_subset(dataset, num_classes, samples_per_class=100):
    selected_indices = []
    class_counts = Counter()
    class_indices = defaultdict(list)

    for idx, (_, target) in enumerate(dataset):
        if target < num_classes and class_counts[target] < samples_per_class:
            class_indices[target].append(idx)
            class_counts[target] += 1
            selected_indices.append(idx)
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    return Subset(dataset, selected_indices)


def filter_dataset(dataset, num_classes, min_samples):
    """
    Фильтрует датасет, оставляя только `num_classes` классов и минимум `min_samples` образцов на класс.
    """
    class_counts = Counter()
    class_indices = defaultdict(list)

    # Собираем индексы для каждого класса
    for idx, (_, target) in enumerate(dataset):
        if target < num_classes:
            class_indices[target].append(idx)
            class_counts[target] += 1

    # Проверяем, что каждый класс имеет минимум образцов
    for cls in range(num_classes):
        if class_counts[cls] < min_samples:
            raise ValueError(f"Класс {cls} имеет только {class_counts[cls]} образцов, требуется минимум {min_samples}.")

    # Ограничиваем количество образцов до min_samples для каждого класса
    selected_indices = []
    for cls in range(num_classes):
        selected_indices.extend(class_indices[cls][:min_samples])

    return Subset(dataset, selected_indices)


# Создание выборки для дообучения
fine_tune_subset = create_fine_tuning_subset(train_dataset, num_classes, samples_per_class)
fine_tune_loader = DataLoader(fine_tune_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Загрузка сохранённых моделей
trainer.load_models(suffix="before_finetune")

# Загрузка и фильтрация тестового датасета
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_subset = filter_dataset(test_dataset, num_classes, min_samples=1000)  # Используем тот же min_samples_per_class

# Подготовка загрузчиков данных (инициализирует test_loader)
trainer.prepare_dataloaders(fine_tune_subset, test_subset)

# Дообучение энкодера
trainer.fine_tune_encoder(fine_tune_loader, epochs=fine_tuning_epochs, lr=0.0005)

# Оценка объединённой модели после дообучения
trainer.acc_combined_after, trainer.error_rate_after, pred_all_after, target_all_after = trainer.evaluate_combined_model(trainer.test_loader)

# Подсчёт точности по каждому классу
trainer.accuracy_per_class(pred_all_after, target_all_after)

# Сохранение моделей после дообучения
trainer.save_models(suffix="after_finetune")

trainer.plot_results()

# %%
trainer.plot_results()
