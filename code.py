import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# КОНФИГУРАЦИЯ ПАРАМЕТРОВ НЕЙРОННОЙ СЕТИ
# ============================================================================

class NeuralNetworkConfig:
    """Класс для хранения параметров конфигурации нейронной сети."""

    # Параметры данных
    TEST_SIZE: float = 0.2
    VALIDATION_SPLIT: float = 0.2
    RANDOM_STATE: int = 42

    # Параметры нормализации и выбор признаков
    FEATURES_TO_USE: list = [
        'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
        'run_ID', 'cam_col', 'field_ID',
        'redshift', 'plate', 'MJD', 'fiber_ID'
    ]
    # ВАЖНО: размер входа равен числу признаков
    INPUT_DIM: int = len(FEATURES_TO_USE)

    # Параметры архитектуры
    HIDDEN_LAYERS: list = [256, 128, 64, 32]
    OUTPUT_DIM: int = 3
    ACTIVATION_HIDDEN: str = 'relu'
    ACTIVATION_OUTPUT: str = 'softmax'
    DROPOUT_RATE: float = 0.3

    # Параметры обучения
    EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.001
    OPTIMIZER: str = 'adam'

    # Параметры ранней остановки
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MONITOR: str = 'val_loss'
    EARLY_STOPPING_RESTORE_WEIGHTS: bool = True

    # Обработка аномальных значений
    ANOMALY_VALUE: float = -9999.0
    REPLACE_WITH_MEAN: bool = True


# ============================================================================
# КЛАСС ДЛЯ ЗАГРУЗКИ И ПОДГОТОВКИ ДАННЫХ
# ============================================================================

class DataPreprocessor:
    """Класс для загрузки, обработки и нормализации данных."""

    def __init__(self, config: NeuralNetworkConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка датасета из CSV файла."""
        print(f"📂 Загрузка данных из: {file_path}")
        df = pd.read_csv(file_path)
        print(
            f"✓ Данные загружены: {df.shape[0]} образцов, "
            f"{df.shape[1]} признаков"
        )
        return df

    def handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка аномальных значений (-9999)."""
        print("🔧 Обработка аномальных значений...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            anomaly_count = (df[col] == self.config.ANOMALY_VALUE).sum()
            if anomaly_count > 0 and self.config.REPLACE_WITH_MEAN:
                mean_val = df[df[col] != self.config.ANOMALY_VALUE][col].mean()
                df[col] = df[col].replace(self.config.ANOMALY_VALUE, mean_val)
                print(
                    f"  → {col}: {anomaly_count} значений заменены на среднее"
                )

        return df

    def prepare_features_and_labels(
        self,
        df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Подготовка признаков и меток класса."""
        print("⚙️ Подготовка признаков...")

        X = df[self.config.FEATURES_TO_USE].values
        y = self.label_encoder.fit_transform(df['class'])
        y_categorical = keras.utils.to_categorical(
            y,
            num_classes=self.config.OUTPUT_DIM
        )

        print(f"  ✓ Количество признаков: {X.shape[1]}")
        print(f"  ✓ Классы: {self.label_encoder.classes_}")
        print("  ✓ Распределение классов:")
        for idx, class_name in enumerate(self.label_encoder.classes_):
            count = (y == idx).sum()
            percentage = (count / len(y)) * 100
            print(f"    - {class_name}: {count} ({percentage:.1f}%)")

        return X, y_categorical

    def split_and_normalize(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделение на тестовое/обучающее и нормализация."""
        print("📊 Разделение и нормализация данных...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        X_train_norm = self.scaler.fit_transform(X_train)
        X_test_norm = self.scaler.transform(X_test)

        print(f"  ✓ Обучающий набор: {X_train_norm.shape[0]} образцов")
        print(f"  ✓ Тестовый набор: {X_test_norm.shape[0]} образцов")

        self.X_train = X_train_norm
        self.X_test = X_test_norm
        self.y_train = y_train
        self.y_test = y_test

        return X_train_norm, X_test_norm, y_train, y_test


# ============================================================================
# КЛАСС НЕЙРОННОЙ СЕТИ
# ============================================================================

class AstronomyNeuralNetwork:
    """Класс для создания и обучения полносвязной нейронной сети."""

    def __init__(self, config: NeuralNetworkConfig) -> None:
        self.config = config
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None

    def build_model(self) -> keras.Model:
        """Построение архитектуры нейронной сети."""
        print("🏗️ Построение архитектуры нейронной сети...")

        model = keras.Sequential()
        model.add(
            layers.Input(
                shape=(self.config.INPUT_DIM,),
                name='input_layer'
            )
        )

        for i, units in enumerate(self.config.HIDDEN_LAYERS):
            model.add(
                layers.Dense(
                    units,
                    activation=self.config.ACTIVATION_HIDDEN,
                    kernel_initializer='he_normal',
                    name=f'hidden_layer_{i + 1}'
                )
            )
            model.add(
                layers.Dropout(
                    self.config.DROPOUT_RATE,
                    name=f'dropout_{i + 1}'
                )
            )
            print(f"  ✓ Скрытый слой {i + 1}: {units} нейронов")

        model.add(
            layers.Dense(
                self.config.OUTPUT_DIM,
                activation=self.config.ACTIVATION_OUTPUT,
                name='output_layer'
            )
        )
        print(f"  ✓ Выходной слой: {self.config.OUTPUT_DIM} нейронов")

        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None
    ) -> keras.callbacks.History:
        """Обучение нейронной сети."""
        print("🚀 Начинаем обучение нейронной сети...")
        print(f"  Эпохи: {self.config.EPOCHS}")
        print(f"  Размер батча: {self.config.BATCH_SIZE}")
        print(f"  Скорость обучения: {self.config.LEARNING_RATE}")

        early_stopping = EarlyStopping(
            monitor=self.config.EARLY_STOPPING_MONITOR,
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=self.config.EARLY_STOPPING_RESTORE_WEIGHTS,
            verbose=1
        )

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )

        self.history = history
        print("✓ Обучение завершено!")
        return history

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict[str, float]:
        """Оценка модели на тестовом наборе."""
        print("\n📈 Оценка модели на тестовом наборе...")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"  ✓ Потери: {loss:.4f}")
        print(f"  ✓ Точность: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        return {'loss': float(loss), 'accuracy': float(accuracy)}

    def get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Получение предсказаний модели."""
        return self.model.predict(X, verbose=0)


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

class Visualizer:
    """Класс для визуализации результатов обучения и метрик."""

    @staticmethod
    def plot_training_history(
        history: keras.callbacks.History
    ) -> None:
        """Визуализация истории обучения."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(
            history.history['accuracy'],
            'b-',
            linewidth=2,
            label='Обучение'
        )
        axes[0].plot(
            history.history['val_accuracy'],
            'r-',
            linewidth=2,
            label='Валидация'
        )
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Точность')
        axes[0].set_title('Точность обучения и валидации')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            history.history['loss'],
            'b-',
            linewidth=2,
            label='Обучение'
        )
        axes[1].plot(
            history.history['val_loss'],
            'r-',
            linewidth=2,
            label='Валидация'
        )
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Потери')
        axes[1].set_title('Потери обучения и валидации')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("✓ График истории обучения сохранён!")

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str]
    ) -> None:
        """Визуализация матрицы ошибок."""
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.title('Матрица ошибок')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("✓ Матрица ошибок сохранена!")

    @staticmethod
    def plot_class_distribution(
        y: np.ndarray,
        class_names: list[str]
    ) -> None:
        """Визуализация распределения классов."""
        y_classes = np.argmax(y, axis=1)
        unique, counts = np.unique(y_classes, return_counts=True)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            [class_names[i] for i in unique],
            counts,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            edgecolor='black'
        )

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )

        plt.xlabel('Класс')
        plt.ylabel('Количество образцов')
        plt.title('Распределение классов')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("✓ График распределения классов сохранён!")


# ============================================================================
# ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================================

def main():
    """Главная функция выполнения всего пайплайна."""
    print("=" * 75)
    print("КЛАССИФИКАЦИЯ АСТРОНОМИЧЕСКИХ ОБЪЕКТОВ С ПОМОЩЬЮ НЕЙРОННОЙ СЕТИ")
    print("Авторы: Рыжков Артём, Грицков Василий")
    print("=" * 75)

    config = NeuralNetworkConfig()
    preprocessor = DataPreprocessor(config)

    print("\n" + "=" * 75)
    print("ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 75)

    df = preprocessor.load_data('star_classification-3.csv')
    df = preprocessor.handle_anomalies(df)
    X, y = preprocessor.prepare_features_and_labels(df)
    X_train, X_test, y_train, y_test = preprocessor.split_and_normalize(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    print(f"✓ Валидационный набор: {X_val.shape[0]} образцов")

    print("\n" + "=" * 75)
    print("ЭТАП 2: ПОСТРОЕНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ")
    print("=" * 75)

    nn = AstronomyNeuralNetwork(config)
    model = nn.build_model()
    print("\n📋 Резюме модели:")
    model.summary()

    print("\n" + "=" * 75)
    print("ЭТАП 3: ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("=" * 75)

    history = nn.train_model(X_train, y_train, X_val, y_val)

    print("\n" + "=" * 75)
    print("ЭТАП 4: ОЦЕНКА МОДЕЛИ")
    print("=" * 75)

    test_metrics = nn.evaluate_model(X_test, y_test)
    y_pred = nn.get_predictions(X_test)

    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n📊 Детальный отчёт классификации:")
    print(
        classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=preprocessor.label_encoder.classes_
        )
    )

    print("\n" + "=" * 75)
    print("ЭТАП 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 75)

    visualizer = Visualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(
        y_test,
        y_pred,
        preprocessor.label_encoder.classes_
    )
    visualizer.plot_class_distribution(
        y,
        preprocessor.label_encoder.classes_
    )

    print("\n" + "=" * 75)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 75)
    print(
        f"✓ Итоговая точность на тестовом наборе: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )
    print(f"✓ Итоговые потери: {test_metrics['loss']:.4f}")
    print(
        f"✓ Количество эпох обучения: "
        f"{len(history.history['loss'])}"
    )
    print("\n✓ Все результаты сохранены в текущую директорию!")
    print("=" * 75)

    return nn, preprocessor, history, test_metrics


if __name__ == "__main__":
    nn, preprocessor, history, metrics = main()
