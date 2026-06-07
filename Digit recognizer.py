"""
Handwritten Digit Recognizer — CNN
====================================
Recognizes handwritten digits (0-9) using a Convolutional
Neural Network trained on the MNIST dataset with Keras/TensorFlow.

Author: Divya Nimbalkar
Tech Stack: Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    print(f"  TensorFlow version: {tf.__version__}")
except ImportError:
    print("  TensorFlow not found. Install: pip install tensorflow")
    exit(1)


# ──────────────────────────────────────────────
# 1. LOAD & PREPROCESS MNIST
# ──────────────────────────────────────────────

def load_and_preprocess():
    """Load MNIST, normalize pixel values, one-hot encode labels."""
    print("  Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1] and reshape for CNN
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    X_train = X_train[..., np.newaxis]   # shape: (60000, 28, 28, 1)
    X_test  = X_test[..., np.newaxis]    # shape: (10000, 28, 28, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


# ──────────────────────────────────────────────
# 2. BUILD CNN MODEL
# ──────────────────────────────────────────────

def build_cnn() -> keras.Model:
    """
    Build a Convolutional Neural Network:
    Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Dropout → Output
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ], name='DigitRecognizer_CNN')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ──────────────────────────────────────────────
# 3. TRAIN MODEL
# ──────────────────────────────────────────────

def train_model(model, X_train, y_train_cat, X_test, y_test_cat):
    """Train with early stopping and learning rate reduction."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=3, verbose=1),
    ]

    print("\n  Training CNN...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=2,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ──────────────────────────────────────────────
# 4. EVALUATE & VISUALIZE
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test, y_test_cat):
    """Evaluate model and print metrics."""
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n" + "=" * 55)
    print("     DIGIT RECOGNIZER — EVALUATION")
    print("=" * 55)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    return y_pred


def plot_results(history, y_test, y_pred, X_test):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Digit Recognizer CNN — Results', fontsize=14, fontweight='bold')

    # Training curves
    axes[0, 0].plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='coral', ls='--')
    axes[0, 0].set_title('Training vs Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    axes[0, 1].plot(history.history['loss'],     label='Train Loss', color='steelblue')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss',   color='coral', ls='--')
    axes[0, 1].set_title('Training vs Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=range(10), yticklabels=range(10))
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted Digit')
    axes[1, 0].set_ylabel('True Digit')

    # Sample predictions
    n_samples = 5
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    axes[1, 1].axis('off')
    for idx, i in enumerate(indices[:10]):
        ax_sub = fig.add_axes([0.52 + (idx % 5) * 0.09,
                               0.06 + (idx // 5) * 0.12,
                               0.075, 0.10])
        ax_sub.imshow(X_test[i].reshape(28, 28), cmap='gray')
        color = 'green' if y_pred[i] == y_test[i] else 'red'
        ax_sub.set_title(f'P:{y_pred[i]}', fontsize=7, color=color)
        ax_sub.axis('off')
    axes[1, 1].set_title('Sample Predictions (green=correct)', fontsize=10, loc='left')

    plt.tight_layout()
    plt.savefig('digit_recognizer_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Plot saved as 'digit_recognizer_results.png'")


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    model = build_cnn()
    model.summary()

    history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)
    y_pred  = evaluate(model, X_test, y_test, y_test_cat)
    plot_results(history, y_test, y_pred, X_test)

    # Save model
    model.save('digit_recognizer_model.h5')
    print("\n  Model saved as 'digit_recognizer_model.h5'")


if __name__ == "__main__":
    main()