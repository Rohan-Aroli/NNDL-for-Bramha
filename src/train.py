import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import save_json

def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Trains the LSTM model with Early Stopping and Model Checkpointing.
    """
    start_time = time.time()

    best_model_path = config.MODELS_DIR / "best_lstm.keras"
    final_model_path = config.MODELS_DIR / "lstm_model.keras"
    history_path = config.REPORTS_DIR / "training_history.json"

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(final_model_path))

    training_time = time.time() - start_time

    # Save training history
    history_dict = history.history
    report = {
        "epochs_completed": len(history_dict['loss']),
        "best_val_loss": float(min(history_dict.get('val_loss', [0]))),
        "training_duration_seconds": float(training_time),
        "lookback": config.LOOKBACK,
        "lstm_units": config.LSTM_UNITS,
        "dropout_rate": config.DROPOUT_RATE,
        "batch_size": config.BATCH_SIZE,
        "history": history_dict
    }
    save_json(report, history_path)

    print(f"Training complete. Best validation loss: {report['best_val_loss']:.4f}")

    return model, history
