import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER

model = YOLO("yolov9t.pt")

best_loss = float('inf')
wait = 0
patience = 5

def early_stopping_callback(epoch, logs):
    global best_loss, wait
    val_loss = logs.get('val/loss')  # Ensure this key matches your logs
    if val_loss is None:
        print("Validation loss not found in logs.")
        return
    improvement = (best_loss - val_loss) / best_loss * 100
    if improvement < 2:
        wait += 1
    else:
        best_loss = val_loss
        wait = 0
    if wait >= patience:
        print("No improvement, stopping early.")
        model.stop_training = True
    print(f"Epoch {epoch + 1}: Improvement {improvement:.2f}%, Best Loss {best_loss:.4f}")

model.add_callback('on_val_epoch_end', early_stopping_callback)

model.train(data='data.yaml', epochs=500)