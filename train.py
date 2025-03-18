# train.py

import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, evaluation
from config import load_config

# Import your separated load functions
from model_download import load_model
from train_dataset_download import load_train_dataset
from val_dataset_download import load_val_dataset

def main():
    # 1. Load config
    config = load_config('config.json')

    MODEL_NAME = config['model_name']
    TRAIN_DATASET = config['train_dataset']
    SHUFFLE_TRAIN_DATA = config.get('shuffle_train_data', True)
    VAL_DATASET = config['val_dataset']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    SAVE_BEST_MODEL = config['save_best_model']
    USE_RECOMMENDED_WARMUP = config.get('use_recommended_warmup', True)
    configured_warmup_steps = config.get('warmup_steps', 0)
    EVALUATION_STEPS = config['evaluation_steps']
    OPTIMIZER_CLASS = getattr(torch.optim, config['optimizer_class'])
    OPTIMIZER_PARAMS = config['optimizer_params']
    SCHEDULER = config['scheduler']
    CHECKPOINT_PATH = config['checkpoint_path']
    CHECKPOINT_SAVE_STEPS = config['checkpoint_save_steps']
    CHECKPOINT_SAVE_TOTAL_LIMIT = config.get('checkpoint_save_total_limit', 5)
    SHOW_PROGRESS_BAR = config['show_progress_bar']
    OUTPUT_PATH = config['output_path']
    STEPS_PER_EPOCH = config.get('steps_per_epoch', None)
    WEIGHT_DECAY = config.get('weight_decay', 0.0)
    MAX_GRAD_NORM = config.get('max_grad_norm', 1.0)
    USE_AMP = config.get('use_amp', False)

    # 2. Load/Download Model
    model = load_model(MODEL_NAME, device='cpu')  # or 'cuda' if you want GPU

    # 3. Load training dataset
    training_data = load_train_dataset(TRAIN_DATASET)
    train_loader = DataLoader(
        training_data, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_TRAIN_DATA
    )

    # 4. Determine warmup steps
    if USE_RECOMMENDED_WARMUP:
        WARMUP_STEPS = int(len(train_loader) * EPOCHS * 0.1)
    else:
        WARMUP_STEPS = configured_warmup_steps

    # 5. Load validation dataset
    sentence1, sentence2, scores = load_val_dataset(VAL_DATASET)
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentence1, 
        sentence2, 
        scores, 
        write_csv=True, 
        name="validation"
    )

    # 6. Define loss
    loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=32)

    # 7. Train with model.fit
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        use_amp=USE_AMP,
        scheduler=SCHEDULER,
        optimizer_class=OPTIMIZER_CLASS,
        optimizer_params=OPTIMIZER_PARAMS,
        evaluator=evaluator,
        evaluation_steps=EVALUATION_STEPS,
        output_path=OUTPUT_PATH,
        save_best_model=SAVE_BEST_MODEL,
        show_progress_bar=SHOW_PROGRESS_BAR,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_save_steps=CHECKPOINT_SAVE_STEPS,
        checkpoint_save_total_limit=CHECKPOINT_SAVE_TOTAL_LIMIT
    )

if __name__ == "__main__":
    main()
