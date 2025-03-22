# train.py

import torch
import random
from torch.utils.data import DataLoader
from sentence_transformers import losses, evaluation
from sentence_transformers import SentenceTransformerModelCardData
from config import load_config

# Import your separated load functions
from model_download import load_model
from train_dataset_download import load_train_dataset
from val_dataset_download import load_val_dataset

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.training_args import BatchSamplers

def main():
    # 1. Load config
    config = load_config('config.json')

    MODEL_NAME = config['model_name']
    TRAIN_DATASET = config['train_dataset']
    VAL_DATASET = config['val_dataset']
    TRAIN_BATCH_SIZE = config['train_batch_size']
    EVAL_BATCH_SIZE = config['eval_batch_size']
    EPOCHS = config['epochs']
    WARMUP_STEPS = config.get('warmup_steps', -1)
    WARMUP_RATIO = config.get('warmup_ratio', 0.1)
    EVALUATION_STRATEGY = config['evaluation_strategy']
    EVALUATION_STEPS = config['evaluation_steps']
    OPTIMIZER_CLASS = config['optimizer_class']
    OPTIMIZER_PARAMS = config['optimizer_params']
    SCHEDULER = config['scheduler']
    CHECKPOINT_SAVE_STEPS = config['checkpoint_save_steps']
    CHECKPOINT_SAVE_TOTAL_LIMIT = config.get('checkpoint_save_total_limit', 5)
    LOGGING_STRATEGY = config['logging_strategy']
    LOGGING_STEPS = config['logging_steps']
    OUTPUT_PATH = config['output_path']
    MAX_STEPS = config.get('max_steps', -1)
    WEIGHT_DECAY = config.get('weight_decay', 0.0)
    MAX_GRAD_NORM = config.get('max_grad_norm', 1.0)

    # 2. Load/Download Model
    model = load_model(model_name = MODEL_NAME, device = "cuda") if torch.cuda.is_available() else load_model(model_name = MODEL_NAME, device = "cpu")

    # 3. Load training dataset
    training_data = load_train_dataset(TRAIN_DATASET)

    # 5. Load validation dataset
    eval_dataset = load_val_dataset(VAL_DATASET)
    eval_sentence1 = eval_dataset["text1"]
    eval_sentence2 = eval_dataset["text2"]
    eval_scores    = eval_dataset["label"]
    evaluator = evaluation.BinaryClassificationEvaluator(
        eval_sentence1, 
        eval_sentence2, 
        eval_scores, 
        write_csv=True, 
        name="validation"
    )
    evaluator(model)

    # 6. Define loss
    loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=32)

    # 7. Train with model.fit
    args = SentenceTransformerTrainingArguments(
        output_dir = OUTPUT_PATH,
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = EVAL_BATCH_SIZE,
        learning_rate = OPTIMIZER_PARAMS['lr'],
        lr_scheduler_type = SCHEDULER,
        optim = OPTIMIZER_CLASS,
        weight_decay = WEIGHT_DECAY,
        warmup_ratio = WARMUP_RATIO,
        max_steps = MAX_STEPS,
        max_grad_norm = MAX_GRAD_NORM,
        eval_strategy = EVALUATION_STRATEGY,
        eval_steps = EVALUATION_STEPS, 
        save_strategy="steps",
        save_steps = CHECKPOINT_SAVE_STEPS,
        save_total_limit = CHECKPOINT_SAVE_TOTAL_LIMIT,
        logging_strategy = LOGGING_STRATEGY,
        logging_steps = LOGGING_STEPS,
        logging_first_step = True,
        run_name = MODEL_NAME
    )

    if WARMUP_STEPS > 0:
        args = args.set_lr_scheduler(name = SCHEDULER, max_steps = MAX_STEPS, warmup_steps = WARMUP_STEPS)
        print(f"Using warmup steps: {WARMUP_STEPS}")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=training_data,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

if __name__ == "__main__":
    main()
