import json
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator

with open('config.json', 'r') as f:
    config = json.load(f)

# Declare variables
MODEL_NAME = config['model_name']
TRAIN_DATASET = config['train_dataset']
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
'''
def log_callback_st(train_idx, epoch, training_steps, lr, loss):
    print({f"train/{loss_names[train_idx]}_loss": loss,
             f"train/{loss_names[train_idx]}_lr": lr,
            "train/steps": training_steps})

CALLBACK = log_callback_st
'''
model = SentenceTransformer(MODEL_NAME, device='cpu')

# Load datasets
with open(TRAIN_DATASET, 'r') as f:
    train_dataset = json.load(f)
with open(VAL_DATASET, 'r') as f:
    val_dataset = json.load(f)

# Prepare training data
training_datas = []
for data in train_dataset:
    trainingdata = InputExample(texts=[data[0], data[1]])
    training_datas.append(trainingdata)
loader = DataLoader(training_datas, batch_size=BATCH_SIZE)

if USE_RECOMMENDED_WARMUP:
    WARMUP_STEPS = int(len(loader) * EPOCHS * 0.1)
else:
    WARMUP_STEPS = configured_warmup_steps

# Prepare validation data
sentence1 = [val[0] for val in val_dataset]
sentence2 = [val[1] for val in val_dataset]
scores = [val[2] for val in val_dataset]
print(OPTIMIZER_CLASS)

evaluator = BinaryClassificationEvaluator(sentence1, sentence2, scores, write_csv=True, name="validation")
# Set loss function
loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=32)

# Train the model with all specified parameters
model.fit(
    train_objectives=[(loader, loss)],
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
    checkpoint_save_total_limit=CHECKPOINT_SAVE_TOTAL_LIMIT,
    #callback=CALLBACK
)