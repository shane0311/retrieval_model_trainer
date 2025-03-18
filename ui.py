import gradio as gr
import json
from dropdown_options import (
    model_dropdown_options,
    train_dataset_options,
    val_dataset_options
)

def update_config(
    # Model
    model_name_choice,
    model_name_custom,
    # Training dataset
    train_dataset_choice,
    train_dataset_custom,
    shuffle_train_data,
    # Validation dataset
    val_dataset_choice,
    val_dataset_custom,
    # Training params
    batch_size,
    epochs,
    save_best_model,
    use_recommended_warmup,
    warmup_steps,
    evaluation_steps,
    optimizer_class,
    optimizer_params_lr,
    scheduler,
    checkpoint_path,
    checkpoint_save_steps,
    checkpoint_save_total_limit,
    show_progress_bar,
    output_path,
    steps_per_epoch,
    weight_decay,
    max_grad_norm,
    use_amp
):
    """
    Decide final model_name, train_dataset, val_dataset based on user dropdown selection
    and custom text input. If the dropdown is 'Custom', we read the custom textbox.
    Otherwise, we use the dropdown value.
    """
    # Model name
    if model_name_choice == "Custom":
        final_model_name = model_name_custom.strip()
    else:
        final_model_name = model_name_choice

    # Train dataset
    if train_dataset_choice == "Custom":
        final_train_dataset = train_dataset_custom.strip()
    else:
        final_train_dataset = train_dataset_choice

    # Validation dataset
    if val_dataset_choice == "Custom":
        final_val_dataset = val_dataset_custom.strip()
    else:
        final_val_dataset = val_dataset_choice

    # steps_per_epoch
    steps_per_epoch_val = None if steps_per_epoch == "" else int(steps_per_epoch)

    # Build config
    config = {
        "model_name": final_model_name,
        "train_dataset": final_train_dataset,
        "shuffle_train_data": shuffle_train_data,
        "val_dataset": final_val_dataset,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "save_best_model": save_best_model,
        "use_recommended_warmup": use_recommended_warmup,
        "warmup_steps": int(warmup_steps),
        "evaluation_steps": int(evaluation_steps),
        "optimizer_class": optimizer_class,
        "optimizer_params": {"lr": float(optimizer_params_lr)},
        "scheduler": scheduler,
        "checkpoint_path": checkpoint_path,
        "checkpoint_save_steps": int(checkpoint_save_steps),
        "checkpoint_save_total_limit": int(checkpoint_save_total_limit),
        "show_progress_bar": show_progress_bar,
        "output_path": output_path,
        "steps_per_epoch": steps_per_epoch_val,
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "use_amp": use_amp
    }

    # Write to config.json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    return "Configuration updated successfully!"

interface = gr.Interface(
    fn=update_config,
    inputs=[
        # Model
        gr.Dropdown(
            label="Model Name (Dropdown)",
            choices=model_dropdown_options,
            value="gte-small"
        ),
        gr.Textbox(
            label="Custom Model Path (if 'Custom' is selected)",
            value="",
            placeholder="Enter your custom model name/path"
        ),

        # Training dataset
        gr.Dropdown(
            label="Train Dataset (Dropdown)",
            choices=train_dataset_options,
            value="./dataset/train_dataset.json"
        ),
        gr.Textbox(
            label="Custom Train Dataset Path (if 'Custom' is selected)",
            value="./dataset/train_dataset.json",
            placeholder="Enter your custom train dataset path"
        ),
        gr.Checkbox(label="Shuffle Train Data", value=True),

        # Validation dataset
        gr.Dropdown(
            label="Val Dataset (Dropdown)",
            choices=val_dataset_options,
            value="./dataset/val_dataset.json"
        ),
        gr.Textbox(
            label="Custom Val Dataset Path (if 'Custom' is selected)",
            value="./dataset/val_dataset.json",
            placeholder="Enter your custom val dataset path"
        ),

        # Other training params
        gr.Number(label="Batch Size", value=64),
        gr.Number(label="Epochs", value=1),
        gr.Checkbox(label="Save Best Model", value=True),
        gr.Checkbox(label="Use Recommended Warmup Steps", value=True),
        gr.Number(label="Warmup Steps (fallback value)", value=10000),
        gr.Number(label="Evaluation Steps", value=1000),
        gr.Dropdown(label="Optimizer Class", choices=["Adam", "AdamW", "SGD"], value="AdamW"),
        gr.Textbox(label="Optimizer Params (LR)", value="5e-5"),
        gr.Dropdown(
            label="Scheduler",
            choices=["constantlr", "warmupconstant", "warmuplinear", "warmupcosine", "warmupcosinewithhardrestarts"],
            value="warmuplinear"
        ),
        gr.Textbox(label="Checkpoint Path", value="./checkpointpath"),
        gr.Number(label="Checkpoint Save Steps", value=100),
        gr.Number(label="Checkpoint Save Total Limit", value=5),
        gr.Checkbox(label="Show Progress Bar", value=True),
        gr.Textbox(label="Output Path", value="./outputpath"),
        gr.Textbox(label="Steps Per Epoch (leave blank for default)", placeholder="None"),
        gr.Number(label="Weight Decay", value=0.0),
        gr.Number(label="Max Grad Norm", value=1.0),
        gr.Checkbox(label="Use AMP", value=False)
    ],
    outputs="text",
    title="Retrieval Model Trainer",
    description="Select or enter custom paths for your model and datasets, and tweak training parameters. (Bi-encoders only now)"
)

if __name__ == "__main__":
    interface.launch()
