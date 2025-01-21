import gradio as gr
import json

def update_config(
    model_name,
    train_dataset,
    val_dataset,
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
    # Convert steps_per_epoch to None if input is empty
    steps_per_epoch_val = None if steps_per_epoch == "" else int(steps_per_epoch)

    config = {
        "model_name": model_name,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
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

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    return "Configuration updated successfully!"

interface = gr.Interface(
    fn=update_config,
    inputs=[
        gr.Textbox(label="Model Name", value="gte-small"),
        gr.Textbox(label="Train Dataset Path", value="./dataset/train_dataset.json"),
        gr.Textbox(label="Validation Dataset Path", value="./dataset/val_dataset.json"),
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
    description="Add description here"
)

interface.launch()

