import gradio as gr
import json
from train import main as train_main

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
    train_batch_size,
    eval_batch_size,
    epochs,
    warmup_steps,
    warmup_ratio,
    evaluation_strategies,
    evaluation_steps,
    optimizer_class,
    optimizer_params_lr,
    scheduler,
    checkpoint_save_steps,
    checkpoint_save_total_limit,
    logging_strategies,
    logging_steps,
    output_path,
    max_steps,
    weight_decay,
    max_grad_norm
):
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

    # Build config
    config = {
        "model_name": final_model_name,
        "train_dataset": final_train_dataset,
        "shuffle_train_data": shuffle_train_data,
        "val_dataset": final_val_dataset,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "epochs": int(epochs),
        "warmup_steps": int(warmup_steps),
        "warmup_ratio": float(warmup_ratio),
        "evaluation_strategy": evaluation_strategies,
        "evaluation_steps": int(evaluation_steps),
        "optimizer_class": optimizer_class,
        "optimizer_params": {"lr": float(optimizer_params_lr)},
        "scheduler": scheduler,
        "checkpoint_save_steps": int(checkpoint_save_steps),
        "checkpoint_save_total_limit": int(checkpoint_save_total_limit),
        "logging_strategy": logging_strategies,
        "logging_steps": int(logging_steps),
        "output_path": output_path,
        "max_steps": int(max_steps) if max_steps else -1,
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm)
    }

    # Write to config.json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    return "Configuration updated successfully!"

def train_model():
    train_main()
    return "Training completed successfully!"

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Sentence Transformer Trainer")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration")

                model_dropdown = gr.Dropdown(
                    label="Model Name (Dropdown)",
                    choices=model_dropdown_options,
                    value="gte-small"
                )
                model_custom_text = gr.Textbox(
                    label="Custom Model Path (if 'Custom' is selected)",
                    value="",
                    placeholder="Enter your custom model name/path"
                )

                # Training dataset
                train_dropdown = gr.Dropdown(
                    label="Train Dataset (Dropdown)",
                    choices=train_dataset_options,
                    value="./dataset/train_dataset.json"
                )
                train_custom_text = gr.Textbox(
                    label="Custom Train Dataset Path (if 'Custom' is selected)",
                    value="./dataset/train_dataset.json",
                    placeholder="Enter your custom train dataset path"
                )
                shuffle_data = gr.Checkbox(label="Shuffle Train Data", value=True)

                # Validation dataset
                val_dropdown = gr.Dropdown(
                    label="Val Dataset (Dropdown)",
                    choices=val_dataset_options,
                    value="./dataset/val_dataset.json"
                )
                val_custom_text = gr.Textbox(
                    label="Custom Val Dataset Path (if 'Custom' is selected)",
                    value="./dataset/val_dataset.json",
                    placeholder="Enter your custom val dataset path"
                )

                # Other training params
                train_batch_size = gr.Number(label="Train Batch Size", value=64)
                eval_batch_size = gr.Number(label="Evaluation Batch Size", value=64)
                epochs = gr.Number(label="Epochs", value=1)
                warmup_steps = gr.Number(label="Warmup Steps (Overrides any effect of warmup_ratio, set it to -1 of not using)", value=-1)
                Warmup_ratio = gr.Number(label="Warmup Ratio", value=0.1)
                evaluation_strategies = gr.Dropdown(
                    label="Evaluation Strategies",
                    choices=["no", "steps", "epochs"],
                    value="steps"
                )
                evaluation_steps = gr.Number(label="Evaluation Steps", value=1000)
                optimizer_class = gr.Dropdown(label="Optimizers", 
                                              choices=["adamw_hf", "adamw_torch", "adamw_torch_fused", 
                                                       "adamw_apex_fused", "adamw_anyprecision", "adafactor"], 
                                              value="adamw_torch")
                optimizer_params_lr = gr.Textbox(label="Optimizer Params (LR)", value="5e-5")
                scheduler = gr.Dropdown(
                    label="Scheduler type",
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", 
                             "inverse_sqrt", "reduce_lr_on_plateau", "cosine_with_min_lr", "warmup_stable_decay"],
                    value="linear"
                )
                checkpoint_save_steps = gr.Number(label="Checkpoint Save Steps", value=100)
                checkpoint_save_total_limit = gr.Number(label="Checkpoint Save Total Limit", value=5)
                logging_strategies = gr.Dropdown(label="Logging Strategy", choices=["no", "steps", "epochs"], value="steps")
                logging_steps = gr.Number(label="Logging Steps", value=100)
                output_path = gr.Textbox(label="Output Path", value="./outputpath")
                max_steps = gr.Textbox(label="Max Steps", placeholder=-1)
                weight_decay = gr.Number(label="Weight Decay", value=0.0)
                max_grad_norm = gr.Number(label="Max Grad Norm", value=1.0)

                update_button = gr.Button("Update Config")
                update_status = gr.Textbox(label="Config Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Training")
                train_button = gr.Button("Train Model")
                train_status = gr.Textbox(label="Training Status", interactive=False)

        # Connect button clicks to functions
        update_button.click(
            fn=update_config,
            inputs=[
                model_dropdown, model_custom_text,
                train_dropdown, train_custom_text, shuffle_data,
                val_dropdown, val_custom_text,
                train_batch_size, eval_batch_size, epochs,
                warmup_steps, Warmup_ratio, evaluation_strategies, evaluation_steps, optimizer_class, optimizer_params_lr,
                scheduler, checkpoint_save_steps, checkpoint_save_total_limit, logging_strategies, logging_steps,
                output_path, max_steps, weight_decay, max_grad_norm
            ],
            outputs=[update_status]
        )

        train_button.click(
            fn=train_model,
            inputs=[],
            outputs=[train_status]
        )

    return demo

if __name__ == "__main__":
    demo_app = build_ui()
    demo_app.launch()