from sentence_transformers import SentenceTransformer

def load_model(model_name: str, device: str = "cpu"):
    """
    Loads a model from huggingface if it's a known name,
    otherwise attempts to load from local/custom path.
    """
    # If you have your own actual HF repo for 'gte-small', update "moka-ai/gte-small" or similar.
    if model_name == "gte-small":
        return SentenceTransformer("moka-ai/gte-small", device=device)
    elif model_name == "bert-base-uncased":
        return SentenceTransformer("bert-base-uncased", device=device)
    elif model_name == "roberta-base":
        return SentenceTransformer("roberta-base", device=device)
    else:
        # Custom or local path
        return SentenceTransformer(model_name, device=device)