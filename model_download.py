from sentence_transformers import SentenceTransformer

def load_model(model_name: str, device: str = "cpu"):
    if model_name == "gte-small":
        return SentenceTransformer("thenlper/gte-small", device=device)
    elif model_name == "bert-base-uncased":
        return SentenceTransformer("google-bert/bert-base-uncase", device=device)
    elif model_name == "roberta-base":
        return SentenceTransformer("FacebookAI/roberta-base", device=device)
    else:
        # Custom or local path
        return SentenceTransformer(model_name, device=device)