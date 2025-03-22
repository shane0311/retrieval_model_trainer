from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerModelCardData

def load_model(model_name: str, device: str = "cpu"):
    if model_name == "gte-small":
        return SentenceTransformer("thenlper/gte-small", device=device,
                                        model_card_data=SentenceTransformerModelCardData(
                                            language="en",
                                            license="apache-2.0",
                                            model_name= f"{model_name}-ft",
                                        )
                                 )
    elif model_name == "bert-base-uncased":
        return SentenceTransformer("google-bert/bert-base-uncase", device=device,
                                        model_card_data=SentenceTransformerModelCardData(
                                            language="en",
                                            license="apache-2.0",
                                            model_name= f"{model_name}-ft",
                                        )
                                 )
    elif model_name == "roberta-base":
        return SentenceTransformer("FacebookAI/roberta-base", device=device,
                                        model_card_data=SentenceTransformerModelCardData(
                                            language="en",
                                            license="apache-2.0",
                                            model_name= f"{model_name}-ft",
                                        )
                                 )
    else:
        # Custom or local path
        return SentenceTransformer(model_name, device=device,
                                        model_card_data=SentenceTransformerModelCardData(
                                            language="en",
                                            license="apache-2.0",
                                            model_name= f"{model_name}-ft",
                                        )
                                 )