import faiss
import numpy as np
import pandas as pd
import torch
from faiss import write_index
from supertriplets.sample import TextImageSample

from movie_retrieval.config import (
    TINY_MMIMDB_DATASET_PATH,
    TRAINED_MODEL_STATE_DICT_PATH,
)
from movie_retrieval.utils import (
    get_embeddings_from_multimodal_model,
    load_trained_multimodal_model,
    prepare_tinymmimdb,
)

if __name__ == "__main__":
    # load mmimdb
    df = prepare_tinymmimdb(dataset_path=TINY_MMIMDB_DATASET_PATH)

    # might take too long on the cpu, make sure you have a gpu available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load model
    model = load_trained_multimodal_model(
        state_dict_path=TRAINED_MODEL_STATE_DICT_PATH, device=device
    )

    # run model
    embeddings = get_embeddings_from_multimodal_model(
        examples=[
            TextImageSample(text=text, image_path=image_path, label=-1)
            for text, image_path in zip(df["plot_outline"], df["image_path"])
        ],
        model=model,
        device=device,
    )

    # creating FAISS index
    embed_dim = embeddings[0].size
    db_vectors = embeddings.copy().astype(np.float32)
    db_ids = df["movie_id"].values.astype(np.int64)

    faiss.normalize_L2(db_vectors)
    index = faiss.IndexFlatIP(embed_dim)
    index = faiss.IndexIDMap(index)  # mapping 'movie_id' as id
    index.add_with_ids(db_vectors, db_ids)

    # saving
    write_index(index, "movie-retrieval.index")
