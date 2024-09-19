import os
import tempfile

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from faiss import read_index
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

# load mmimdb
df = prepare_tinymmimdb(dataset_path=TINY_MMIMDB_DATASET_PATH)
df.set_index("movie_id", inplace=True)

# prefer gpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load model
model = load_trained_multimodal_model(
    state_dict_path=TRAINED_MODEL_STATE_DICT_PATH, device=device
)

# load movie retrieval index
index = read_index("movie-retrieval.index")

# streamlit stuff
st.title("MultiModal Movie Retrieval :movie_camera:")
st.text("https://github.com/gabrieltardochi/multimodal-movie-retrieval-pytorch")
st.subheader("", divider="rainbow")

# input fields for image URL, format, and name
poster_image = st.file_uploader("Upload poster image")
plot_outline = st.text_area("Enter movie plot outline", max_chars=500, height=250)

if st.button("Search"):
    # encode (poster image, plot outline) pair
    with tempfile.TemporaryDirectory() as tmpdirname:
        poster_image_bytes = poster_image.getvalue()
        poster_path = os.path.join(tmpdirname, "img.jpg")

        with open(poster_path, "wb") as file:
            file.write(poster_image_bytes)

        # encode input
        embeddings = get_embeddings_from_multimodal_model(
            examples=[
                TextImageSample(text=plot_outline, image_path=poster_path, label=-1)
            ],
            model=model,
            device=device,
        )

    # searching with input embeddings
    find_k_most_similar = 3

    faiss.normalize_L2(embeddings)
    similarities, similarities_ids = index.search(embeddings, k=find_k_most_similar)

    similarities = similarities[0]  # we are only searching with a single datapoint
    similarities_ids = similarities_ids[
        0
    ]  # we are only searching with a single datapoint
    similarities = np.around(np.clip(similarities, 0, 1), decimals=4)

    df.loc[similarities_ids, "similarity_score"] = similarities

    # build output
    results = {}
    for k in range(find_k_most_similar):
        results[f"top{k+1}-similar-retrieved"] = df.loc[similarities_ids[k]].to_dict()

    # return
    st.success("Retrieved successfully.")
    st.json(
        results,
        expanded=2,
    )
