import os

import numpy as np
import pandas as pd
import torch
from supertriplets.dataset import SampleEncodingDataset
from supertriplets.models import load_pretrained_model
from supertriplets.sample import TextImageSample
from supertriplets.utils import move_tensors_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_tinymmimdb(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(dataset_path, "data.csv"))
    df = df.reset_index(drop=True)
    df["plot_outline"] = df["plot outline"].astype(str)
    df["image_path"] = (
        df["image_path"]
        .apply(lambda x: os.path.join(dataset_path, "images", x))
        .astype(str)
    )
    df["genre_id"] = df["genre_id"].astype(int)
    df["movie_id"] = df.index
    df = df[
        [
            "movie_id",
            "plot_outline",
            "image_path",
            "genre",
            "genre_id",
            "title",
            "split",
        ]
    ]
    return df


def load_trained_multimodal_model(
    state_dict_path: str, device: str | torch.device
) -> torch.nn.Module:
    model = load_pretrained_model(model_name="CLIPViTB32EnglishEncoder")
    model.to(device)
    model.load_state_dict(
        torch.load(state_dict_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model


def get_embeddings_from_multimodal_model(
    examples: list[TextImageSample], model: torch.nn.Module, device: str | torch.device
) -> np.ndarray:
    model.to(device)
    model.eval()

    # our data points are pairs of (poster image, plot outline) labeled with the movie genre identifier
    dataset = SampleEncodingDataset(
        examples=examples,
        sample_loading_func=model.load_input_example,
    )

    # creating torch dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )

    # encoding dataset
    batch_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding samples"):
            inputs = batch["samples"]
            del inputs["label"]
            inputs["text_input"] = move_tensors_to_device(
                obj=inputs["text_input"], device=device
            )
            inputs["image_input"] = move_tensors_to_device(
                obj=inputs["image_input"], device=device
            )
            this_batch_embeddings = model(**inputs)
            batch_embeddings.append(this_batch_embeddings.cpu())
    embeddings = torch.cat(batch_embeddings, dim=0).numpy()
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, 0)
    return embeddings
