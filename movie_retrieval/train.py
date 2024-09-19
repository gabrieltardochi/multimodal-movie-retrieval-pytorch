import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv
from supertriplets.dataset import OnlineTripletsDataset, StaticTripletsDataset
from supertriplets.distance import EuclideanDistance
from supertriplets.encoder import PretrainedSampleEncoder
from supertriplets.evaluate import HardTripletsMiner, TripletEmbeddingsEvaluator
from supertriplets.loss import BatchHardTripletLoss
from supertriplets.models import load_pretrained_model
from supertriplets.sample import TextImageSample
from supertriplets.utils import move_tensors_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from movie_retrieval.config import TINY_MMIMDB_DATASET_PATH
from movie_retrieval.utils import prepare_tinymmimdb


def plot_results(training_stats: dict, save_path: str) -> None:
    # create plots
    _, axs = plt.subplots(1, 3, figsize=(18, 5))

    # plot 1: Train Batch Triplet Loss over time
    axs[0].plot(
        training_stats["train_batch_triplet_loss"], label="Triplet Loss", color="blue"
    )
    axs[0].set_title("Train Batch Triplet Loss Over Time")
    axs[0].set_xlabel("Batch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # plot 2: Validation Epoch Accuracy
    axs[1].plot(
        training_stats["valid_epoch_accuracy_cosine"], label="Cosine", marker="o"
    )
    axs[1].plot(
        training_stats["valid_epoch_accuracy_euclidean"], label="Euclidean", marker="o"
    )
    axs[1].plot(
        training_stats["valid_epoch_accuracy_manhattan"], label="Manhattan", marker="o"
    )
    axs[1].set_title("Validation Epoch Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    # plot 3: Test Final Accuracy (Bar Plot)
    test_accuracy = [
        training_stats["test_final_accuracy_cosine"],
        training_stats["test_final_accuracy_euclidean"],
        training_stats["test_final_accuracy_manhattan"],
    ]
    accuracy_labels = ["Cosine", "Euclidean", "Manhattan"]
    bars = axs[2].bar(accuracy_labels, test_accuracy, color=["red", "green", "blue"])
    axs[2].set_title("Test Final Accuracy")
    axs[2].set_ylabel("Accuracy")
    for bar in bars:
        height = bar.get_height()
        axs[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # saving
    plt.tight_layout()
    plt.savefig(save_path)


def prepare_tinymmimdb_split(split):
    df = prepare_tinymmimdb(dataset_path=TINY_MMIMDB_DATASET_PATH)
    df = df[df["split"] == split][
        ["plot_outline", "image_path", "genre_id"]
    ].reset_index(drop=True)
    return df


def get_triplet_embeddings(dataloader, model, device):
    model.eval()
    embeddings = {"anchors": [], "positives": [], "negatives": []}
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            for input_type in ["anchors", "positives", "negatives"]:
                inputs = {k: v for k, v in batch[input_type].items() if k != "label"}
                inputs = move_tensors_to_device(obj=inputs, device=device)
                batch_embeddings = model(**inputs).cpu()
                embeddings[input_type].append(batch_embeddings)
    embeddings = {k: torch.cat(v, dim=0).numpy() for k, v in embeddings.items()}
    return embeddings


@dataclass(frozen=True)
class TrainingConfig:
    in_batch_num_samples_per_label: int
    batch_size: int
    learning_rate: float
    num_epochs: int


def train(cfg: TrainingConfig) -> None:
    # setup experiment saving config
    current_datetime = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    save_path = os.path.join("artifacts", current_datetime)
    os.makedirs(save_path, exist_ok=False)

    # read dataset splits
    train = prepare_tinymmimdb_split("train")
    valid = prepare_tinymmimdb_split("dev")
    test = prepare_tinymmimdb_split("test")

    # might take too long on the cpu, make sure you have a gpu available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # our data points are pairs of (poster image, plot outline) labeled with the movie genre identifier
    train_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            train["plot_outline"], train["image_path"], train["genre_id"]
        )
    ]
    valid_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            valid["plot_outline"], valid["image_path"], valid["genre_id"]
        )
    ]
    test_examples = [
        TextImageSample(text=text, image_path=image_path, label=label)
        for text, image_path, label in zip(
            test["plot_outline"], test["image_path"], test["genre_id"]
        )
    ]

    # load a pretrained encoder just to help find hard triplets in valid and test set
    pretrained_encoder = PretrainedSampleEncoder(modality="text_english-image")
    valid_embeddings = pretrained_encoder.encode(
        examples=valid_examples, device=device, batch_size=cfg.batch_size
    )
    test_embeddings = pretrained_encoder.encode(
        examples=test_examples, device=device, batch_size=cfg.batch_size
    )
    del pretrained_encoder

    # this will efficiently find hard positives and negatives from a number of labeled samples
    hard_triplet_miner = HardTripletsMiner(use_gpu_powered_index_if_available=True)
    (
        valid_anchor_examples,
        valid_positive_examples,
        valid_negative_examples,
    ) = hard_triplet_miner.mine(
        examples=valid_examples,
        embeddings=valid_embeddings,
        normalize_l2=True,
        sample_from_topk_hardest=10,
    )
    (
        test_anchor_examples,
        test_positive_examples,
        test_negative_examples,
    ) = hard_triplet_miner.mine(
        examples=test_examples,
        embeddings=test_embeddings,
        normalize_l2=True,
        sample_from_topk_hardest=10,
    )
    del hard_triplet_miner

    # init model for finetuning, any torch nn would work
    model = load_pretrained_model(model_name="CLIPViTB32EnglishEncoder")
    model.to(device)
    model.eval()

    # train dataset will have its hard triplets mined for each training batch,
    # its a torch.utils.data.IterableDataset implementation
    trainset = OnlineTripletsDataset(
        examples=train_examples,
        in_batch_num_samples_per_label=cfg.in_batch_num_samples_per_label,
        batch_size=cfg.batch_size,
        sample_loading_func=model.load_input_example,
        sample_loading_kwargs={},
    )

    # validset and testset triplets were mined earlier, keep it static to measure performance improvements,
    # its a torch.utils.data.Dataset implementation
    validset = StaticTripletsDataset(
        anchor_examples=valid_anchor_examples,
        positive_examples=valid_positive_examples,
        negative_examples=valid_negative_examples,
        sample_loading_func=model.load_input_example,
        sample_loading_kwargs={},
    )
    testset = StaticTripletsDataset(
        anchor_examples=test_anchor_examples,
        positive_examples=test_positive_examples,
        negative_examples=test_negative_examples,
        sample_loading_func=model.load_input_example,
        sample_loading_kwargs={},
    )

    # init dataloaders
    trainloader = DataLoader(
        dataset=trainset, batch_size=cfg.batch_size, num_workers=0, drop_last=True
    )
    validloader = DataLoader(
        dataset=validset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    testloader = DataLoader(
        dataset=testset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # calculate initial anchor, positive and negative embeddings for valid and test datasets
    valid_triplet_embeddings = get_triplet_embeddings(
        dataloader=validloader, model=model, device=device
    )
    test_triplet_embeddings = get_triplet_embeddings(
        dataloader=testloader, model=model, device=device
    )

    # init evaluator, abstracts evaluating triplet embeddings using different distance metrics
    triplet_embeddings_evaluator = TripletEmbeddingsEvaluator(
        calculate_by_cosine=True,
        calculate_by_manhattan=True,
        calculate_by_euclidean=True,
    )

    # getting initial valid accuracy (means distance[anchor, positive] < distance[anchor, negative])
    valid_start_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=valid_triplet_embeddings["anchors"],
        embeddings_positives=valid_triplet_embeddings["positives"],
        embeddings_negatives=valid_triplet_embeddings["negatives"],
    )

    # init optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    # init triplet loss
    criterion = BatchHardTripletLoss(
        distance=EuclideanDistance(squared=False), margin=5
    )

    # init metrics tracking
    train_loss_progress = []
    valid_accuracy_progress = [valid_start_accuracy]

    # training loop
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for batch in tqdm(trainloader, total=len(trainloader), desc=f"Epoch {epoch}"):
            data = batch["samples"]
            labels = move_tensors_to_device(obj=data.pop("label"), device=device)
            inputs = move_tensors_to_device(obj=data, device=device)

            optimizer.zero_grad()

            embeddings = model(**inputs)
            loss = criterion(embeddings=embeddings, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            train_loss_progress.append(loss.item())

        # epoch valid metrics
        model.eval()
        valid_triplet_embeddings = get_triplet_embeddings(
            dataloader=validloader, model=model, device=device
        )
        valid_accuracy = triplet_embeddings_evaluator.evaluate(
            embeddings_anchors=valid_triplet_embeddings["anchors"],
            embeddings_positives=valid_triplet_embeddings["positives"],
            embeddings_negatives=valid_triplet_embeddings["negatives"],
        )
        valid_accuracy_progress.append(valid_accuracy)

    test_triplet_embeddings = get_triplet_embeddings(
        dataloader=testloader, model=model, device=device
    )
    test_final_accuracy = triplet_embeddings_evaluator.evaluate(
        embeddings_anchors=test_triplet_embeddings["anchors"],
        embeddings_positives=test_triplet_embeddings["positives"],
        embeddings_negatives=test_triplet_embeddings["negatives"],
    )

    training_stats = {
        "train_batch_triplet_loss": train_loss_progress,
        "valid_epoch_accuracy_cosine": [
            acc["accuracy_cosine"] for acc in valid_accuracy_progress
        ],
        "valid_epoch_accuracy_euclidean": [
            acc["accuracy_euclidean"] for acc in valid_accuracy_progress
        ],
        "valid_epoch_accuracy_manhattan": [
            acc["accuracy_manhattan"] for acc in valid_accuracy_progress
        ],
        "test_final_accuracy_cosine": test_final_accuracy["accuracy_cosine"],
        "test_final_accuracy_euclidean": test_final_accuracy["accuracy_euclidean"],
        "test_final_accuracy_manhattan": test_final_accuracy["accuracy_manhattan"],
    }

    # save training config
    json.dump(asdict(cfg), open(os.path.join(save_path, "training-config.json"), "w"))

    # save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model_state_dict.pt"))

    # plot results
    plot_results(
        training_stats=training_stats,
        save_path=os.path.join(save_path, "training-stats.png"),
    )

    # serialize training metrics into file
    json.dump(training_stats, open(os.path.join(save_path, "training-stats.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration parser")
    parser.add_argument(
        "--in_batch_num_samples_per_label",
        type=int,
        default=2,
        help="Number of samples per label in a batch (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Size of each batch (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training (default: 2e-5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs for training (default: 2)",
    )
    args = parser.parse_args()

    training_config = TrainingConfig(
        in_batch_num_samples_per_label=args.in_batch_num_samples_per_label,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )
    train(cfg=training_config)  # train
