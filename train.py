import copy
import os
import typing
from argparse import ArgumentParser
from pathlib import Path

# ruff: noqa: E402 (Disable import at top lint, because of this workaround)
# unsloth hardcodes "cuda:0" in an attempt to disallow multi-GPU as they want to
# force you to buy their pro version, so as a workaround you have to limit the visible
# GPUs with CUDA_VISIBLE_DEVICES.
# This needs to be done before loading torch as the environment variable is only
# respected once when setting up the GPU.
num_processes = int(os.environ.get("WORLD_SIZE", "1"))
if num_processes > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", "0")


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from wandb.sdk.wandb_run import Run as WandbRun

from dataset import InstructDataset
from dataset.collate import InstructCollator
from lr_scheduler import create_lr_scheduler
from model import vision
from trainer import InstructTrainer
from utils.hardware import HardwareConfig, MixedPrecisionChoice


def parse_args():
    parser = ArgumentParser(description="Fine-tune a Vision Language Model (VLM)")
    parser.add_argument(
        "--train-data",
        dest="train_data",
        type=Path,
        required=True,
        help=(
            "Path to train dataset, either a directory with JSON files or a "
            "TSV file listing the JSON files to use."
        ),
    )
    parser.add_argument(
        "--validation-data",
        dest="validation_data",
        type=Path,
        required=True,
        help=(
            "Path to validation dataset, either a directory with JSON files or a "
            "TSV file listing the JSON files to use."
        ),
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        required=True,
        help="Name of the experiment for the logging and saved model",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        required=True,
        help="Name or path of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--pad-token",
        dest="pad_token",
        help=(
            "Set the a different padding token, as sometimes the one defined in the "
            "model config may not work correctly, e.g. when it's the <eos> token, the "
            "model would just never learn when to stop."
        ),
    )
    parser.add_argument(
        "--split-model",
        dest="split_model",
        action="store_true",
        help=(
            "Split the model across available GPUs. Not allowed when launching "
            "the training script distributed, as that would be conflicting."
        ),
    )
    parser.add_argument(
        "-l",
        "--lr",
        dest="lr",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "-r", "--rank", dest="lora_rank", type=int, default=16, help="Lora rank"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        dest="lora_alpha",
        type=float,
        help="Lora alpha scaling factor. [Default: 2·rank]",
    )
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        type=str,
        default="auto",
        choices=typing.get_args(MixedPrecisionChoice),
        help=(
            "Type for mixed-precision. By default it will use auto, which will use "
            "bf16 if the GPU supports it, and otherwise fallback to fp16. "
            "To disable it, use none as the argument."
        ),
    )
    parser.add_argument(
        "--project",
        dest="project",
        type=str,
        default="vision-finetune",
        help="Name of the project for the logging [Default: vision-finetune]",
    )
    parser.add_argument(
        "--tags",
        dest="tags",
        type=str,
        default=[],
        nargs="+",
        help="Additional tags to add to the run in wandb",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=1234, help="Random seed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_main = True
    if num_processes > 1:
        dist.init_process_group(backend="nccl")
        is_main = dist.get_rank() == 0

    hardware_config = HardwareConfig(mixed_precision=args.mixed_precision)

    model, processor = vision.create_lora_model(
        args.model,
        rank=args.lora_rank,
        alpha=args.lora_alpha if args.lora_alpha else 2 * args.lora_rank,
        pad_token=args.pad_token,
        device_map="auto" if args.split_model else None,
    )
    # Saving this to avoid having to unwrap the DDP model just to log its config.
    model_config = model.config

    if num_processes > 1:
        # Always device 0, because it needs to be limited with CUDA_VISIBLE_DEVICES, as
        # unsloth deliberately uses "cuda:0" in the attempt of disallowing multi-gpu.
        model = DistributedDataParallel(
            model, device_ids=[0], find_unused_parameters=False
        )

    train_dataset = InstructDataset(
        args.train_data,
        processor=processor,
    )
    train_collator = InstructCollator(processor=processor)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=num_processes, shuffle=True)
        if num_processes > 1
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=mp.cpu_count() // num_processes,
        # Only shuffle when not using a sampler
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=hardware_config.is_cuda(),
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=True,
        collate_fn=train_collator,
    )

    validation_processor = copy.deepcopy(processor)
    validation_processor.padding_side = "left"
    validation_dataset = InstructDataset(
        args.validation_data,
        processor=validation_processor,
    )

    validation_collator = InstructCollator(processor=validation_processor)
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas=num_processes,
            # Don't shuffle for the validation set.
            shuffle=False,
        )
        if num_processes > 1
        else None
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=mp.cpu_count() // num_processes,
        shuffle=False,
        sampler=validation_sampler,
        pin_memory=hardware_config.is_cuda(),
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=True,
        collate_fn=validation_collator,
    )

    optimiser = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = create_lr_scheduler(
        "cos",
        optimiser,
        peak_lr=args.lr,
        # Warmup for 20% of the first epoch.
        # TODO: Should probably be adapted to the size of the dataset
        warmup_steps=len(train_data_loader) // 5,
        total_steps=len(train_data_loader) * args.num_epochs,
        end_lr=1e-8,
        # To not crash when choosing schedulers that don't support all arguments.
        allow_extra_args=True,
    )

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    run: WandbRun | None = None
    if is_main:
        # Tags to make filtering easier. Some of them are a bit redundant, since you
        # could just filter by the config values, but these are the most interseting
        # ones, so you also don't need to go through all the config values.
        tags = [
            # For some reason pyright thinks it's a dict, but it's not and the config is
            # actually not subscriptable.
            f"model:{model_config._name_or_path}",
            f"rank:{args.lora_rank}",
            *args.tags,
        ]
        # The run can be disabled, which returns a separate class.
        # In that case just set it to None, as there is no point in using it, so might
        # as well continue without one.
        # Also slightly needed for the type checking as `RunDisabled` is annoying.
        maybe_run = wandb.init(
            project=args.project,
            name=args.name,
            config=vars(args),
            tags=tags,
            dir=str(log_dir.resolve()),
        )
        if isinstance(maybe_run, WandbRun):
            run = maybe_run
            # By defining the metrics, the runs will show the best values for these.
            run.define_metric("validation.accuracy", summary="max")
            run.define_metric("validation.f1", summary="max")

    trainer = InstructTrainer(
        model=model,
        optimiser=optimiser,
        processor=processor,
        save_dir=log_dir / args.name,
        wandb=run,
        hardware=hardware_config,
        lr_scheduler=lr_scheduler,
        max_new_tokens=512,
        max_grad_norm=0.3,
        num_epochs=args.num_epochs,
    )

    trainer.train(
        train_data_loader,
        validation_data_loader,
    )

    if run:
        # Shouldn't be necessary, but just in case.
        run.finish()


if __name__ == "__main__":
    main()
