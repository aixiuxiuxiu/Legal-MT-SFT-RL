import copy
import os
from pathlib import Path

from config.train import TrainConfig

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
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from wandb.sdk.wandb_run import Run as WandbRun

from dataset import InstructDataset
from dataset.collate import InstructCollator
from model import vision
from trainer import InstructTrainer


def main() -> None:
    cfg = TrainConfig.parse_config()
    torch.manual_seed(cfg.hardware.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_main = True
    if num_processes > 1:
        dist.init_process_group(backend="nccl")
        is_main = dist.get_rank() == 0

    hardware_manager = cfg.hardware.create_manager()

    model, processor = vision.create_lora_model(
        cfg.model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.calculate_alpha(),
        pad_token=cfg.pad_token,
        device_map="auto" if cfg.hardware.split_model else None,
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
        cfg.train_data,
        processor=processor,
        prompts=cfg.prompts,
    )
    train_collator = InstructCollator(processor=processor)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=num_processes, shuffle=True)
        if num_processes > 1
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hardware.batch_size,
        num_workers=cfg.hardware.calculate_num_workers(),
        # Only shuffle when not using a sampler
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=hardware_manager.is_cuda(),
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=cfg.hardware.has_persistent_workers(),
        collate_fn=train_collator,
    )

    validation_processor = copy.deepcopy(processor)
    validation_processor.tokenizer.padding_side = "left"  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    validation_dataset = InstructDataset(
        cfg.validation_data,
        processor=validation_processor,
        prompts=cfg.prompts,
        first_prompt_only=True,
    )

    validation_collator = InstructCollator(
        processor=validation_processor, include_answer=False
    )
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
        batch_size=cfg.hardware.batch_size,
        num_workers=cfg.hardware.calculate_num_workers(),
        shuffle=False,
        sampler=validation_sampler,
        pin_memory=hardware_manager.is_cuda(),
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=cfg.hardware.has_persistent_workers(),
        collate_fn=validation_collator,
    )

    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg.lr.peak_lr,
        betas=(0.9, cfg.optim.adam_beta2),
        eps=cfg.optim.adam_eps,
        weight_decay=cfg.optim.weight_decay,
    )
    lr_scheduler = cfg.lr.create_scheduler(
        optimiser,
        train_data_loader,
        num_epochs=cfg.num_epochs,
    )

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    run: WandbRun | None = None
    if is_main:
        # Tags to make filtering easier. Some of them are a bit redundant, since you
        # could just filter by the config values, but these are the most interseting
        # ones, so you also don't need to go through all the config values.
        tags = [
            f"model:{model_config._name_or_path}",
            f"rank:{cfg.lora.rank}",
            *cfg.tags,
        ]
        # The run can be disabled, which returns a separate class.
        # In that case just set it to None, as there is no point in using it, so might
        # as well continue without one.
        # Also slightly needed for the type checking as `RunDisabled` is annoying.
        maybe_run = wandb.init(
            project=cfg.project,
            name=cfg.get_name(),
            config=cfg.to_dict(),
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
        save_dir=cfg.get_log_dir(base_dir=log_dir),
        wandb=run,
        hardware=hardware_manager,
        lr_scheduler=lr_scheduler,
        max_new_tokens=512,
        max_grad_norm=0.3,
        num_epochs=cfg.num_epochs,
    )

    trainer.train(
        train_data_loader,
        validation_data_loader,
    )

    if run:
        # Shouldn't be necessary, but just in case.
        run.finish()

    if num_processes > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
