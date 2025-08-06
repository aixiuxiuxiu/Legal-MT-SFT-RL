import csv
import json
import os
from pathlib import Path

import torch
from progrich import ProgressBar
from torch.utils.data import DataLoader

from config.evaluate import EvaluateConfig
from dataset import InstructDataset
from dataset.collate import InstructCollator
from dataset.prefill import prefix_completions_with_prefill
from metric import MetricTracker
from metric.functional import classification_accuracy
from metric.metrics import CLASS_ACCURACY, CLASS_ACCURACY_UNCASED
from model.loader import load_model_for_inference
from model.utils import unwrap_tokeniser
from reward.classification import extract_answer


@torch.inference_mode()
def main() -> None:
    cfg = EvaluateConfig.parse_config()
    torch.manual_seed(cfg.hardware.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hardware_manager = cfg.hardware.create_manager()
    model, processor = load_model_for_inference(
        cfg.model,
        pad_token=cfg.pad_token,
        device_map="auto" if cfg.hardware.split_model else None,
    )

    dataset = InstructDataset(
        cfg.data,
        processor=processor,
        prompts=cfg.prompts,
        random_prompt_probability=0.0,
        first_prompt_only=True,
        image_resizer=cfg.image.create_resizer(),
    )
    collator = InstructCollator(
        processor=processor, include_answer=False, prefill=cfg.prefill
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.hardware.batch_size,
        num_workers=cfg.hardware.calculate_num_workers(),
        shuffle=False,
        pin_memory=not cfg.hardware.no_pin_memory and hardware_manager.is_cuda(),
        collate_fn=collator,
    )

    model_path = Path(cfg.model)
    cp_name = model_path.name
    if cp_name in ["best", "latest"]:
        # When it is the best/latest model that was created from the training, the
        # parent directory's name is used instead, since that is the actual name of the
        # experiment.
        cp_name = model_path.parent.name
    out_dir = cfg.out_dir / cp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_fd = open(out_dir / "predictions.tsv", "w", encoding="utf-8")
    writer = csv.writer(tsv_fd, delimiter="\t")

    tokeniser = unwrap_tokeniser(processor)
    metrics = MetricTracker([CLASS_ACCURACY, CLASS_ACCURACY_UNCASED])
    with ProgressBar("Evaluating", total=len(dataset), persist=True) as pbar:
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            inputs = batch.data.to(hardware_manager.device)
            with hardware_manager.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokeniser.pad_token_id,
                )
            preds = [
                tokeniser.decode(out[input_ids.size(0) :], skip_special_tokens=True)
                for input_ids, out in zip(inputs.input_ids, outputs)
            ]
            preds = prefix_completions_with_prefill(preds, prefill=cfg.prefill)
            pred_answers = [extract_answer(pred) or pred for pred in preds]
            metrics.append(
                dict(
                    accuracy={
                        "class": classification_accuracy(
                            pred_answers, batch.answers, ignore_case=False
                        ),
                        "class_uncased": classification_accuracy(
                            pred_answers, batch.answers, ignore_case=True
                        ),
                    },
                )
            )

            for pred, pred_answer, answer, path in zip(
                preds, pred_answers, batch.answers, batch.info["path"]
            ):
                writer.writerow([path.stem, pred_answer, answer, pred])
            tsv_fd.flush()
            pbar.advance(curr_batch_size)

    mean_metrics = metrics.mean()
    print(mean_metrics)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fd:
        json.dump(mean_metrics, fd, indent=2)

    tsv_fd.close()


if __name__ == "__main__":
    main()
