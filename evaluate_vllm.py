import csv
import json
import time
from itertools import batched
from pathlib import Path

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

from config.evaluate import EvaluateConfig
from dataset.instruct import InstructDataset
from metric.functional import translation_chrf
from metric.metrics import TRANSLATION_CHRF
from metric.tracker import MetricTracker
from reward.translation import extract_translation


def main() -> None:
    cfg = EvaluateConfig.parse_config()
    torch.manual_seed(cfg.hardware.seed)

    print(f"== {cfg.model} (enable_thinking={not cfg.no_thinking})")

    model = LLM(
        cfg.model,
        reasoning_config=ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str="...\n\nI have to give the translation now based on my current thinking.</think>",
        ),
    )
    sampling_params = SamplingParams(
        max_tokens=cfg.max_new_tokens, thinking_token_budget=1024
    )
    dataset = InstructDataset(
        cfg.data,
        # No processor needed, as that is handled by vLLM but using the dataset makes
        # the data loading simpler.
        processor=None,  # pyright: ignore[reportArgumentType]
        prompts=cfg.prompts,
        random_prompt_probability=0.0,
        first_prompt_only=True,
        image_resizer=cfg.image.create_resizer(),
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

    stats: dict = dict(
        batch_size=cfg.hardware.batch_size,
        time_total=0.0,
        time_per_batch=[],
        tokens_per_batch=[],
    )
    global_start_time = time.time()
    metrics = MetricTracker([TRANSLATION_CHRF])
    for batch in tqdm(batched(dataset, cfg.hardware.batch_size)):  # pyright: ignore[reportArgumentType]
        messages = [sample.as_chat(include_answer=False) for sample in batch]  # pyright: ignore[reportAttributeAccessIssue]

        generation_start_time = time.time()
        outputs = model.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs=dict(enable_thinking=not cfg.no_thinking),
        )
        generation_time = time.time() - generation_start_time
        stats["time_per_batch"].append(generation_time)
        stats["tokens_per_batch"].append(
            sum(len(output.outputs[0].token_ids) for output in outputs)
        )

        preds = [out.outputs[0].text for out in outputs]
        pred_answers = [extract_translation(pred) or pred for pred in preds]
        metrics.append(
            dict(
                translation={
                    "chrf": translation_chrf(
                        pred_answers,
                        [
                            sample.answer  # pyright: ignore[reportAttributeAccessIssue]
                            for sample in batch
                        ],
                    )
                },
            )
        )

        for pred, pred_answer, sample in zip(preds, pred_answers, batch):
            writer.writerow(
                [
                    sample.info["path"].stem,  # pyright: ignore[reportAttributeAccessIssue]
                    pred_answer,
                    sample.answer,  # pyright: ignore[reportAttributeAccessIssue]
                    pred,
                ]
            )
        tsv_fd.flush()

    stats["time_total"] = time.time() - global_start_time
    mean_metrics = metrics.mean()
    mean_metrics["throughput"] = dict(
        time_total=stats["time_total"],
        time_generation=sum(stats["time_per_batch"]),
        tokens_total=sum(stats["tokens_per_batch"]),
        tokens_per_second=sum(stats["tokens_per_batch"]) / sum(stats["time_per_batch"]),
    )
    print(mean_metrics)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fd:
        json.dump(mean_metrics, fd, indent=2)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as fd:
        json.dump(stats, fd, indent=2)
    tsv_fd.close()


if __name__ == "__main__":
    main()
