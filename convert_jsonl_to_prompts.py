import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from progrich import ProgressBar
from simple_parsing import choice, field

from config.entry import ConfigEntry

LANGUAGE_CODES = {
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "en": "English",
    "rm": "Romansh",
}

PROMPT_SIMPLE = """### Instruction
Translate the following sentence from {lang_source} to {lang_target} while respecting Swiss legal parlance.

### Input
{source}"""

PROMPT_GRPO = """### Instruction
Translate the following sentence from {lang_source} to {lang_target} while respecting Swiss legal parlance. Respond in the following format:
<think>
...
</think>
<translation>...</translation>

Explain your reasoning inside the <think> tag and give your final translation in the <translation> tag.

### Input
{source}"""


@dataclass
class CliConfig(ConfigEntry):
    # Paths to datasets as JSONL files.
    inputs: list[Path] = field(alias="-i")
    # Output directory to save the converted dataset.
    out_dir: Path = field(default=Path("data"), alias="-o")
    # Prompt template to use.
    prompt: Literal["simple", "grpo"] = choice(
        ["simple", "grpo"], default="simple", alias="-p"
    )


def create_prompt(
    data,
    prompt_template=PROMPT_SIMPLE,
):
    prompt = prompt_template.format(
        lang_source=LANGUAGE_CODES[data["src_lang"]],
        lang_target=LANGUAGE_CODES[data["tgt_lang"]],
        source=data["src_sent"],
    )
    return dict(question=prompt)


def main():
    cfg = CliConfig.parse_config()

    for input_path in ProgressBar.iter(cfg.inputs, desc="Converting"):
        with open(input_path) as fd:
            data_lines = [json.loads(line) for line in fd.readlines()]

        data_dir = cfg.out_dir / f"{input_path.stem}-{cfg.prompt}"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(
            cfg.out_dir / f"{data_dir.name}.tsv", "w", encoding="utf-8"
        ) as tsv_fd:
            writer = csv.writer(tsv_fd, delimiter="\t")
            total = len(data_lines)
            for i, data in enumerate(
                ProgressBar.iter(data_lines, desc=f"{data_dir.name}")
            ):
                json_path = data_dir / f"{i:0>{len(str(total))}}.json"
                with open(json_path, "w", encoding="utf-8") as out_fd:
                    json.dump(
                        dict(
                            **create_prompt(
                                data,
                                prompt_template=PROMPT_GRPO,
                            ),
                            answer=data["tgt_sent"],
                        ),
                        out_fd,
                        indent=2,
                        ensure_ascii=False,
                    )
                writer.writerow(
                    [
                        json_path.relative_to(cfg.out_dir),
                        data["src_lang"],
                        data["tgt_lang"],
                    ]
                )


if __name__ == "__main__":
    main()
