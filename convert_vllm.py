from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field
from unsloth import FastModel

from config.entry import ConfigEntry


@dataclass
class ConvertConfig(ConfigEntry):
    # Name or path of the trained LLM
    model: Path = field(alias="-m")
    # Output directory to save the converted vLLM model.
    out_dir: Path = field(default=Path("checkpoints/vllm/"), alias="-o")


def main():
    cfg = ConvertConfig.parse_config()
    model, processor = FastModel.from_pretrained(str(cfg.model))

    model.save_pretrained_merged(
        cfg.out_dir / cfg.model.name, processor, save_method="merged_16bit"
    )


if __name__ == "__main__":
    main()
