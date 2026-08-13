from transformers import AutoProcessor

from dataset.chat.processing import MessageBoundaries

MODELS = [
    "unsloth/Qwen2-VL-7B-Instruct",
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    "unsloth/gemma-3-27b-it",
    "unsloth/Qwen3-VL-8B-Instruct",
    "unsloth/Qwen3.5-9B",
    "unsloth/gemma-4-12b-it",
]


# NOTE: Pixtral does not work for this.
def test_identify_assistant():
    for model in MODELS:
        processor = AutoProcessor.from_pretrained(model)
        # Nothing is done with this, since there are asssertions in that method, to make
        # sure that the assistant has been identifierd correctly.
        # So if the test passes, this means that the model worked.
        _ = MessageBoundaries.identify_assistant(processor)
