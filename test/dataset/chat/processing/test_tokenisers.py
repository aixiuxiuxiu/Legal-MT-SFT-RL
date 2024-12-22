from transformers import AutoProcessor

from dataset.chat.processing import MessageBoundaries

MODELS = [
    "unsloth/Qwen2-VL-7B-Instruct",
    "unsloth/Llama-3.2-11B-Vision-Instruct",
]


# NOTE: Pixtral does not work for this.
def test_identify_assistant():
    for model in MODELS:
        processor = AutoProcessor.from_pretrained(model)
        # Nothing is done with this, since there are asssertions in that method, to make
        # sure that the assistant has been identifierd correctly.
        # So if the test passes, this means that the model worked.
        _ = MessageBoundaries.identify_assistant(processor)
