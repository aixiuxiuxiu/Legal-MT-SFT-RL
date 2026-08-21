from reward.translation import extract_translation

EXPECTED_TRANSLATION = "Expected Translation"


def test_extract_translation():
    response = f"<translation>{EXPECTED_TRANSLATION}</translation>"
    extracted = extract_translation(response)
    assert extracted == EXPECTED_TRANSLATION


def test_extract_translation_repeated_tag():
    response = f"""<think>
Desired format:
<think>...</think>
<translation>...</translation>
</think>
<translation>{EXPECTED_TRANSLATION}</translation>"""
    extracted = extract_translation(response)
    assert extracted == EXPECTED_TRANSLATION


def test_extract_translation_no_tag():
    response = "No Tags in this response"
    extracted = extract_translation(response)
    assert extracted is None
