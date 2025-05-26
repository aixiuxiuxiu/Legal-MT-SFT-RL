def prefix_completions_with_prefill(
    completion_strs: list[str], prefill: str | None = None
) -> list[str]:
    """
    Prefix all completions with the prefilled text.

    When the answer is prefilled to force the model into starting with a certain
    structure, the output will only contain whatever comes after. Therefore, this
    prefixes the completions with the prefill, so that it gives the full expected
    responses.

    Args:
        completion_strs (listr[str]): List of completions
        prefill (str, optional): Prefill to prefix. Can be None for the cases where no
            prefill was used.

    Returns:
        out (list[str]): All completions prefixed with the prefill.
    """
    if prefill is None:
        return completion_strs
    else:
        return [prefill + comp_str for comp_str in completion_strs]
