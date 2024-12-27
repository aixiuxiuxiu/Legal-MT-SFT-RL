type NestedDict[T] = dict[str, T | NestedDict[T]]


def get_recursive[T](d: NestedDict[T], key: str) -> T | NestedDict[T] | None:
    out = d
    for k in key.split("."):
        if not isinstance(out, dict):
            raise KeyError(f"Cannot access {k!r} on type {type(out)} from key={key!r}")
        curr = out.get(k)
        if curr is None:
            return None
        out = curr
    return out


def set_recursive[T](d: NestedDict[T], key: str, value: T):
    keys = key.split(".")
    last_key = keys[-1]
    curr = d
    for k in keys[:-1]:
        if not isinstance(curr, dict):
            raise KeyError(f"Cannot access {k!r} on {curr} from key={key!r}")
        # Make sure the output dictionary has the nested dictionaries
        if k not in curr:
            curr[k] = {}
        curr = curr[k]
    if not isinstance(curr, dict):
        raise KeyError(f"Cannot access {last_key!r} on {curr} from key={key!r}")
    curr[last_key] = value


def nested_keys[T](d: NestedDict[T], keep_none: bool = True) -> list[str]:
    keys: list[str] = []
    for key, value in d.items():
        if isinstance(value, dict):
            keys.extend([f"{key}.{k}" for k in nested_keys(value, keep_none=keep_none)])
        elif keep_none or value is not None:
            keys.append(key)
    return keys
