from typing import Any, Tuple


def get_nested_ref(dictionary: dict, key: str, sep: str = "/", default: Any = None) -> Tuple[dict, str]:
    """
    Retrieves a nested value from a dictionary using a key with a specified separator.

    Args:
        dictionary (dict): The dictionary to retrieve the value from.
        key (str): The key to retrieve the value from.
        sep (str, optional): The separator to use when splitting the key. Defaults to "/".
        default (any, optional): The default value to return if the key is not found. Defaults to None.

    Returns:
        (dict, str): A reference to the leaf dictionary that holds the final key, and the final key string.
    """
    if sep in key:
        first_key, other_keys = key.split(sep, 1)
        sub_dict = dictionary.setdefault(first_key, {})
        return get_nested_ref(sub_dict, other_keys, sep, default=default)

    if key not in dictionary:
        dictionary[key] = default() if callable(default) else default
    return dictionary, key
