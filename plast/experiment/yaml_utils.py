import re


def dump_yaml(data, indent=0):
    """
    Serializes a Python dict/list/scalar to YAML format.
    """
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                lines.append(dump_yaml(v, indent + 2))
            elif isinstance(v, (list, tuple)):
                lines.append(" " * indent + f"{k}:")
                for item in v:
                    if isinstance(item, dict):
                        # List of dicts
                        lines.append(" " * (indent + 2) + "-")
                        lines.append(dump_yaml(item, indent + 4))
                    else:
                        lines.append(" " * (indent + 2) + f"- {_dump_scalar(item)}")
            else:
                lines.append(" " * indent + f"{k}: {_dump_scalar(v)}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                lines.append(" " * indent + "-")
                lines.append(dump_yaml(item, indent + 2))
            else:
                lines.append(" " * indent + f"- {_dump_scalar(item)}")
    else:
        lines.append(" " * indent + _dump_scalar(data))

    return "\n".join([line for line in lines if line is not None])


def _dump_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    # Check if string contains special chars, if so quote it
    s = str(v)
    if any(c in s for c in ":#-,[]{}&*!|>\"'"):
        return f'"{s}"'
    return s


def load_yaml(content):
    """
    Parses a simple YAML string into a Python dictionary.
    Supports indentation, dictionaries, and simple lists.
    """
    lines = content.split("\n")
    data = {}
    stack = [(-1, data)]  # list of (indent, dict_or_list_ref)

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        # Keep poping stack until we find the parent container
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        current_container = stack[-1][1]

        if stripped.startswith("-"):
            # List item
            val_str = stripped[1:].strip()
            # If the current container isn't a list, convert the last key of the parent dict to a list
            if isinstance(current_container, dict):
                # This should not happen in a clean YAML, but handle it
                raise ValueError(f"YAML Parse Error: list item found at line {line_num}")

            # If val_str contains key: value, it's a dict item inside a list
            if ":" in val_str and not (val_str.startswith('"') or val_str.startswith("'")):
                # Nested dict
                new_dict = {}
                current_container.append(new_dict)
                stack.append((indent + 2, new_dict))
                # Process the key: value
                k_str, v_str = val_str.split(":", 1)
                k_str = k_str.strip()
                v_str = v_str.strip()
                new_dict[k_str] = _parse_scalar(v_str)
            else:
                current_container.append(_parse_scalar(val_str))
        else:
            if ":" in stripped:
                k_str, v_str = stripped.split(":", 1)
                k_str = k_str.strip()
                # Clean quotes if any
                if (k_str.startswith('"') and k_str.endswith('"')) or (
                    k_str.startswith("'") and k_str.endswith("'")
                ):
                    k_str = k_str[1:-1]
                v_str = v_str.strip()

                if not v_str:
                    # It's either a nested dictionary or a nested list
                    # Peek next line to see if it starts with '-' (list) or 'key:' (dict)
                    is_list = False
                    for next_line in lines[line_num + 1 :]:
                        if next_line.strip() and not next_line.strip().startswith("#"):
                            if next_line.strip().startswith("-"):
                                is_list = True
                            break

                    if is_list:
                        new_container = []
                    else:
                        new_container = {}

                    current_container[k_str] = new_container
                    stack.append((indent, new_container))
                else:
                    current_container[k_str] = _parse_scalar(v_str)
            else:
                # Scalar line (should not happen in dict-based yaml)
                pass

    return data


def _parse_scalar(s):
    if not s:
        return None
    if s.lower() == "null":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # Check quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # Try integer
    try:
        return int(s)
    except ValueError:
        pass

    # Try float
    try:
        return float(s)
    except ValueError:
        pass

    # Return string
    return s
