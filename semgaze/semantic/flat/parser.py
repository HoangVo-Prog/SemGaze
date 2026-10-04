import re


def parse_flat_output(text, n, m):
    if n <= 0 or m <= 0:
        raise ValueError("positive N and M required")
    result = {"flat_format_valid": False, "what": {}, "why": {}, "how": None, "errors": []}
    order = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        match = re.fullmatch(r"(WHAT|WHY)\s+(\d+)\s*:\s*(.+?)\s*", line)
        how = re.fullmatch(r"HOW\s*:\s*(.+?)\s*", line)
        if match:
            kind, index, value = match.groups()
            index, key, value = int(index), kind.lower(), value.strip()
            order.append((kind, index))
            if not 1 <= index <= (n if kind == "WHAT" else m) or not value or index in result[key]:
                result["errors"].append(f"invalid/duplicate {kind} index or empty value: {line!r}")
            else:
                result[key][index] = value
        elif how:
            order.append(("HOW", 1))
            value = how.group(1).strip()
            if result["how"] is not None or not value:
                result["errors"].append("duplicate or empty HOW")
            else:
                result["how"] = value
        else:
            result["errors"].append(f"unrecognized line: {line!r}")
    expected = [("WHAT", t) for t in range(1, n + 1)] + [("WHY", t) for t in range(1, m + 1)] + [("HOW", 1)]
    if order != expected:
        result["errors"].append("missing, excess, out-of-range or unordered lines")
    result["flat_format_valid"] = not result["errors"]
    return result
