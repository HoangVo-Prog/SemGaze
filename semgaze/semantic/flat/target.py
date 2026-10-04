from semgaze.data.semantic import validate_semantic


def flatten_text(text):
    result = " ".join(text.strip().split())
    if not result:
        raise ValueError("empty semantic text")
    return result


def build_flat_target(semantic):
    validate_semantic(semantic, len(semantic.what), "flat_target")
    return "\n".join([*(f"WHAT {t}: {flatten_text(w)}" for t, w in enumerate(semantic.what, 1)),
                      *(f"WHY {m}: {flatten_text(g.why)}" for m, g in enumerate(semantic.why_groups, 1)),
                      f"HOW: {flatten_text(semantic.how)}"])
