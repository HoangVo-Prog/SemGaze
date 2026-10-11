from .prompt import build_where_prompt
from .serialization import serialize_xyd_record


SUPPORT_PROMPT_STYLES = ('full', 'concise')


def image_user(text):
    return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}


def build_where_support_prompt(task, n):
    """Exact B2 support prompt; the final query always uses the full WHERE instruction."""
    return (f'Support reference. Task: {task}. '
            f'Fixation count: {n}. Follow the query response format.')


def build_where_conversation(episode, end_fix_token="<END_FIX>", *, support_prompt_style="concise"):
    """Use B2 concise support instructions by default, preserving every support GT."""
    if support_prompt_style not in SUPPORT_PROMPT_STYLES:
        raise ValueError(f'invalid WHERE support_prompt_style: {support_prompt_style!r}')
    messages, images = [], []
    for record in episode.supports:
        prompt = (build_where_prompt(record.task, len(record.x_px), end_fix_token)
                  if support_prompt_style == 'full'
                  else build_where_support_prompt(record.task, len(record.x_px)))
        messages.append(image_user(prompt))
        images.append(record.image_path)
        messages.append({'role': 'assistant', 'content': serialize_xyd_record(record, end_fix_token)})
    query = episode.query
    messages.append(image_user(build_where_prompt(query.task, len(query.x_px), end_fix_token)))
    images.append(query.image_path)
    return messages, images
