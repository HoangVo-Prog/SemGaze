from .prompt import build_where_prompt
from .serialization import serialize_xyd_record


def image_user(text):
    return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}


def build_where_conversation(episode):
    messages, images = [], []
    for record in (*episode.supports, episode.query):
        messages.append(image_user(build_where_prompt(record.task, len(record.x_px))))
        images.append(record.image_path)
        if record is not episode.query:
            messages.append({'role': 'assistant', 'content': serialize_xyd_record(record)})
    return messages, images
