from semgaze.where.generation import generate_where


def evaluate_where_episode(bundle, episode, *, cache=None):
    return {'oracle_length_conditioned': True, **generate_where(bundle, episode, cache=cache)}
