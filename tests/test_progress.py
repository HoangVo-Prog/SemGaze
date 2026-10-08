from semgaze.evaluation.progress import RollingRate, format_eta
from semgaze.training.loop import format_train_step


def test_rolling_rate_waits_for_stable_observations():
    rate = RollingRate(alpha=1.0, min_observations=2)
    rate.reset(now=0.0)
    rate.update(1, now=1.0)
    assert rate.eta(4) is None
    rate.update(2, now=2.0)
    assert rate.eta(4) == 4.0


def test_relative_eta_is_human_readable_and_timezone_independent():
    assert format_eta(None) == 'estimating...'
    assert format_eta(0) == '0m'
    assert format_eta(59) == '1m'
    assert format_eta(30 * 60) == '30m'
    assert format_eta(10 * 3600 + 42 * 60) == '10h 42m'
    assert RollingRate().eta(0) == 0.0


def test_training_progress_is_epoch_centric_and_has_no_step_timer():
    text = format_train_step({
        'step': 3500, 'epoch_index': 1, 'epoch_total': 5,
        'epoch_step': 3500, 'epoch_steps': 18508,
        'epoch_progress_pct': 18.9, 'epoch_eta_sec': 4355,
        'train_eta_sec': 9000,
        'loss_total': 3.4821, 'loss_where': 0.9214, 'loss_flat': 2.5607,
        'learning_rate': 3.21e-5, 'rejected_where_episodes': 0,
    })
    assert '[TRAIN] epoch 1/5 | 3500/18508 | 18.9%' in text
    assert 'ETA epoch=1h 13m' in text and 'ETA train=2h 30m' in text
    assert 'finish~' not in text and 'rejected=' not in text
    assert 'elapsed=' not in text and 'time=' not in text

def test_prediction_eta_does_not_treat_record_yield_bursts_as_inference():
    """Two completed 64-episode windows cost 8 minutes each.

    The 82nd and 123rd written records both belong to window two. A bogus
    update at 123 could imply thousands of episodes/sec. No window completed,
    so the inferred speed must remain unchanged until window three finishes.
    """
    rate = RollingRate(alpha=0.2, min_observations=2)
    rate.reset(now=0.0)
    rate.update(64, now=480.0)
    assert rate.eta(806 - 64) is None
    rate.update(128, now=960.0)
    second_window_eta = rate.eta(806 - 128)
    assert second_window_eta == 5085.0  # 678 episodes / (128 / 960 sec)
    # At record 123 we have merely written another record from window two:
    assert rate.eta(806 - 128) == second_window_eta
    rate.update(192, now=1440.0)
    assert rate.eta(806 - 192) == 4605.0

