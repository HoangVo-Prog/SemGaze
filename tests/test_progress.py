from datetime import datetime, timezone

from semgaze.evaluation.progress import RollingRate, format_finish_time
from semgaze.training.loop import format_train_step


def test_rolling_rate_waits_for_stable_observations():
    rate = RollingRate(alpha=1.0, min_observations=2)
    rate.reset(now=0.0)
    rate.update(1, now=1.0)
    assert rate.eta(4) is None
    rate.update(2, now=2.0)
    assert rate.eta(4) == 4.0


def test_finish_time_uses_local_clock_date_boundary():
    now = datetime(2026, 10, 6, 23, 30, tzinfo=timezone.utc)
    assert format_finish_time(30 * 60, now=now) == 'Oct 07 00:00'


def test_training_progress_is_epoch_centric_and_has_no_step_timer():
    text = format_train_step({
        'step': 3500, 'epoch_index': 1, 'epoch_total': 5,
        'epoch_step': 3500, 'epoch_steps': 18508,
        'epoch_progress_pct': 18.9, 'epoch_eta_sec': 4355,
        'loss_total': 3.4821, 'loss_where': 0.9214, 'loss_flat': 2.5607,
        'learning_rate': 3.21e-5, 'rejected_where_episodes': 0,
    })
    assert '[TRAIN] epoch 1/5 | 3500/18508 | 18.9%' in text
    assert 'ETA epoch=' in text and 'finish~' in text
    assert 'elapsed=' not in text and 'time=' not in text
