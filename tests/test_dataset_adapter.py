from pathlib import Path
import pytest
pytest.importorskip('PIL')
from PIL import Image
from semgaze.data import cocosearch18


def test_duration_identity_and_verified_coordinate_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(cocosearch18, 'IMAGES_ROOT', tmp_path)
    Image.new('RGB', (80, 40)).save(tmp_path / 'image.jpg')
    raw = {'record_id': 'sample', 'stimulus_id': 'image.jpg', 'name': 'image.jpg', 'subject': 1,
           'task': 'car', 'condition': 'present', 'X': [10, 20], 'Y': [10, 20], 'T': [1200.5, -1],
           'prediction': {'fixations': [{'fixation': 1, 'what': 'a'}, {'fixation': 2, 'what': 'b'}],
                          'regions': [{'fixations': [1, 2], 'why': 'search'}], 'how': 'scan'}}
    adapter = cocosearch18.CocoSearch18Adapter(tmp_path, annotation_frame='original_image')
    record = adapter(raw)
    assert record.duration_ms == (1200.5, -1)  # no early clipping or unit conversion
    assert (record.image_width, record.image_height) == (80, 40)
    explicit = cocosearch18.CocoSearch18Adapter(tmp_path, annotation_frame={'width': 160, 'height': 80})(raw)
    assert explicit.image_width == 160
    with pytest.raises(ValueError, match='unresolved'):
        cocosearch18.CocoSearch18Adapter(tmp_path, annotation_frame=None)
    raw['name'] = 'missing.jpg'
    with pytest.raises(FileNotFoundError, match='found 0'):
        adapter(raw)
