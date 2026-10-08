from __future__ import annotations

import pytest

from skyguard.vision.classes import (
    SKYGUARD_COCO_IDS,
    SkyGuardClass,
    get_color,
    get_label_cn,
    get_name,
    get_skyguard_class_from_coco,
)


def test_skyguard_enum_values():
    assert SkyGuardClass.DRONE == 0
    assert SkyGuardClass.BIRD == 1
    assert SkyGuardClass.AIRPLANE == 2
    assert SkyGuardClass.HELICOPTER == 3
    assert SkyGuardClass.BALLOON == 4
    assert SkyGuardClass.KITE == 5


def test_coco_mapping_for_known_classes():
    # airplane, bird, kite exist in COCO
    assert SKYGUARD_COCO_IDS[SkyGuardClass.AIRPLANE] == 5
    assert SKYGUARD_COCO_IDS[SkyGuardClass.BIRD] == 14
    assert SKYGUARD_COCO_IDS[SkyGuardClass.KITE] == 38


def test_coco_mapping_for_unknown_classes():
    # drone, helicopter, balloon are not in COCO
    assert SKYGUARD_COCO_IDS[SkyGuardClass.DRONE] is None
    assert SKYGUARD_COCO_IDS[SkyGuardClass.HELICOPTER] is None
    assert SKYGUARD_COCO_IDS[SkyGuardClass.BALLOON] is None


def test_reverse_coco_lookup():
    assert get_skyguard_class_from_coco(5) == SkyGuardClass.AIRPLANE
    assert get_skyguard_class_from_coco(14) == SkyGuardClass.BIRD
    assert get_skyguard_class_from_coco(38) == SkyGuardClass.KITE
    assert get_skyguard_class_from_coco(999) is None


def test_names_and_labels():
    assert get_name(SkyGuardClass.DRONE) == "drone"
    assert get_label_cn(SkyGuardClass.DRONE) == "无人机"


def test_color_is_bgr_tuple():
    color = get_color(SkyGuardClass.DRONE)
    assert isinstance(color, tuple)
    assert len(color) == 3
    assert all(0 <= c <= 255 for c in color)


def test_invalid_class_raises():
    with pytest.raises(ValueError):
        get_name(99)
