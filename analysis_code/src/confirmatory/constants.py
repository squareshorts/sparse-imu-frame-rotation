from __future__ import annotations

SENSOR_NAMES = ("pelvis", "thigh", "shank", "foot")
SENSOR_INDEX = {name: i for i, name in enumerate(SENSOR_NAMES)}

PRIMARY_TASKS = {
    "knee_two": {
        "sensors": (1, 2),
        "targets": (5, 6),
        "target_names": ("knee_stance_flex_deg", "knee_swing_flex_deg"),
    },
    "knee_thigh": {
        "sensors": (1,),
        "targets": (5, 6),
        "target_names": ("knee_stance_flex_deg", "knee_swing_flex_deg"),
    },
    "knee_shank": {
        "sensors": (2,),
        "targets": (5, 6),
        "target_names": ("knee_stance_flex_deg", "knee_swing_flex_deg"),
    },
    "ankle_two": {
        "sensors": (2, 3),
        "targets": (7, 8),
        "target_names": ("ankle_dorsiflex_deg", "ankle_plantarflex_deg"),
    },
    "ankle_shank": {
        "sensors": (2,),
        "targets": (7, 8),
        "target_names": ("ankle_dorsiflex_deg", "ankle_plantarflex_deg"),
    },
    "ankle_foot": {
        "sensors": (3,),
        "targets": (7, 8),
        "target_names": ("ankle_dorsiflex_deg", "ankle_plantarflex_deg"),
    },
}

MODEL_SELECTION_TASKS = ("knee_two", "ankle_two")
PRIMARY_TARGET_INDICES = (5, 6, 7, 8)
PRIMARY_TARGET_NAMES = (
    "knee_stance_flex_deg",
    "knee_swing_flex_deg",
    "ankle_dorsiflex_deg",
    "ankle_plantarflex_deg",
)

GRF_TARGET_INDICES = (0, 1, 2, 3, 4)
