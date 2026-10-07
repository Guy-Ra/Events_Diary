"""
Visualize configured Flightradar24 UI regions on a reference screenshot.

This script is intended for manual UI-layout calibration and visual validation.
"""

from pathlib import Path

import cv2

from events_diary.config import UIConfig
from events_diary.debug.ui import draw_rois

INPUT_PATH = Path("data/reference_ui.png")
OUTPUT_PATH = Path("data/debug/ui_regions.png")


def main() -> None:
    config = UIConfig()

    frame = cv2.imread(str(INPUT_PATH))

    if frame is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_PATH}")

    regions = {
        "ticket_zone": config.ticket_zone,
        "ticket_header": config.ticket_header,
        "aircraft_image": config.aircraft_image,
        "ticket_scroll_body": config.ticket_scroll_body,
        "ticket_bottom_actions": config.ticket_bottom_actions,
        "map_analysis_region": config.map_analysis_region,
    }

    debug_frame = draw_rois(
        frame=frame,
        regions=regions,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        debug_frame,
    )

    if not success:
        raise RuntimeError(f"Could not write image: {OUTPUT_PATH}")

    print(f"Debug image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
