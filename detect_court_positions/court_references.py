"""
Module: TennisCourtCoordinates

This module defines the coordinates of various key points on a tennis court.

The coordinates are represented as tuples of (x, y), where (0, 0) is the top-left corner of the court.
The units of the coordinates are pixels.

Coordinates:
- TOP_LEFT: The top-left corner of the tennis court.
- TOP_CENTER_LEFT: The top-center left point of the tennis court.
- TOP_CENTER_RIGHT: The top-center right point of the tennis court.
- TOP_RIGHT: The top-right corner of the tennis court.
- TOP_INNER_LEFT: The top inner left point of the tennis court.
- TOP_INNER_MIDDLE: The top inner middle point of the tennis court.
- TOP_INNER_RIGHT: The top inner right point of the tennis court.
- BOTTOM_INNER_LEFT: The bottom inner left point of the tennis court.
- BOTTOM_INNER_MIDDLE: The bottom inner middle point of the tennis court.
- BOTTOM_INNER_RIGHT: The bottom inner right point of the tennis court.
- BOTTOM_LEFT: The bottom-left corner of the tennis court.
- BOTTOM_CENTER_LEFT: The bottom-center left point of the tennis court.
- BOTTOM_CENTER_RIGHT: The bottom-center right point of the tennis court.
- BOTTOM_RIGHT: The bottom-right corner of the tennis court.

Usage:
    import TennisCourtCoordinates

    print(TennisCourtCoordinates.TOP_LEFT)  # Output: (582, 316)
    print(TennisCourtCoordinates.TOP_CENTER_LEFT)  # Output: (676, 316)
    # ...

Note: The coordinates are based on a specific representation of a tennis court and may vary depending on the actual court layout or scale used.
"""
TOP_LEFT = (582, 316)
TOP_CENTER_LEFT = (676, 316)
TOP_CENTER_RIGHT = (1237, 316)
TOP_RIGHT = (1332, 316)

TOP_INNER_LEFT = (646, 392)
TOP_INNER_MIDDLE = (957, 392)
TOP_INNER_RIGHT = (1268, 392)

BOTTOM_INNER_LEFT = (531, 666)
BOTTOM_INNER_MIDDLE = (958, 666)
BOTTOM_INNER_RIGHT = (1385, 666)

BOTTOM_LEFT = (279, 857)
BOTTOM_CENTER_LEFT = (452, 857)
BOTTOM_CENTER_RIGHT = (1468, 857)
BOTTOM_RIGHT = (1641, 857)
"""
