"""Approximate, pixel-coordinate geometry extracted from segmentation masks."""

from floorplan_di.geometry.entity_linking import link_text_to_rooms
from floorplan_di.geometry.room_graph import link_doors_to_rooms
from floorplan_di.geometry.vectorize import vectorize_mask

__all__ = ["link_doors_to_rooms", "link_text_to_rooms", "vectorize_mask"]
