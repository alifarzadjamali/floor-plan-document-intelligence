from floorplan_di.geometry.entity_linking import link_text_to_rooms
from floorplan_di.geometry.room_graph import link_doors_to_rooms


def test_text_is_linked_only_when_inside_or_conservatively_near() -> None:
    rooms = [{"id": "R01", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]}]
    tokens = [
        {"id": "T01", "bbox": [10, 10, 10, 10]},
        {"id": "T02", "bbox": [200, 200, 10, 10]},
    ]
    links = link_text_to_rooms(tokens, rooms, nearest_threshold=10)
    assert links == [
        {"text_id": "T01", "room_id": "R01", "method": "contains", "distance_pixels": 15.0}
    ]


def test_door_link_requires_two_nearby_rooms() -> None:
    rooms = [
        {"id": "R01", "polygon": [[0, 0], [40, 0], [40, 40], [0, 40]]},
        {"id": "R02", "polygon": [[60, 0], [100, 0], [100, 40], [60, 40]]},
    ]
    links, graph = link_doors_to_rooms([{"id": "D01", "centroid": [50, 20]}], rooms, threshold=15)
    assert links[0]["room_ids"] == ["R01", "R02"]
    assert graph == [{"door_id": "D01", "rooms": ["R01", "R02"]}]
