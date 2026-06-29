from person_id_pi.beverage_store import BeverageStore
from person_id_pi.beverage_types import BeverageEvent


def test_get_total_beer_count(tmp_path):
    store = BeverageStore(tmp_path / "store.json")
    event1 = BeverageEvent(
        event_id="event1",
        video_id="video1",
        frame_idx=0,
        track_id=1,
        user_id="alice",
        beverage_label="can",
        object_track_id=101,
        confidence=0.9,
        timestamp_utc="2024-01-01T00:00:00Z",
    )
    store.add_event(event1)
    event2 = BeverageEvent(
        event_id="event2",
        video_id="video1",
        frame_idx=10,
        track_id=1,
        user_id="alice",
        beverage_label="bottle",
        object_track_id=102,
        confidence=0.85,
        timestamp_utc="2024-01-01T00:00:10Z",
    )
    store.add_event(event2)
    dup_event1 = BeverageEvent(
        event_id="event1",  # Duplicate ID
        video_id="video1",
        frame_idx=20,
        track_id=1,
        user_id="alice",
        beverage_label="cup",
        object_track_id=103,
        confidence=0.8,
        timestamp_utc="2024-01-01T00:00:20Z",
    )
    assert not store.add_event(dup_event1)  # Should return False due to duplicate
    assert store.total_beers_by_user() == {"alice": 2}
    assert store.list_events() == [event1, event2]
