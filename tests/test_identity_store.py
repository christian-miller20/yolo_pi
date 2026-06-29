import numpy as np

from person_id_pi.identity_store import IdentityStore


def test_rename_user_success_preserves_templates(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    renamed = store.rename_user("alice", "alicia")

    assert renamed is True
    assert store.has_user("alice") is False
    assert store.has_user("alicia") is True
    templates = store.get_templates("alicia")
    assert len(templates) == 1
    assert np.allclose(templates[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_rename_user_rejects_existing_target(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    store.add_template("bob", np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    renamed = store.rename_user("alice", "bob")

    assert renamed is False
    assert store.has_user("alice") is True
    assert store.has_user("bob") is True


def test_rename_user_rejects_missing_source(tmp_path):
    store = IdentityStore(tmp_path / "store.json")

    renamed = store.rename_user("missing", "new_id")

    assert renamed is False
    assert store.has_user("new_id") is False


def test_rename_user_same_name_is_no_op_success(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_user("alice")

    renamed = store.rename_user("alice", "alice")

    assert renamed is True
    assert store.has_user("alice") is True


def test_load_skips_malformed_records(tmp_path):
    path = tmp_path / "store.json"
    path.write_text('[{"user_id": "alice", "templates": [[1, 0]]}, {"templates": []}]')

    store = IdentityStore(path)

    assert store.list_users() == ["alice"]


def test_display_name_round_trip_and_uniqueness(tmp_path):
    path = tmp_path / "store.json"
    store = IdentityStore(path)
    store.add_user("user_0000")
    store.add_user("user_0001")

    assert store.set_display_name("user_0000", "Alice") is True
    assert store.set_display_name("user_0001", "alice") is False

    reloaded = IdentityStore(path)
    assert reloaded.get_display_name("user_0000") == "Alice"
    assert reloaded.display_label("user_0000") == "Alice"
    assert reloaded.display_label("user_0001") == "user_0001"


def test_external_label_survives_running_store_save(tmp_path):
    path = tmp_path / "store.json"
    running = IdentityStore(path)
    running.add_user("user_0000")
    reviewer = IdentityStore(path)
    assert reviewer.set_display_name("user_0000", "Alice") is True

    running.add_template("user_0000", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    reloaded = IdentityStore(path)
    assert reloaded.get_display_name("user_0000") == "Alice"
