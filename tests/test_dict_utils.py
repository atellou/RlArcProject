from rlarcworld.utils.dict_utils import get_nested_ref


def test_get_nested_ref_creates_nested_dicts():
    d = {}
    ref, leaf_key = get_nested_ref(d, "a/b/c")
    # Should have created nested dicts a -> b -> {}
    assert "a" in d and isinstance(d["a"], dict)
    assert "b" in d["a"] and isinstance(d["a"]["b"], dict)
    # Returned reference should be the leaf container and key should be 'c'
    assert ref is d["a"]["b"]
    assert leaf_key == "c"
    # Assign through the reference and ensure the original dict sees it
    ref[leaf_key] = 42
    assert d["a"]["b"]["c"] == 42


def test_get_nested_ref_default_callable():
    d = {}

    def make_list():
        return []

    # First call should create with default callable
    ref, leaf = get_nested_ref(d, "x/y", default=make_list)
    assert ref is d["x"]
    assert leaf == "y"
    assert d["x"]["y"] == []

    # Second call should not overwrite existing value
    ref2, leaf2 = get_nested_ref(d, "x/y", default=lambda: [1, 2, 3])
    assert ref2 is ref and leaf2 == leaf
    assert d["x"]["y"] == []


def test_get_nested_ref_custom_separator():
    d = {}
    ref, key = get_nested_ref(d, "k1.k2.k3", sep=".")
    assert ref is d["k1"]["k2"]
    assert key == "k3"
