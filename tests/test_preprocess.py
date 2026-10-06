import pandas as pd

from data import preprocess
from data.preprocess import HELD_OUT_SOURCES, load_and_clean, split_seen_pool, write_summary


def raw_frame():
    rows = [
        ("  Human essay one.\r\nSecond line.  ", 0, "p1", "persuade_corpus"),
        ("Human essay two.", 0, "p2", "train_essays"),
        ("AI essay from gpt.", 1, "p1", "chat_gpt_moth"),
        ("AI essay with unknown generator.", 1, "p2", "train_essays"),
    ]
    return pd.DataFrame(rows, columns=["text", "label", "prompt_name", "source"]).assign(RDizzl3_seven=False)


def test_held_out_sources_are_the_two_documented_ones():
    assert HELD_OUT_SOURCES == ["falcon_180b_v1", "llama_70b_v1"]


def test_load_and_clean_drops_ai_rows_without_a_generator(tmp_path):
    path = tmp_path / "raw.csv"
    raw_frame().to_csv(path, index=False)
    df, n_dropped = load_and_clean(path)
    assert n_dropped == 1
    assert len(df) == 3
    assert "AI essay with unknown generator." not in df["text"].tolist()


def test_load_and_clean_labels_sources_and_normalizes_text(tmp_path):
    path = tmp_path / "raw.csv"
    raw_frame().to_csv(path, index=False)
    df, _ = load_and_clean(path)
    assert df.columns.tolist() == ["text", "label", "generation_source", "prompt_name", "char_count", "word_count"]
    assert df["generation_source"].tolist() == ["human", "human", "chat_gpt_moth"]
    assert df["text"].iloc[0] == "Human essay one.\nSecond line."
    assert df["char_count"].iloc[0] == len("Human essay one.\nSecond line.")
    assert df["word_count"].tolist() == [5, 3, 4]


def make_seen_pool(n_per_source=40):
    parts = []
    for src, label in [("human", 0), ("gpt", 1), ("mistral", 1)]:
        parts.append(
            pd.DataFrame(
                {"text": [f"{src} {i}" for i in range(n_per_source)], "label": label, "generation_source": src}
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_split_is_70_15_15_and_disjoint():
    pool = make_seen_pool()
    train, val, test = split_seen_pool(pool)
    assert (len(train), len(val), len(test)) == (84, 18, 18)
    assert not set(train["text"]) & set(val["text"])
    assert not set(train["text"]) & set(test["text"])
    assert not set(val["text"]) & set(test["text"])
    assert len(set(train["text"]) | set(val["text"]) | set(test["text"])) == len(pool)


def test_split_is_stratified_by_generation_source():
    train, val, test = split_seen_pool(make_seen_pool())
    for split in (train, val, test):
        counts = split["generation_source"].value_counts()
        assert set(counts.index) == {"human", "gpt", "mistral"}
        assert counts.max() - counts.min() <= 1


def test_split_is_deterministic():
    pool = make_seen_pool()
    a = split_seen_pool(pool)
    b = split_seen_pool(pool)
    for x, y in zip(a, b, strict=True):
        assert x["text"].tolist() == y["text"].tolist()


def test_summary_lists_held_out_counts_and_marks_them():
    df = pd.DataFrame(
        {
            "text": list("abcd"),
            "label": [0, 1, 1, 1],
            "generation_source": ["human", "falcon_180b_v1", "llama_70b_v1", "chat_gpt_moth"],
        }
    )
    held = df[df["generation_source"].isin(HELD_OUT_SOURCES)]
    seen = df.drop(held.index)
    text = write_summary(df, seen, seen.iloc[0:0], seen.iloc[0:0], held, n_dropped=3)
    assert "Dropped 3 row(s)" in text
    assert "- `falcon_180b_v1`: 1 rows" in text
    assert "| falcon_180b_v1 | 1 | 1 | yes |" in text
    assert "| human | 0 | 1 |  |" in text


def test_main_reports_missing_raw_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(preprocess, "RAW_PATH", tmp_path / "missing.csv")
    assert preprocess.main() == 1
    assert "Run data/download.py first" in capsys.readouterr().out


def test_main_keeps_held_out_sources_out_of_every_split(monkeypatch, tmp_path):
    rows = []
    for src, label in [("human", 0), ("chat_gpt_moth", 1), ("falcon_180b_v1", 1), ("llama_70b_v1", 1)]:
        for i in range(30):
            rows.append((f"{src} essay {i}", label, "p", "persuade_corpus" if label == 0 else src))
    raw = pd.DataFrame(rows, columns=["text", "label", "prompt_name", "source"])
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(preprocess, "RAW_PATH", raw_path)
    monkeypatch.setattr(preprocess, "OUT_DIR", out_dir)
    assert preprocess.main() == 0
    held = pd.read_csv(out_dir / "held_out.csv")
    assert set(held["generation_source"]) == set(HELD_OUT_SOURCES)
    for name in ("train", "val", "test"):
        split = pd.read_csv(out_dir / f"{name}.csv")
        assert not set(split["generation_source"]) & set(HELD_OUT_SOURCES)
    assert (out_dir / "dataset_summary.md").exists()
