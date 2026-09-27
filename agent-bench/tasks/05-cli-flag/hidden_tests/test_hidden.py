from dusum.cli import main


def _tree(p):
    (p / "a.txt").write_bytes(b"x" * 10)
    (p / "b.txt").write_bytes(b"x" * 100)
    (p / "c.py").write_bytes(b"x" * 5)


def test_min_size(tmp_path, capsys):
    _tree(tmp_path)
    main([str(tmp_path), "--min-size", "10"])
    out = capsys.readouterr().out
    assert "TOTAL\t110" in out and ".py" not in out and ".txt\t110" in out


def test_default(tmp_path, capsys):
    _tree(tmp_path)
    main([str(tmp_path)])
    assert "TOTAL\t115" in capsys.readouterr().out
