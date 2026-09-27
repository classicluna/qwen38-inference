from dusum.cli import main


def test_basic(tmp_path, capsys):
    (tmp_path / "a.txt").write_bytes(b"x" * 10)
    main([str(tmp_path)])
    assert "TOTAL\t10" in capsys.readouterr().out
