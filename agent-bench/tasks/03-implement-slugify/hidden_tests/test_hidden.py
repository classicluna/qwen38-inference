from blog.text import slugify
from blog.tags import tag_url


def test_rules():
    assert slugify("  Crème Brûlée: 10 Tips!! ") == "creme-brulee-10-tips"
    assert slugify("a---b___c") == "a-b-c"
    assert slugify("!!!") == "untitled"
    assert slugify("") == "untitled"
    s = slugify("word " * 40)
    assert len(s) <= 50 and not s.endswith("-")


def test_tag():
    assert tag_url("Machine Learning & AI Systems Design") == "/tags/machine-learning-ai"
