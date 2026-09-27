from blog.posts import post_url


def test_simple():
    assert post_url(7, "Hello World") == "/posts/7/hello-world"
