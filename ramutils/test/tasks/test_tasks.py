from ramutils.tasks import make_task


def test_make_task():
    def diff(a, b):
        return a - b

    wrapped = make_task(diff, 2, 1)
    assert wrapped == 1

    wrapped = make_task(diff, 3, 1, cache=False)
    assert wrapped == 2
