import unittest

from adcc.misc import cached_member_function


class SomeClass:
    def uncached(self, x=0, y=1):
        return (x, y)

    @cached_member_function
    def cached(self, x=0, y=1):
        return (x, y)


class TestCachedMemberFunction(unittest.TestCase):
    def test_cache(self):
        instance = SomeClass()
        assert instance.uncached() is not instance.uncached()
        res = instance.cached(0, 1)
        assert res is instance.cached(0, 1)
        assert res is instance.cached(0)
        assert res is instance.cached()
        assert res is instance.cached(x=0)
        assert res is instance.cached(y=1)
        assert res is instance.cached(y=1, x=0)

    def test_wrappable(self):
        def kwargs_only(self, *, kwarg):
            pass

        def kwargs(self, **kwargs):
            pass

        def positional_only(self, /, arg):
            pass

        def args(self, *args):
            pass

        with self.assertRaises(ValueError):
            cached_member_function(kwargs_only)
        with self.assertRaises(ValueError):
            cached_member_function(kwargs)
        cached_member_function(positional_only)
        cached_member_function(args)
