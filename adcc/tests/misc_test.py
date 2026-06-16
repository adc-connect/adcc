import unittest

from adcc.misc import cached_member_function
from adcc.timings import Timer


class SomeClass:
    def __init__(self):
        self.my_timer = Timer()

    def uncached(self, x=0, y=1):
        return (x, y)

    @cached_member_function()
    def cached_untimed(self, x=0, y=1):
        return (x, y)

    @cached_member_function(timer="my_timer")
    def cached_timed(self, x=0, y=1):
        return (x, y)

    @cached_member_function(timer="my_timer", separate_timings_by_args=False)
    def cached_timed_combined(self, x=0, y=1):
        return (x, y)

    @cached_member_function(timer="my_timer")
    def cached_evaluatable_timed(self):
        return OtherClass()

    @cached_member_function(timer="other_timer")
    def cached_evaluatable(self):
        return OtherClass()


class OtherClass:
    def evaluate(self):
        return (0, 1)


class TestCachedMemberFunction(unittest.TestCase):
    def test_cache(self):
        instance = SomeClass()
        assert instance.uncached() is not instance.uncached()
        res = instance.cached_untimed(0, 1)
        assert res is instance.cached_untimed(0, 1)
        assert res is instance.cached_untimed(0)
        assert res is instance.cached_untimed()
        assert res is instance.cached_untimed(x=0)
        assert res is instance.cached_untimed(y=1)
        assert res is instance.cached_untimed(y=1, x=0)

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
            cached_member_function()(kwargs_only)
        with self.assertRaises(ValueError):
            cached_member_function()(kwargs)
        cached_member_function()(positional_only)
        cached_member_function()(args)

    def test_timer(self):
        instance = SomeClass()
        res = instance.cached_timed()
        assert res is instance.cached_timed()
        assert len(instance.my_timer.intervals("cached_timed/0_1")) == 1
        # store the timings independent of the args
        instance.cached_timed_combined(0, 0)
        instance.cached_timed_combined(0, 1)
        assert len(instance.my_timer.intervals("cached_timed_combined")) == 2

    def test_evaluate(self):
        instance = SomeClass()
        res = instance.cached_evaluatable()
        assert res == (0, 1)
        assert res is instance.cached_evaluatable()
        with self.assertRaises(ValueError):  # there should be no timings available
            instance.my_timer.intervals("cached_evaluatable/")
        res = instance.cached_evaluatable_timed()
        assert res == (0, 1)
        assert res is instance.cached_evaluatable_timed()
        assert len(instance.my_timer.intervals("cached_evaluatable_timed/")) == 1
