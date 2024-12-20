import unittest
import sys
import os
import subprocess
import time

from reloading import reloading

SRC_FILE_NAME = "temporary_testing_file.py"


def run_and_update_source(init_src, updated_src=None, update_after=0.5):
    """Runs init_src in a subprocess and updates source to updated_src after
    update_after seconds. Returns the standard output of the subprocess and
    whether the subprocess produced an uncaught exception.
    """
    with open(SRC_FILE_NAME, "w") as f:
        f.write(init_src)

    cmd = ["python", SRC_FILE_NAME]
    with subprocess.Popen(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as proc:
        if updated_src is not None:
            time.sleep(update_after)
            with open(SRC_FILE_NAME, "w") as f:
                f.write(updated_src)
                f.flush()

        try:
            stdout, _ = proc.communicate(timeout=2)
            stdout = stdout.decode("utf-8")
            has_error = False
        except Exception:
            stdout = ""
            has_error = True
            proc.terminate()

    if os.path.isfile(SRC_FILE_NAME):
        os.remove(SRC_FILE_NAME)

    return stdout, has_error


class TestReloadingForLoopWithoutChanges(unittest.TestCase):
    def test_no_argument(self):
        i = 0
        for _ in reloading():
            i += 1
            if i > 10:
                break

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 11)

    def test_range_pass(self):
        for _ in reloading(range(10)):
            pass

    def test_range_body(self):
        i = 0
        for _ in reloading(range(10)):
            i += 1

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 10)

    def test_complex_iteration_variables(self):
        i = 0
        j = 0
        for j, (a, b) in reloading(enumerate(zip(range(10), range(10)))):
            i += 1

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 10)
            self.assertEqual(j, 9)

    def test_empty_iterator(self):
        i = 0
        for _ in reloading([]):
            i += 1

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 0)

    def test_use_outside_loop(self):
        with self.assertRaises(Exception):
            reloading(range(10))

    def test_continue(self):
        j = 0
        for i in range(10):
            i += 1
            if i > 5:
                continue
            j = i

        self.assertEqual(i, 10)
        self.assertEqual(j, 5)


class TestReloadingWhileLoopWithoutChanges(unittest.TestCase):
    def test_no_argument(self):
        i = 0
        while reloading():
            i += 1
            if i == 10:
                break

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 10)

    def test_false(self):
        i = 0
        while reloading(False):
            i += 1

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 0)

    def test_true_break(self):
        i = 0
        while reloading(True):
            i += 1
            if i > 9:
                break

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 10)

    def test_condition_changes(self):
        i = 0

        def condition():  # type: ignore
            return True

        while reloading(condition()):
            i += 1
            if i > 9:
                def condition():
                    return False

        if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
            self.assertEqual(i, 10)

    def test_condition(self):
        i = 0
        while i < 10:
            i += 1

        self.assertEqual(i, 10)

    def test_use_outside_loop(self):
        with self.assertRaises(Exception):
            reloading(True)

    def test_continue(self):
        i = 0
        j = 0
        while i < 10:
            i += 1
            if i > 5:
                continue
            j = i

        self.assertEqual(i, 10)
        self.assertEqual(j, 5)


class TestReloadingFunctionWithoutChanges(unittest.TestCase):
    def test_empty_function_definition(self):
        @reloading
        def function():
            pass

    def test_empty_function_run(self):
        @reloading
        def function():
            pass

        function()

    def test_function_return_value(self):
        @reloading
        def function():
            return "result"

        self.assertEqual(function(), "result")

    def test_nested_function(self):
        def outer():
            @reloading
            def inner():
                return "result"
            return inner()

        self.assertEqual(outer(), "result")

    def test_function_signature_is_preserved(self):
        @reloading
        def some_func(a, b, c):
            return "result"

        import inspect
        self.assertEqual(str(inspect.signature(some_func)), "(a, b, c)")

    def test_decorated_function(self):
        def decorator(f):
            def wrap():
                return "<"+str(f())+">"
            return wrap

        @decorator
        @reloading
        def f():
            return 2

        self.assertEqual(f(), "<2>")


class TestReloadingForLoopWithChanges(unittest.TestCase):
    def test_changing_source_loop(self):
        code = """
from reloading import reloading
from time import sleep

for epoch1, epoch2 in reloading(zip(range(10), range(1,11))):
    sleep(0.2)
    print('INITIAL_FILE_CONTENTS')
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED"),
        )
        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_changing_line_number_of_loop(self):
        code = """
from reloading import reloading
from time import sleep

pass

for epoch in reloading(range(10)):
    sleep(0.2)
    print('INITIAL_FILE_CONTENTS')
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("pass", "pass\npass\n").
            replace("INITIAL", "CHANGED"),
        )
        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_keep_local_variables(self):
        code = """
from reloading import reloading
from time import sleep

text = "DON'T CHANGE ME"
for epoch in reloading(range(10)):
    sleep(0.2)
    print('INITIAL_FILE_CONTENTS')
    assert text == "DON'T CHANGE ME"
"""
        stdout, has_error = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED")
        )
        self.assertFalse(has_error)
        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_persist_after_loop(self):
        code = """
from reloading import reloading
from time import sleep

state = 'INIT'
for epoch in reloading(range(1)):
    state = 'CHANGED'

assert state == 'CHANGED'
"""
        _, has_error = run_and_update_source(init_src=code)
        self.assertFalse(has_error)

    def test_comment_after_loop(self):
        code = """
from reloading import reloading
from time import sleep

for epoch in reloading(range(10)):
    sleep(0.2)
    print('INITIAL_FILE_CONTENTS')

# a comment here should not cause an error
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED"),
        )

        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_format_str_in_loop(self):
        code = """
from reloading import reloading
from time import sleep

for epoch in reloading(range(10)):
    sleep(0.2)
    file_contents = 'FILE_CONTENTS'
    print(f'INITIAL_{file_contents}')
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED").rstrip("\n"),
        )

        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_unicode(self):
        code = """
from reloading import reloading
from time import sleep

for epoch, ep in reloading(zip(range(10), range(1,11))):
    sleep(0.2)
    print('INITIAL_FILE_CONTENTS'+'😊')
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED"),
        )

        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_nested_loop(self):
        code = """
from reloading import reloading
from time import sleep

for i in range(1):
    static = 'A'
    for j in reloading(range(10, 20)):
        dynamic = 'B'
        print(static+dynamic)
        sleep(0.2)
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("dynamic = 'B'", "dynamic = 'C'").
            replace("static = 'A'", "static = 'D'"),
        )

        self.assertIn("AB", stdout)
        self.assertIn("AC", stdout)

    def test_function_in_loop(self):
        code = """
from reloading import reloading
from time import sleep

def f():
    return 'f'

for i in reloading(range(10)):
    print(f()+'g')
    sleep(0.2)
"""
        stdout, stderr = run_and_update_source(
            init_src=code,
            updated_src=code.replace("'f'", "'F'").
            replace("'g'", "'G'"),
        )

        self.assertIn("fg", stdout)
        self.assertIn("fG", stdout)
        self.assertNotIn("FG", stdout)


class TestReloadingWhileLoopWithChanges(unittest.TestCase):
    def test_changing_source_loop(self):
        code = """
from reloading import reloading
from time import sleep

i = 0
while reloading(i < 100):
    sleep(0.2)
    print(i)
    i += 1
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("i < 100", "i < 10"),
        )
        max_i = stdout.strip().split("\n")[-1]
        self.assertEqual(max_i, "9")

    def test_function_in_loop(self):
        code = """
from reloading import reloading
from time import sleep

def f():
    return 'f'

i = 0
while reloading(i<10):
    print(f()+'g')
    sleep(0.2)
    i += 1
"""
        stdout, stderr = run_and_update_source(
            init_src=code,
            updated_src=code.replace("'f'", "'F'").
            replace("'g'", "'G'"),
        )

        self.assertIn("fg", stdout)
        self.assertIn("fG", stdout)
        self.assertNotIn("FG", stdout)


class TestReloadingFunctionsWithChanges(unittest.TestCase):
    def test_changing_source_function(self):
        code = """
from reloading import reloading
from time import sleep

@reloading
def reload_this_fn():
    print('INITIAL_FILE_CONTENTS')

for epoch in reloading(range(10)):
    sleep(0.2)
    reload_this_fn()
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("INITIAL", "CHANGED"),
        )
        self.assertIn("INITIAL_FILE_CONTENTS", stdout)
        self.assertIn("CHANGED_FILE_CONTENTS", stdout)

    def test_reloading_function(self):
        code = """
from reloading import reloading
from time import sleep

@reloading
def some_func(a, b):
    sleep(0.2)
    print(a+b)

for _ in range(10):
    some_func(2,1)
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("a+b", "a-b"),
        )

        self.assertIn("3", stdout)
        self.assertIn("1", stdout)

    def test_nested_function(self):
        code = """
from reloading import reloading
from time import sleep

def outer():
    static = 'A'
    @reloading
    def inner(x):
        dynamic = 'B'
        return x + dynamic
    return inner(static)

for i in range(10):
    print(outer())
    sleep(0.2)
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("dynamic = 'B'", "dynamic = 'C'").
            replace("static = 'D'", "static = 'D'"),
        )

        self.assertIn("AB", stdout)
        self.assertIn("AC", stdout)

    def test_multiple_function(self):
        code = """
from reloading import reloading
from time import sleep

@reloading
def f():
    return 'f'

@reloading
def g():
    return 'g'

for i in range(10):
    print(f()+g())
    sleep(0.2)
"""
        stdout, _ = run_and_update_source(
            init_src=code,
            updated_src=code.replace("'f'", "'F'").
            replace("'g'", "'G'"),
        )

        self.assertIn("fg", stdout)
        self.assertIn("FG", stdout)


class TestReloadingMixedWithChanges(unittest.TestCase):
    def test_function_for_loop(self):
        code = """
from reloading import reloading
from time import sleep

@reloading
def f():
    return 'f'

for i in reloading(range(10)):
    print(f()+'g')
    sleep(0.2)
"""
        stdout, stderr = run_and_update_source(
            init_src=code,
            updated_src=code.replace("'f'", "'F'").
            replace("'g'", "'G'"),
        )

        self.assertIn("fg", stdout)
        self.assertIn("FG", stdout)

    def test_function_while_loop(self):
        code = """
from reloading import reloading
from time import sleep

@reloading
def f():
    return 'f'

i = 0
while reloading(i<10):
    print(f()+'g')
    sleep(0.2)
    i += 1
"""
        stdout, stderr = run_and_update_source(
            init_src=code,
            updated_src=code.replace("'f'", "'F'").
            replace("'g'", "'G'"),
        )

        self.assertIn("fg", stdout)
        self.assertIn("FG", stdout)

if __name__ == "__main__":
    unittest.main(verbosity=2)
