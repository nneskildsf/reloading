# Reloading
[![CI](https://github.com/nneskildsf/reloading/actions/workflows/CI.yml/badge.svg)](https://github.com/nneskildsf/reloading/actions/workflows/CI.yml)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python utility to reload a function or loop body from source on each iteration without losing state.

## Installing Reloading and Supported Versions
This fork of reloading is *not* available on PyPi. Install it from Github:
```console
$ pip install https://github.com/nneskildsf/reloading/archive/refs/heads/master.zip
```

This fork of reloading supports Python 3.6+.

## Supported Features
- Reload functions, `for` loops and `while` loops
- Works in Jupyter Notebook
- `break` and `continue` in loops
- Multiple reloading functions and loops in one file
- Reloaded functions preserve their original call signature
- Only reload source code when changed for faster performance
- Comprehensive exceptions and logging to let you know of errors
- Exports locals of reloaded loops to parent locals (Python 3.13 and newer)

## Usage

### For Loop
To reload the body of a `for` loop from source before each iteration, wrap the iterator with `reloading`:
```python
from reloading import reloading

for i in reloading(range(10)):
    # This code will be reloaded before each iteration
    print(i)
```

### While Loop
To reload the body and condition of a `while` loop from source before each iteration, wrap the condition with `reloading`:
```python
from reloading import reloading

i = 0
while reloading(i<10):
    # This code and the condition (i<10) will be reloaded before each iteration
    print(i)
    i += 1
```

### Function
To reload a function from source before each execution, decorate the function
definition with `@reloading`:
```python
from reloading import reloading

@reloading
def function():
    # This code will be reloaded before each function call
    pass
```

It is also possible to mark a function for reload after defining it:
```python
from reloading import reloading

def function():
    # This function will be reloaded before each function call
    pass

function = reloading(function)
```

## Additional Options

### Iterate Forever in For Loop
To iterate forever in a `for` loop you can omit the argument:
```python
from reloading import reloading

for _ in reloading():
    # This code will loop forever and reload from source before each iteration
    pass
```

### Code Changes Logged
On Python 3.9 and newer, a diff is logged when the source code is updated.
Consider the following code as an example.
```python
from reloading import reloading
from time import sleep
import logging

log = logging.getLogger("reloading")
log.setLevel(logging.DEBUG)

for i in reloading(range(100)):
    print(i)
    sleep(1.0)
```
After some time the code is edited. `i = 2*i` is added before `print(i)`,
resulting in the following log output:
```console
INFO:reloading:For loop at line 10 of file "../example.py" has been reloaded.
DEBUG:reloading:Code changes:
+i = i * 2
 print(i)
 sleep(1.0)
```

## Known Issus

On Python version [less than 3.13](https://docs.python.org/3/reference/datamodel.html#frame.f_locals) it is not possible to properly export the local variables from a loop to parent locals. The following example demonstrates this:
```python
from reloading import reloading

def function():
    i = 0
    while reloading(i < 10):
        i += 1
    print(i)

function() # Prints 0. Not 10 as expected. Fixed in Python 3.13.
```
A warning is emitted when the issue arises:
```console
WARNING:reloading:Variable(s) "i" in reloaded loop were not exported to the scope which called the reloaded loop at line...
```

## Lint, Type Check and Testing

Run:
```console
$ pip install -e ".[development]"
$ ruff check .
$ flake8
$ pyright
$ mypy .
$ python -m unittest
```
