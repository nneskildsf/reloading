# Reloading
[![CI](https://github.com/nneskildsf/reloading/actions/workflows/CI.yml/badge.svg)](https://github.com/nneskildsf/reloading/actions/workflows/CI.yml)

A Python utility to reload a function or loop body from source on each iteration without losing state.

## Installing Reloading and Supported Versions
This fork of reloading is *not* available on PyPi. Install it from Github:
```console
$ pip install https://github.com/nneskildsf/reloading/archive/refs/heads/master.zip
```

This fork of reloading supports Python 3.6+.

## Supported Features
- Reload functions, `for` loops and `while` loops
- `break` and `continue` in loops
- Multiple reloading functions and loops in one file
- Reloaded functions preserve their original call signature
- Only reload source code when changed for faster performance
- Comprehensive exceptions and logging to let you know of errors
- Exports locals of reloaded loops to parent locals (Python 3.13 and newer)

## Not Supported
- Use of reloading in Jupyter Notebooks and IPython

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

## Additional Options

To iterate forever in a `for` loop you can omit the argument:
```python
from reloading import reloading

for _ in reloading():
    # This code will loop forever and reload from source before each iteration
    pass
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
A warning is emitted when the issue arises: `WARNING:reloading:Variable(s) "i" in reloaded loop were not exported to the scope which called the reloaded loop at line...`.

## Lint, Type Check and Testing

Run:
```console
$ pip install -e ".[development]"
$ flake8
$ pyright
$ python -m unittest
```
