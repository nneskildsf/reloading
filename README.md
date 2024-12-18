This is a fork of https://github.com/julvo/reloading with the following improvements:
* Typing (pyright compliance)
* Linting (flak8 compliance)
* Expanded test suite
* Supports multiple invocations of `reloading` in a single file
* Supports `while` loop
* Preserves function signature for functions decorated with `@reloading`
* CI with Github Actions workflow
* Reload code only if source code file has been changed
* With Python 3.13: Exports locals of reloaded loop to parent locals

# Reloading
A Python utility to reload a function or loop body from source on each iteration without losing state.

## Install
```
pip install git+https://github.com/nneskildsf/reloading.git
```

## Usage

To reload the body of a `for` loop from source before each iteration, wrap the iterator with `reloading`:
```python
from reloading import reloading

for i in reloading(range(10)):
    # This code will be reloaded before each iteration
    print(i)
```

To reload the body and condition of a `while` loop from source before each iteration, wrap the condition with `reloading`:
```python
from reloading import reloading

i = 0
while reloading(i<10):
    # This code and the condition (i<10) will be reloaded before each iteration
    print(i)
    i += 1
```

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

## Lint, Type Check and Testing

Run:
```
$ flake8
$ pyright
$ python -m unittest
```
