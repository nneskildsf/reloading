This is a fork of https://github.com/julvo/reloading with the following improvements:
* Typing (pyright compliance)
* Linting (flak8e compliance)
* Expanded test suite
* Supports multiple invocations of `reloading` in a single file
* Supports `while` loop
* Supports `continue` and `break` in loops
* Improved performance (Reloaded `while` and `for` loops are only about 40 times slower than regular)
* Preserves function signature for functions decorated with `@reloading`
* CI with Github Actions workflow
* Reload code only if source code file has been changed
* With Python 3.13: Exports locals of reloaded loop to parent locals
* Improved error handling

# Reloading
A Python utility to reload a function or loop body from source on each iteration without losing state.

## Install
```
pip install git+https://github.com/nneskildsf/reloading.git
```
or if you do not have Git
```
pip install https://github.com/nneskildsf/reloading/archive/refs/heads/master.zip
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

## Known Issus

On Python version less than 3.13 it is not possible to properly export the local variables from a loop to parent locals. The following example demonstrates this:
```python
from reloading import reloading

def function():
    i = 0
    while reloading(i < 10):
        i += 1
    print(i)

function() # Prints 0. Not 10 as expected.
```
A warning is emitted when the issue arises: `WARNING:reloading:Variable(s) "i" in reloaded loop were not exported to the scope which called the reloaded loop at line...`.

## Lint, Type Check and Testing

Run:
```
$ pip install -e ".[development]"
$ flake8
$ pyright
$ python -m unittest
```
