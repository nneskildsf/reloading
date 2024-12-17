import inspect
import sys
import ast
import traceback
import types
from typing import Optional, Union, Callable, Iterable, Dict, Any, overload, Tuple, List
from itertools import chain
from functools import partial, update_wrapper
from copy import deepcopy

class NoIterPartial(partial):
    """
    Make our own partial in case someone wants to use reloading as an
    iterator. Without any arguments they would get a partial back
    because a call without an iterator argument is assumed to be a decorator.
    Getting a "TypeError: 'functools.partial' object is not iterable"
    is not really descriptive.
    Hence we overwrite the iter to make sure that the error makes sense.
    """
    def __iter__(self):
        raise TypeError(
            "Nothing to iterate over. Please pass an iterable to reloading."
        )

@overload
def reloading(fn_or_seq: Iterable) -> Iterable: ...

@overload
def reloading(fn_or_seq: None) -> Iterable: ...

@overload
def reloading(fn_or_seq: Callable) -> Callable: ...

def reloading(fn_or_seq: Optional[Union[Iterable, Callable]] = None) -> Union[Iterable, Callable]:
    """
    Wraps a loop iterator or decorates a function to reload the source code
    before every loop iteration or function invocation.

    When wrapped around the outermost iterator in a `for` loop, e.g.
    `for i in reloading(range(10))`, causes the loop body to reload from source
    before every iteration while keeping the state.
    When used as a function decorator, the decorated function is reloaded from
    source before each execution.

    Args:
        fn_or_seq (function | iterable): A function or loop iterator which should
            be reloaded from source before each invocation or iteration,
            respectively.
    """
    if fn_or_seq:
        if isinstance(fn_or_seq, Callable):
            return _reloading_function(fn_or_seq)
        elif isinstance(fn_or_seq, Iterable):
            return _reloading_loop(fn_or_seq)
        else:
            raise TypeError(
                f"{reloading.__name__} expected function or iterable, got {type(fn_or_seq)}"
            )
    else:
        return _reloading_loop(iter(int, 1))
    # return this function with the keyword arguments partialed in,
    # so that the return value can be used as a decorator
    decorator = update_wrapper(NoIterPartial(reloading), reloading)
    return decorator


def unique_name(seq: chain) -> str:
    """
    Function to generate string which is unique
    relative to the supplied sequence
    """
    return max(seq, key=len) + "0"


def format_iteration_variables(ast_node: Union[ast.Name, ast.Tuple, ast.List]) -> str:
    """
    Formats an `ast_node` of loop iteration variables as string.
    """
    # ast.Name corresponds to cases where the iterator returns
    # a single element.
    # Example:
    # for i in range(10):
    #   pass
    # ast.Tuple/ast.List corresponds to multiple elements:
    # for i, j in zip(range(10), range(10)):
    #   pass
    if isinstance(ast_node, ast.Name):
        return ast_node.id

    names = []
    for child in ast_node.elts:
        if isinstance(child, ast.Name):
            names.append(child.id)
        elif isinstance(child, ast.Tuple) or isinstance(child, ast.List):
            # Recursion to handle additional tuples such as "a, (b, c)"
            names.append("("+format_iteration_variables(child)+")")

    return ", ".join(names)


def load_file(filepath: str) -> str:
    """
    Read contents of file containing reloading code.
    Handle case of file appearing empty on read.
    """
    while True:
        with open(filepath, "r") as f:
            src = f.read()
        if len(src):
            return src + "\n"


def parse_file_until_successful(filepath: str) -> ast.Module:
    """
    Parse source code of file containing reloading code.
    File may appear incomplete as as it is read so retry until successful.
    """
    source = load_file(filepath)
    while True:
        try:
            tree = ast.parse(source)
            return tree
        except SyntaxError:
            handle_exception(filepath)
            source = load_file(filepath)


def isolate_loop_body_and_get_iteration_variables(reloaded_file_ast: ast.Module,
                                                  lineno: int,
                                                  loop_id: Union[None, str]) -> Tuple[ast.Module, ast.Name, str]:
    """
    Traverse AST for the entire reloaded file in a search for the
    loop which is reloaded.
    """
    candidate_nodes = []
    for node in ast.walk(reloaded_file_ast):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if getattr(node.iter.func, "id") == "reloading" and (
                (loop_id is not None and loop_id == get_loop_id(node))
                or getattr(node, "lineno", None) == lineno):
                candidate_nodes.append(node)
    if len(candidate_nodes) > 1:
        raise LookupError(
            "The reloading loop is ambigious. Use `reloading` only once per line and make sure that the code in that line is unique within the source file."
        )

    if len(candidate_nodes) < 1:
        raise LookupError(
            "Could not locate reloading loop. Please make sure the code in the line that uses `reloading` doesn't change between reloads."
        )

    loop_node = candidate_nodes[0]
    for_loop_ast = deepcopy(reloaded_file_ast) # Use reloaded_file_ast as template.
    for_loop_ast.body = loop_node.body
    return for_loop_ast, loop_node.target, get_loop_id(loop_node)


def get_loop_id(ast_node: ast.For) -> str:
    """
    Generates a unique identifier for an `ast_node`.
    Used to identify the loop in the changed source file.
    """
    return ast.dump(ast_node.target) + "__" + ast.dump(ast_node.iter)


def get_loop_code(loop_frame_info: inspect.FrameInfo, loop_id: Union[None, str]) -> Tuple[ast.Module, str, str]:
    filepath: str = loop_frame_info.filename
    while True:
        reloaded_file_ast: ast.Module = parse_file_until_successful(filepath)
        try:
            for_loop_ast, iteration_variables, found_loop_id = isolate_loop_body_and_get_iteration_variables(
                reloaded_file_ast, lineno=loop_frame_info.lineno, loop_id=loop_id
            )
            return (
                for_loop_ast,
                format_iteration_variables(iteration_variables),
                found_loop_id,
            )
        except LookupError:
            handle_exception(filepath)


def handle_exception(filepath: str):
    """
    Output helpful error message to user regarding exception in reloaded code.
    """
    exception = traceback.format_exc()
    exception = exception.replace('File "<string>"', f'File "{filepath}"')
    sys.stderr.write(exception + "\n")

    if sys.stdin.isatty():
        print(
            f"An error occurred. Please edit the file '{filepath}' to fix the issue and press return to continue or Ctrl+C to exit."
        )
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(1)
    else:
        # get error line number
        line_number = int(exception.split(", line ")[-1].split(",")[0])
        print(line_number)
        raise Exception(
            f"An error occurred. Please fix the issue in the file '{filepath}' and run the script again."
        )


def _reloading_loop(seq: Iterable) -> Iterable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack() i.e. _reloading_loop
    assert stack[0].function == '_reloading_loop'
    # The second element is the caller of the first, i.e. reloading
    assert stack[1].function == 'reloading'
    # The third element is the loop which called reloading.
    loop_frame_info: inspect.FrameInfo = stack[2]
    filepath: str = loop_frame_info.filename

    caller_globals: Dict[str, Any] = loop_frame_info.frame.f_globals
    caller_locals: Dict[str, Any] = loop_frame_info.frame.f_locals

    # Make up a name for a variable which is not already present in the global
    # or local namespace.
    vacant_variable_name: str = unique_name(chain(caller_locals.keys(),
                                                  caller_globals.keys()))
    loop_id = None

    for_loop_ast, iteration_variables, loop_id = get_loop_code(
        loop_frame_info, loop_id=loop_id
    )
    for i, iteration_variable_values in enumerate(seq):
        # Reload code
        for_loop_ast, iteration_variables, loop_id = get_loop_code(
            loop_frame_info, loop_id=loop_id
        )
        # Store iteration variable values in vacant variable in local scope
        caller_locals[vacant_variable_name] = iteration_variable_values
        # Reassign variable values from vacant variable in local scope
        exec(iteration_variables + " = " + vacant_variable_name, caller_globals, caller_locals)
        try:
            # Run loop body
            compiled_body = compile(for_loop_ast, filename="", mode="exec")
            exec(compiled_body, caller_globals, caller_locals)
        except Exception:
            handle_exception(filepath)

    return []


def get_decorator_name_or_none(decorator_node):
    if hasattr(decorator_node, "id"):
        return decorator_node.id
    elif hasattr(decorator_node.func, "id"):
        return decorator_node.func.id
    elif hasattr(decorator_node.func.value, "id"):
        return decorator_node.func.value.id
    else:
        return None


def strip_reloading_decorator(function_with_decorator: ast.FunctionDef):
    """
    Remove the 'reloading' decorator and all decorators before it.
    """
    # Find decorators
    decorator_names = [get_decorator_name_or_none(dec) for dec in function_with_decorator.decorator_list]
    # Find index of "reloading" decorator
    reloading_index = decorator_names.index("reloading")
    function_without_decorator = deepcopy(function_with_decorator) # Use function with decorator as template
    function_without_decorator.decorator_list = function_with_decorator.decorator_list[reloading_index + 1:]
    return function_without_decorator


def isolate_function_def(qualname: str, function: Callable, reloaded_file_ast: ast.Module) -> Union[None, ast.Module]:
    """
    Traverse AST for the entire reloaded file in a search for the
    function (minus the reloading decorator) which is reloaded.
    """
    length = len(qualname.split("."))
    function_name = qualname.split(".")[-1]
    class_name = qualname.split(".")[length - 2] if length > 1 else None

    for node in ast.walk(reloaded_file_ast):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode.name == function_name:
                    if "reloading" in [
                        get_decorator_name_or_none(decorator)
                        for decorator in subnode.decorator_list
                    ]:
                        function_node = strip_reloading_decorator(subnode)
                        function_node_ast = deepcopy(reloaded_file_ast) # Use reloaded_file_ast as template.
                        function_node_ast.body = [function_node]
                        return function_node_ast
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if "reloading" in [
                get_decorator_name_or_none(decorator)
                for decorator in node.decorator_list
            ]:
                function_node = strip_reloading_decorator(node)
                function_node_ast = deepcopy(reloaded_file_ast) # Use reloaded_file_ast as template.
                function_node_ast.body = [ function_node ]
                return function_node_ast
    return None


def get_function_def_code(filepath: str, function: Callable) -> Union[None, ast.Module]:
    reloaded_file_ast: ast.Module = parse_file_until_successful(filepath)
    function_ast = isolate_function_def(function.__qualname__, function, reloaded_file_ast)
    return function_ast


def get_reloaded_function(caller_globals: Dict[str, Any],
                          caller_locals: Dict[str, Any],
                          filepath: str,
                          function: Callable) -> Union[None, Callable]:
    function_ast = get_function_def_code(filepath, function)
    if function_ast is None:
        return None
    # need to copy locals, otherwise the exec will overwrite the decorated with the undecorated new version
    # this became a need after removing the reloading decorator from the newly defined version
    caller_locals_copy = caller_locals.copy()
    compiled_body = compile(function_ast, filename="", mode="exec")
    exec(compiled_body, caller_globals, caller_locals_copy)
    function = caller_locals_copy[function.__name__]
    return function


def _reloading_function(function: Callable) -> Callable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack() i.e. _reloading_function
    assert stack[0].function == '_reloading_function'
    # The second element is the caller of the first, i.e. reloading
    assert stack[1].function == 'reloading'
    # The third element is the loop which called reloading.
    function_frame_info: inspect.FrameInfo = stack[2]
    filepath: str = function_frame_info.filename

    caller_globals = function_frame_info.frame.f_globals
    caller_locals = function_frame_info.frame.f_locals
    # crutch to use dict as python2 doesn't support nonlocal
    state = {
        "function": function,
        "reloads": 0,
    }

    def wrapped(*args, **kwargs):
        state["function"] = (
            get_reloaded_function(caller_globals,
                                  caller_locals,
                                  filepath,
                                  function)
            or state["function"]
        )
        state["reloads"] += 1
        while True:
            try:
                result = state["function"](*args, **kwargs)
                return result
            except Exception:
                handle_exception(filepath)
                state["func"] = (
                    get_reloaded_function(caller_globals,
                                          caller_locals,
                                          filepath,
                                          function)
                    or state["function"]
                )

    wrapped.__signature__ = inspect.signature(function) # type: ignore
    caller_locals[function.__name__] = wrapped
    return wrapped
