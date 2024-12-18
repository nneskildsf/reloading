import inspect
import sys
import os
import ast
import traceback
from typing import (Optional,
                    Union,
                    Callable,
                    Iterable,
                    Dict,
                    Any,
                    List,
                    overload)
from itertools import chain
from functools import partial, update_wrapper
from copy import deepcopy
import logging

log = logging.getLogger("reloading")
logging.basicConfig(level=logging.INFO)


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
def reloading(fn_or_seq_or_bool: Iterable) -> Iterable: ...


@overload
def reloading(fn_or_seq_or_bool: None) -> Iterable: ...


@overload
def reloading(fn_or_seq_or_bool: bool) -> Iterable: ...


@overload
def reloading() -> Iterable: ...


@overload
def reloading(fn_or_seq_or_bool: Callable) -> Callable: ...


def reloading(fn_or_seq_or_bool: Optional[
              Union[Iterable,
                    Callable,
                    bool]] = None) -> Union[Iterable,
                                            Callable]:
    """
    Wraps a loop iterator or decorates a function to reload the source code
    before every loop iteration or function invocation.

    When wrapped around the outermost iterator in a `for` loop, e.g.
    `for i in reloading(range(10))`, causes the loop body to reload from source
    before every iteration while keeping the state.
    When wrapped around the condition of a `while` loop, e.g.
    `while reloading(i<10)`, causes the loop body and condition to reload from
    source before every iteration while keeping the state.
    When used as a function decorator, the decorated function is reloaded from
    source before each execution.

    Args:
        fn_or_seq_or_bool (function | iterable | bool):
            A function, iterator or condition which should be reloaded from
            source before each invocation or iteration, respectively.
    """
    if fn_or_seq_or_bool:
        if isinstance(fn_or_seq_or_bool, Callable):
            return _reloading_function(fn_or_seq_or_bool)
        elif (isinstance(fn_or_seq_or_bool, Iterable) or
              isinstance(fn_or_seq_or_bool, bool)):
            return _reloading_loop(fn_or_seq_or_bool)
        else:
            raise TypeError(
                f'{reloading.__name__} expected function or iterable'
                f', got "{type(fn_or_seq_or_bool)}"'
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


def format_iteration_variables(ast_node: Union[ast.Name,
                                               ast.Tuple,
                                               ast.List, None]) -> str:
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
    if ast_node is None:
        return ""

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


class WhileLoop:
    """
    Object to hold ast and test-function for a reloading while loop.
    """
    def __init__(self, ast_module: ast.Module, test: ast.Call, id: str):
        self.ast: ast.Module = ast_module
        self.test: ast.Call = test
        self.id: str = id
        # Replace "break" and "continue" with custom exceptions.
        # Otherwise SyntaxError is raised because these instructions
        # are called outside a loop.
        code = ast.unparse(ast_module)
        code = code.replace("break", "raise Exception('break')")
        code = code.replace("continue", "raise Exception('continue')")
        # Compile loop body
        self.compiled_body = compile(code, filename="", mode="exec")


class ForLoop:
    """
    Object to hold ast and iteration variables for a reloading for loop.
    """
    def __init__(self,
                 ast_module: ast.Module,
                 iteration_variables: Union[ast.Name,
                                            ast.Tuple,
                                            ast.List],
                 id: str):
        self.ast: ast.Module = ast_module
        self.iteration_variables: Union[ast.Name,
                                        ast.Tuple,
                                        ast.List] = iteration_variables
        self.iteration_variables_str: str = format_iteration_variables(
                                            iteration_variables)
        self.id: str = id
        # Replace "break" and "continue" with custom exceptions.
        # Otherwise SyntaxError is raised because these instructions
        # are called outside a loop.
        code = ast.unparse(ast_module)
        code = code.replace("break", "raise Exception('break')")
        code = code.replace("continue", "raise Exception('continue')")
        # Compile loop body
        self.compiled_body = compile(code, filename="", mode="exec")


def get_loop_object(loop_frame_info: inspect.FrameInfo,
                    reloaded_file_ast: ast.Module,
                    loop_id: Union[None, str]) -> Union[WhileLoop, ForLoop]:
    """
    Traverse AST for the entire reloaded file in a search for the
    loop which is reloaded.
    """
    candidates = []
    for node in ast.walk(reloaded_file_ast):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if getattr(node.iter.func, "id") == "reloading" and (
               (loop_id is not None and loop_id == get_loop_id(node))
               or getattr(node, "lineno", None) == loop_frame_info.lineno):
                candidates.append(node)
        if isinstance(node, ast.While) and isinstance(node.test, ast.Call):
            if getattr(node.test.func, "id") == "reloading" and (
               (loop_id is not None and loop_id == get_loop_id(node))
               or getattr(node, "lineno", None) == loop_frame_info.lineno):
                candidates.append(node)

    # Select the candidate node which is closest to function_frame_info
    def sorting_function(candidate):
        return abs(candidate.lineno - loop_frame_info.lineno)
    candidate = min(candidates, key=sorting_function)
    # Use reloaded_file_ast as template.
    loop_node_ast = deepcopy(reloaded_file_ast)
    loop_node_ast.body = candidate.body
    if isinstance(candidate, ast.For):
        assert isinstance(candidate.target, (ast.Name, ast.Tuple, ast.List))
        return ForLoop(loop_node_ast, candidate.target, get_loop_id(candidate))
    elif isinstance(candidate, ast.While):
        assert isinstance(candidate.test, ast.Call)
        return WhileLoop(loop_node_ast, candidate.test, get_loop_id(candidate))
    raise Exception("Unable to find reloading loop node.")


def get_loop_id(ast_node: Union[ast.For, ast.While]) -> str:
    """
    Generates a unique identifier for an `ast_node`.
    Used to identify the loop in the changed source file.
    """
    if isinstance(ast_node, ast.For):
        return ast.dump(ast_node.target) + "__" + ast.dump(ast_node.iter)
    elif isinstance(ast_node, ast.While):
        return ast.dump(ast_node.test)


def get_loop_code(loop_frame_info: inspect.FrameInfo,
                  loop_id: Union[None, str]) -> Union[WhileLoop, ForLoop]:
    filepath: str = loop_frame_info.filename
    while True:
        reloaded_file_ast: ast.Module = parse_file_until_successful(filepath)
        try:
            return get_loop_object(
                loop_frame_info, reloaded_file_ast, loop_id=loop_id
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
            f"An error occurred. Please edit the file '{filepath}' to fix"
            "the issue and press return to continue or Ctrl+C to exit."
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
            'An error occurred. Please fix the issue in the file'
            f'"{filepath}" and run the script again.'
        )


def execute_for_loop(seq: Iterable, loop_frame_info: inspect.FrameInfo):
    filepath = loop_frame_info.filename
    caller_globals: Dict[str, Any] = loop_frame_info.frame.f_globals
    caller_locals: Dict[str, Any] = loop_frame_info.frame.f_locals

    file_stat: int = os.stat(filepath).st_mtime_ns
    for_loop = get_loop_code(
        loop_frame_info, loop_id=None
    )

    for i, iteration_variable_values in enumerate(seq):
        # Reload code if possibly modified
        if file_stat != os.stat(filepath).st_mtime_ns:
            log.info(f'For loop at line {loop_frame_info.lineno} of file '
                     f'"{filepath}" has been reloaded.')
            for_loop = get_loop_code(
                loop_frame_info, loop_id=for_loop.id
            )
            file_stat = os.stat(filepath).st_mtime_ns
        assert isinstance(for_loop, ForLoop)
        # Make up a name for a variable which is not already present in
        # the global or local namespace.
        vacant_variable_name: str = unique_name(chain(caller_locals.keys(),
                                                      caller_globals.keys()))
        # Store iteration variable values in vacant variable in local scope
        caller_locals[vacant_variable_name] = iteration_variable_values
        # Reassign variable values from vacant variable in local scope
        exec(for_loop.iteration_variables_str + " = " + vacant_variable_name,
             caller_globals, caller_locals)
        # Clean up namespace
        del caller_locals[vacant_variable_name]
        try:
            exec(for_loop.compiled_body, caller_globals, caller_locals)
        except Exception as exception:
            # A "break" inside the loop body will cause a SyntaxError
            # because the code is executed outside the scope of a loop.
            # We catch the exception and break *this* loop.
            if exception.args == ("break",):
                break
            if exception.args == ("continue",):
                continue
            else:
                handle_exception(filepath)


def execute_while_loop(loop_frame_info: inspect.FrameInfo):
    filepath = loop_frame_info.filename
    caller_globals: Dict[str, Any] = loop_frame_info.frame.f_globals
    caller_locals: Dict[str, Any] = loop_frame_info.frame.f_locals

    file_stat: int = os.stat(filepath).st_mtime_ns
    while_loop = get_loop_code(
        loop_frame_info, loop_id=None
    )

    def condition(while_loop):
        test = ast.unparse(while_loop.test).replace("reloading", "")
        # Make up a name for a variable which is not already present in
        # the global or local namespace.
        vacant_variable_name: str = unique_name(chain(caller_locals.keys(),
                                                      caller_globals.keys()))
        exec(vacant_variable_name+" = "+test, caller_globals, caller_locals)
        result = deepcopy(caller_locals[vacant_variable_name])
        del caller_locals[vacant_variable_name]
        return result

    i = 0
    while condition(while_loop):
        i += 1
        # Reload code if possibly modified
        if file_stat != os.stat(filepath).st_mtime_ns:
            log.info(f'While loop at line {loop_frame_info.lineno} of file '
                     f'"{filepath}" has been reloaded.')
            while_loop = get_loop_code(
                loop_frame_info, loop_id=while_loop.id
            )
            file_stat = os.stat(filepath).st_mtime_ns
        try:
            exec(while_loop.compiled_body, caller_globals, caller_locals)
        except Exception as exception:
            # A "break" inside the loop body will cause a SyntaxError
            # because the code is executed outside the scope of a loop.
            # We catch the exception and break *this* loop.
            if exception.args == ("break",):
                break
            if exception.args == ("continue",):
                continue
            else:
                handle_exception(filepath)


def _reloading_loop(seq: Union[Iterable, bool]) -> Iterable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack()
    # i.e. _reloading_loop
    assert stack[0].function == "_reloading_loop"
    # The second element is the caller of the first, i.e. reloading
    assert stack[1].function == "reloading"
    # The third element is the loop which called reloading.
    loop_frame_info: inspect.FrameInfo = stack[2]
    loop_object = get_loop_code(
        loop_frame_info, loop_id=None
    )

    if isinstance(loop_object, ForLoop):
        assert isinstance(seq, Iterable)
        execute_for_loop(seq, loop_frame_info)
    elif isinstance(loop_object, WhileLoop):
        execute_while_loop(loop_frame_info)
    # If there is a third element, then it is the scope which called the loop.
    # It is only possible to modify variables in this scope since Python 3.13.
    if (len(stack) > 3 and
       sys.version_info.major >= 3 and
       sys.version_info.minor >= 13):
        # Copy locals from loop to caller of loop.
        # This ensures that the following results in '9':
        # for i in reloading(range(10)):
        #   pass
        # print(i)
        loop_caller_frame: inspect.FrameInfo = stack[3]
        loop_caller_frame.frame.f_locals.update(loop_frame_info.frame.f_locals)
    else:
        variables = ", ".join(
                    f'"{k}"' for k in loop_frame_info.frame.f_locals.keys())
        log.warning(f"Variable(s) {variables} in reloaded loop were not "
                    "exported to the scope which called the reloaded loop at "
                    f'line {loop_frame_info.lineno} in file '
                    f'"{loop_frame_info.filename}".')
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
    # Create shorthand for readability
    fwd = function_with_decorator
    # Find decorators
    decorator_names = [get_decorator_name_or_none(decorator)
                       for decorator
                       in fwd.decorator_list]
    # Find index of "reloading" decorator
    reloading_index = decorator_names.index("reloading")
    # Use function with decorator as template
    fwod = deepcopy(fwd)
    fwod.decorator_list = fwd.decorator_list[reloading_index + 1:]
    function_without_decorator = fwod
    return function_without_decorator


def isolate_function_def(function_frame_info: inspect.FrameInfo,
                         function: Callable,
                         reloaded_file_ast: ast.Module) -> Union[None,
                                                                 ast.Module]:
    """
    Traverse AST for the entire reloaded file in a search for the
    function (minus the reloading decorator) which is reloaded.
    """
    qualname = function.__qualname__
    length = len(qualname.split("."))
    function_name = qualname.split(".")[-1]
    class_name = qualname.split(".")[length - 2] if length > 1 else None

    candidates = []
    for node in ast.walk(reloaded_file_ast):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for subnode in node.body:
                if (isinstance(subnode, ast.FunctionDef) and
                   subnode.name == function_name):
                    if "reloading" in [
                        get_decorator_name_or_none(decorator)
                        for decorator in subnode.decorator_list
                    ]:
                        candidates.append(subnode)
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if "reloading" in [
                get_decorator_name_or_none(decorator)
                for decorator in node.decorator_list
            ]:
                candidates.append(node)
    # Select the candidate node which is closest to function_frame_info
    if len(candidates):
        def sorting_function(candidate):
            return abs(candidate.lineno - function_frame_info.lineno)
        candidate = min(candidates, key=sorting_function)
        function_node = strip_reloading_decorator(candidate)
        function_node_ast = deepcopy(reloaded_file_ast)
        function_node_ast.body = [function_node]
        return function_node_ast
    return None


def get_function_def_code(function_frame_info: inspect.FrameInfo,
                          function: Callable) -> Union[None, ast.Module]:
    reloaded_file_ast: ast.Module = parse_file_until_successful(
                                        function_frame_info.filename)
    return isolate_function_def(function_frame_info,
                                function,
                                reloaded_file_ast)


def get_reloaded_function(caller_globals: Dict[str, Any],
                          caller_locals: Dict[str, Any],
                          function_frame_info: inspect.FrameInfo,
                          function: Callable) -> Union[None, Callable]:
    function_ast = get_function_def_code(function_frame_info, function)
    if function_ast is None:
        return None
    # Copy locals to avoid exec overwriting the decorated function with the new
    # undecorated function.
    caller_locals_copy = caller_locals.copy()
    compiled_body = compile(function_ast, filename="", mode="exec")
    exec(compiled_body, caller_globals, caller_locals_copy)
    function = caller_locals_copy[function.__name__]
    return function


def _reloading_function(function: Callable) -> Callable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack()
    # That is, this very function.
    assert stack[0].function == "_reloading_function"
    # The second element is the caller of the first, i.e. reloading
    assert stack[1].function == "reloading"
    # The third element is the loop which called reloading.
    function_frame_info: inspect.FrameInfo = stack[2]
    filepath: str = function_frame_info.filename

    caller_globals = function_frame_info.frame.f_globals
    caller_locals = function_frame_info.frame.f_locals

    file_stat: int = os.stat(filepath).st_mtime_ns

    # crutch to use dict as python2 doesn't support nonlocal
    state = {
        "function": get_reloaded_function(caller_globals,
                                          caller_locals,
                                          function_frame_info,
                                          function),
        "reloads": 0,
    }

    def wrapped(*args, **kwargs):
        nonlocal file_stat
        # Reload code if possibly modified
        if file_stat != os.stat(filepath).st_mtime_ns:
            log.info(f'Function at line {function_frame_info.lineno} '
                     f'of file "{filepath}" has been reloaded.')
            state["function"] = (
                get_reloaded_function(caller_globals,
                                      caller_locals,
                                      function_frame_info,
                                      function)
                or state["function"]
            )
            file_stat = os.stat(filepath).st_mtime_ns
        state["reloads"] += 1
        while True:
            try:
                result = state["function"](*args, **kwargs)
                return result
            except Exception:
                handle_exception(filepath)

    wrapped.__signature__ = inspect.signature(function)  # type: ignore
    caller_locals[function.__name__] = wrapped
    return wrapped
