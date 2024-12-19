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


class ReloadingException(Exception):
    pass


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
    if fn_or_seq_or_bool is not None:
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


break_ast = ast.parse('raise Exception("break")').body
continue_ast = ast.parse('raise Exception("continue")').body


class ReplaceBreakContineWithExceptions(ast.NodeTransformer):
    def visit_Break(self, node):
        return break_ast

    def visit_Continue(self, node):
        return continue_ast


def replace_break_continue(ast_module: ast.Module):
    # Replace "break" and "continue" with custom exceptions.
    # Otherwise SyntaxError is raised because these instructions
    # are called outside a loop.
    transformer = ReplaceBreakContineWithExceptions()
    transformed_ast_module = transformer.visit(ast_module)
    ast.fix_missing_locations(transformed_ast_module)
    return compile(transformed_ast_module, filename="", mode="exec")


class WhileLoop:
    """
    Object to hold ast and test-function for a reloading while loop.
    """
    def __init__(self, ast_module: ast.Module, test: ast.Call, id: str):
        self.ast: ast.Module = ast_module
        self.test: ast.Call = test
        self.id: str = id
        self.compiled_body = replace_break_continue(self.ast)
        # If no argument was supplied, then loop forever
        ast_condition = ast.Expression(body=ast.Constant(True))
        if len(test.args) > 0:
            # Create expression to evaluate condition
            ast_condition = ast.Expression(body=test.args[0])
        ast.fix_missing_locations(ast_condition)
        self.condition = compile(ast_condition, filename="", mode="eval")


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
        self.compiled_body = replace_break_continue(self.ast)


def get_loop_object(loop_frame_info: inspect.FrameInfo,
                    reloaded_file_ast: ast.Module,
                    loop_id: Union[None, str]) -> Union[WhileLoop, ForLoop]:
    """
    Traverse AST for the entire reloaded file in a search for the
    loop which is reloaded.
    """
    candidates: List[Union[ast.For, ast.While]] = []
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

    if len(candidates) == 0 and loop_id is None:
        raise Exception("Reloading used outside the context of a loop.")
    elif len(candidates) == 0 and loop_id:
        raise ReloadingException(
            f'Unable to reload loop initially defined at line '
            f'{loop_frame_info.lineno} '
            f'in file "{loop_frame_info.filename}". '
            'The loop might have been removed.'
        )

    # Select the candidate node which is closest to function_frame_info
    def sorting_function(candidate):
        return abs(candidate.lineno - loop_frame_info.lineno)
    candidate = min(candidates, key=sorting_function)
    loop_node_ast = ast.Module(candidate.body, type_ignores=[])
    if isinstance(candidate, ast.For):
        assert isinstance(candidate.target, (ast.Name, ast.Tuple, ast.List))
        return ForLoop(loop_node_ast, candidate.target, get_loop_id(candidate))
    elif isinstance(candidate, ast.While):
        assert isinstance(candidate.test, ast.Call)
        return WhileLoop(loop_node_ast, candidate.test, get_loop_id(candidate))
    raise ReloadingException("No loop node found.")


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
        except (LookupError, ReloadingException):
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
            f"An error occurred. Please edit the file '{filepath}' to fix "
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

    # Initialize variables
    file_stat: int = 0
    vacant_variable_name: str = ""
    assign_compiled = compile('', filename='', mode='exec')
    for_loop = get_loop_code(
        loop_frame_info, loop_id=None
    )

    for i, iteration_variable_values in enumerate(seq):
        # Reload code if possibly modified
        if file_stat != os.stat(filepath).st_mtime_ns:
            if i > 0:
                log.info(f'For loop at line {loop_frame_info.lineno} of file '
                         f'"{filepath}" has been reloaded.')
            for_loop = get_loop_code(
                loop_frame_info, loop_id=for_loop.id
            )
            assert isinstance(for_loop, ForLoop)
            file_stat = os.stat(filepath).st_mtime_ns
            # Make up a name for a variable which is not already present in
            # the global or local namespace.
            vacant_variable_name = unique_name(
                chain(caller_locals.keys(), caller_globals.keys())
            )
            # Reassign variable values from vacant variable in local scope
            assign = ast.Module([
                ast.Assign(targets=[for_loop.iteration_variables],
                           value=ast.Name(vacant_variable_name, ast.Load()))],
                           type_ignores=[])
            ast.fix_missing_locations(assign)
            assign_compiled = compile(assign, filename='', mode='exec')
        # Store iteration variable values in vacant variable in local scope
        caller_locals[vacant_variable_name] = iteration_variable_values
        exec(assign_compiled, caller_globals, caller_locals)
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
        return eval(while_loop.condition, caller_globals, caller_locals)

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
    elif len(stack) > 3:
        variables = ", ".join(
                    f'"{k}"' for k in loop_frame_info.frame.f_locals.keys())
        log.warning(f"Variable(s) {variables} in reloaded loop were not "
                    "exported to the scope which called the reloaded loop "
                    f'initially defined at line {loop_frame_info.lineno} in '
                    f'file "{loop_frame_info.filename}".')
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
    fwod = function_with_decorator
    # Find decorator names
    decorator_names = [get_decorator_name_or_none(decorator)
                       for decorator
                       in fwod.decorator_list]
    fwod.decorator_list = [decorator for decorator, name
                           in zip(fwod.decorator_list, decorator_names)
                           if name != "reloading"]
    function_without_decorator = fwod
    return function_without_decorator


def isolate_function_def(function_frame_info: inspect.FrameInfo,
                         function: Callable,
                         reloaded_file_ast: ast.Module) -> ast.Module:
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
        function_node_ast = ast.Module([function_node], type_ignores=[])
        return function_node_ast
    else:
        raise ReloadingException(
            f'Unable to reload function "{function_name}" '
            f'in file "{function_frame_info.filename}". '
            'The function might have been renamed or the '
            'decorator might have been removed.'
        )


def get_reloaded_function(caller_globals: Dict[str, Any],
                          caller_locals: Dict[str, Any],
                          function_frame_info: inspect.FrameInfo,
                          function: Callable) -> Callable:
    reloaded_file_ast: ast.Module = parse_file_until_successful(
                                        function_frame_info.filename)
    function_ast = isolate_function_def(function_frame_info,
                                        function,
                                        reloaded_file_ast)
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
    rfunction = get_reloaded_function(caller_globals,
                                      caller_locals,
                                      function_frame_info,
                                      function)
    i: int = 0

    def wrapped(*args, **kwargs):
        nonlocal file_stat, function, rfunction, i
        # Reload code if possibly modified
        if file_stat != os.stat(filepath).st_mtime_ns:
            log.info(f'Function "{function.__name__}" at line '
                     f'{function_frame_info.lineno} '
                     f'of file "{filepath}" has been reloaded.')
            rfunction = get_reloaded_function(caller_globals,
                                              caller_locals,
                                              function_frame_info,
                                              function)
            file_stat = os.stat(filepath).st_mtime_ns
        i += 1
        while True:
            try:
                result = rfunction(*args, **kwargs)
                return result
            except Exception:
                handle_exception(filepath)

    wrapped.__signature__ = inspect.signature(function)  # type: ignore
    caller_locals[function.__name__] = wrapped
    return wrapped
