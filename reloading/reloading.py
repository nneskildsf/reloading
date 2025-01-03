import ast
import difflib
import functools
import inspect
import itertools
import logging
import os
import sys
import traceback
import types
from typing import (Any,
                    Callable,
                    Dict,
                    Iterable,
                    List,
                    Optional,
                    overload,
                    Union)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_diff_text(ast_before: ast.Module, ast_after: ast.Module):
    """
    Calculate difference between two versions of reloaded code.
    """
    # Unparse was introduced in Python 3.9.
    if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
        # mypy complains that unparse is not available in some
        # versions of Python.
        unparse = getattr(ast, "unparse")
        code_before = unparse(ast_before)
        code_after = unparse(ast_after)
        diff = difflib.unified_diff(code_before.splitlines(),
                                    code_after.splitlines(),
                                    lineterm="")
        # Omit first three lines because they contain superfluous information.
        changes = list(diff)[3:]
        if len(changes):
            return "\n".join(["Code changes:"]+changes)
        else:
            return "No code changes."
    else:
        return "Cannot compute code changes. Requires Python > 3.9."


class ReloadingException(Exception):
    pass


@overload
def reloading(fn_or_seq_or_bool: Iterable) -> Iterable: ...


@overload
def reloading(fn_or_seq_or_bool: None) -> Iterable: ...


@overload
def reloading(fn_or_seq_or_bool: bool) -> Iterable: ...


@overload
def reloading(*, interactive_exception: bool) -> Callable: ...


@overload
def reloading() -> Callable: ...


@overload
def reloading(fn_or_seq_or_bool: Callable) -> Callable: ...


def reloading(fn_or_seq_or_bool: Optional[
              Union[Iterable,
                    Callable,
                    bool]] = None,
              interactive_exception: bool = True) -> Union[Iterable,
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
        fn_or_seq_or_bool:
            A function, iterator or condition which should be reloaded from
            source before each invocation or iteration, respectively.
        interactive_exception:
            Exceptions raised from reloading code are caught and you can fix
            the code without losing state.
    """

    def wrap(x):
        if callable(x):
            return _reloading_function(x, interactive_exception)
        else:
            raise TypeError(f'reloading expected function, got'
                            f', "{type(fn_or_seq_or_bool)}"')

    if fn_or_seq_or_bool is not None:
        if callable(fn_or_seq_or_bool):
            return _reloading_function(fn_or_seq_or_bool,
                                       interactive_exception)
        elif (isinstance(fn_or_seq_or_bool, Iterable) or
              isinstance(fn_or_seq_or_bool, bool)):
            return _reloading_loop(fn_or_seq_or_bool,
                                   interactive_exception)
        else:
            raise TypeError(
                f'reloading expected function. iterable or bool'
                f', got "{type(fn_or_seq_or_bool)}"'
            )
    else:
        # If reloading was called as a decorator with an argument,
        # then we expect fn_or_seq_or_bool to be None, which is OK.
        # However, if reloading was not called as a decorator and it
        # did not get an argument then we assume that the user desired
        # infinite iteration for a loop.
        # Source: https://stackoverflow.com/questions/52191968/
        current_frame = inspect.currentframe()
        assert isinstance(current_frame, types.FrameType)
        assert isinstance(current_frame.f_back, types.FrameType)
        frame = inspect.getframeinfo(current_frame.f_back, context=1)
        assert frame.code_context is not None
        # Remove whitespace due to indentation before .startswith
        if frame.code_context[0].strip().startswith("@"):
            return wrap
        else:
            return _reloading_loop(itertools.count(), interactive_exception)


def unique_name(seq: itertools.chain) -> str:
    """
    Function to generate string which is unique
    relative to the supplied sequence.
    """
    return max(seq, key=len) + "0"


def format_iteration_variables(ast_node: Union[ast.Name,
                                               ast.Tuple,
                                               ast.List,
                                               None]) -> str:
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


def load_file(filename: str) -> str:
    """
    Read contents of file containing reloading code.
    Handle case of file appearing empty on read.
    """
    src = ""
    while True:
        with open(filename, "r", encoding="utf-8") as f:
            if filename.endswith(".ipynb"):
                import nbformat
                # Read Jupyter Notebook v. 4
                notebook = nbformat.read(f, 4)
                # Create list of all code blocks
                blocks = [cell.source for cell in notebook.cells
                          if cell["cell_type"] == "code"]
                # Join all blocks (a block is a multiline string of code)
                jupyter_code = "\n".join(blocks)
                # Jupyter has a magic meaning of !. Lines which start
                # with "!" are not Python code. Comment them out.
                lines = [line.replace("!", "# !", 1) if line.startswith("!")
                         else line for line in jupyter_code.split("\n")]
                src = "\n".join(lines)
            else:
                src = f.read()
        if len(src):
            return src + "\n"


def parse_file_until_successful(filename: str,
                                interactive_exception: bool) -> ast.Module:
    """
    Parse source code of file containing reloading code.
    File may appear incomplete as as it is read so retry until successful.
    """
    source = load_file(filename)
    while True:
        try:
            tree = ast.parse(source)
            return tree
        except SyntaxError:
            handle_exception(filename, interactive_exception)
            source = load_file(filename)


break_ast = ast.parse('raise Exception("break")').body
continue_ast = ast.parse('raise Exception("continue")').body


class ReplaceBreakContinueWithExceptions(ast.NodeTransformer):
    def visit_Break(self, node):
        return break_ast

    def visit_Continue(self, node):
        return continue_ast


class WhileLoop:
    """
    Object to hold ast and test-function for a reloading while loop.
    """
    def __init__(self,
                 ast_module: ast.Module,
                 test: ast.Call,
                 filename: str,
                 node_id: str):
        self.ast: ast.Module = ast_module
        self.test: ast.Call = test
        self.id: str = node_id
        # Replace "break" and "continue" with custom exceptions.
        # Otherwise SyntaxError is raised because these instructions
        # are called outside a loop.
        ReplaceBreakContinueWithExceptions().visit(ast_module)
        ast.fix_missing_locations(ast_module)
        self.compiled_body = compile(ast_module,
                                     filename=filename,
                                     mode="exec")
        # If no argument was supplied, then loop forever
        ast_condition = ast.Expression(body=ast.Constant(True))
        if len(test.args) > 0:
            # Create expression to evaluate condition
            # reloading only takes one argument, so we can
            # pick the first element of the args.
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
                 filename: str,
                 node_id: str):
        self.ast: ast.Module = ast_module
        # Replace "break" and "continue" with custom exceptions.
        # Otherwise SyntaxError is raised because these instructions
        # are called outside a loop.
        ReplaceBreakContinueWithExceptions().visit(ast_module)
        ast.fix_missing_locations(ast_module)
        self.compiled_body = compile(ast_module,
                                     filename=filename,
                                     mode="exec")
        self.iteration_variables: Union[ast.Name,
                                        ast.Tuple,
                                        ast.List] = iteration_variables
        self.iteration_variables_str: str = format_iteration_variables(
                                            iteration_variables)
        self.id: str = node_id


def get_loop_object(loop_frame_info: inspect.FrameInfo,
                    reloaded_file_ast: ast.Module,
                    filename: str,
                    loop_id: Union[None, str]) -> Union[WhileLoop, ForLoop]:
    """
    Traverse AST for the entire reloaded file in a search for the
    loop which is reloaded.
    """
    Parentage().visit(reloaded_file_ast)
    candidates: List[Union[ast.For, ast.While]] = []
    for node in ast.walk(reloaded_file_ast):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            # Handle "for reloading(iter)" as well as
            # "for reloading.reloading(iter)".
            if (getattr(node.iter.func, "id", "") == "reloading" or
               getattr(node.iter.func, "attr", "") == "reloading") and (
               (loop_id is not None and loop_id == get_loop_id(node))
               or getattr(node, "lineno", None) == loop_frame_info.lineno):
                candidates.append(node)
        if isinstance(node, ast.While) and isinstance(node.test, ast.Call):
            if (getattr(node.test.func, "id", "") == "reloading" or
               getattr(node.test.func, "attr", "") == "reloading") and (
               (loop_id is not None and loop_id == get_loop_id(node))
               or getattr(node, "lineno", None) == loop_frame_info.lineno):
                candidates.append(node)

    if len(candidates) == 0 and loop_id is None:
        raise ReloadingException("Reloading used outside a loop.")
    elif len(candidates) == 0 and loop_id:
        raise ReloadingException(
            f'Unable to reload loop initially defined at line '
            f'{loop_frame_info.lineno} '
            f'in file "{filename}". '
            'The loop might have been removed.'
        )

    # Select the candidate node which is closest to function_frame_info
    def sorting_function(candidate):
        return abs(candidate.lineno - loop_frame_info.lineno)
    candidate = min(candidates, key=sorting_function)
    loop_node_ast = ast.Module(candidate.body, type_ignores=[])
    if isinstance(candidate, ast.For):
        assert isinstance(candidate.target, (ast.Name, ast.Tuple, ast.List))
        return ForLoop(loop_node_ast,
                       candidate.target,
                       filename,
                       get_loop_id(candidate))
    elif isinstance(candidate, ast.While):
        assert isinstance(candidate.test, ast.Call)
        return WhileLoop(loop_node_ast,
                         candidate.test,
                         filename,
                         get_loop_id(candidate))
    raise ReloadingException("No loop node found.")


def get_loop_id(ast_node: Union[ast.For, ast.While]) -> str:
    """
    Generates a unique identifier for an `ast_node`.
    Used to identify the loop in the changed source file.
    """
    if isinstance(ast_node, ast.For):
        return "_".join([get_node_id(ast_node),
                         ast.dump(ast_node.target),
                         ast.dump(ast_node.iter)])
    elif isinstance(ast_node, ast.While):
        return "_".join([get_node_id(ast_node),
                         ast.dump(ast_node.test)])


def get_loop_code(loop_frame_info: inspect.FrameInfo,
                  loop_id: Union[None, str],
                  filename: str,
                  interactive_exception: bool) -> Union[WhileLoop, ForLoop]:
    while True:
        reloaded_file_ast: ast.Module = parse_file_until_successful(
            filename,
            interactive_exception
        )
        try:
            return get_loop_object(
                loop_frame_info, reloaded_file_ast, filename, loop_id=loop_id
            )
        except (LookupError, ReloadingException):
            handle_exception(filename, interactive_exception=False)


def handle_exception(filename: str, interactive_exception):
    """
    Output helpful error message to user regarding exception in reloaded code.
    """
    # Report traceback stack starting with the reloaded file.
    # This avoids listing stack frames from this library.
    # Source: https://stackoverflow.com/questions/45771299
    frame_summaries = traceback.extract_tb(sys.exc_info()[2])
    count = len(frame_summaries)
    # find the first occurrence of the module file name
    for frame_summary in frame_summaries:
        if frame_summary.filename == filename:
            break
        count -= 1
    exception_text = traceback.format_exc(limit=-count)
    # Even though "filename" is passed to calls to compile,
    # we still have to replace the filename when an error
    # occours during compiling.
    exception_text = exception_text.replace('File "<string>"',
                                            f'File "{filename}"')
    exception_text = exception_text.replace('File "<unknown>"',
                                            f'File "{filename}"')
    if sys.stdin.isatty() and interactive_exception:
        log.error("Exception occourred. Press <Enter> to continue "
                  "or <CTRL+C> to exit:\n" + exception_text)
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            log.info("\nExiting...")
            sys.exit(1)
    else:
        raise


def execute_for_loop(seq: Iterable,
                     loop_frame_info: inspect.FrameInfo,
                     filename: str,
                     interactive_exception: bool):
    caller_globals: Dict[str, Any] = loop_frame_info.frame.f_globals
    caller_locals: Dict[str, Any] = loop_frame_info.frame.f_locals

    # Initialize variables
    file_stat: int = 0
    vacant_variable_name: str = ""
    assign_compiled = compile("", filename="", mode="exec")
    for_loop = get_loop_code(loop_frame_info,
                             None,
                             filename,
                             interactive_exception)

    for i, iteration_variable_values in enumerate(seq):
        # Reload code if possibly modified
        file_stat_: int = os.stat(filename).st_mtime_ns
        if file_stat != file_stat_:
            if i > 0:
                log.info(f'For loop at line {loop_frame_info.lineno} of file '
                         f'"{filename}" has been reloaded.')
            ast_before = for_loop.ast
            for_loop = get_loop_code(loop_frame_info,
                                     for_loop.id,
                                     filename,
                                     interactive_exception)
            assert isinstance(for_loop, ForLoop)
            ast_after = for_loop.ast
            log.debug(get_diff_text(ast_before, ast_after))
            file_stat = file_stat_
            # Make up a name for a variable which is not already present in
            # the global or local namespace.
            vacant_variable_name = unique_name(
                itertools.chain(caller_locals.keys(), caller_globals.keys())
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
                handle_exception(filename, interactive_exception)


def execute_while_loop(loop_frame_info: inspect.FrameInfo,
                       filename: str,
                       interactive_exception: bool):
    caller_globals: Dict[str, Any] = loop_frame_info.frame.f_globals
    caller_locals: Dict[str, Any] = loop_frame_info.frame.f_locals

    file_stat: int = os.stat(filename).st_mtime_ns
    while_loop = get_loop_code(loop_frame_info,
                               None,
                               filename,
                               interactive_exception)

    def condition(while_loop):
        return eval(while_loop.condition, caller_globals, caller_locals)

    i = 0
    while condition(while_loop):
        i += 1
        # Reload code if possibly modified
        file_stat_: int = os.stat(filename).st_mtime_ns
        if file_stat != file_stat_:
            log.info(f'While loop at line {loop_frame_info.lineno} of file '
                     f'"{filename}" has been reloaded.')
            ast_before = while_loop.ast
            while_loop = get_loop_code(loop_frame_info,
                                       while_loop.id,
                                       filename,
                                       interactive_exception)
            ast_after = while_loop.ast
            log.debug(get_diff_text(ast_before, ast_after))
            file_stat = file_stat_
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
                handle_exception(filename, interactive_exception)


def _reloading_loop(seq: Union[Iterable, bool],
                    interactive_exception) -> Iterable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack()
    # i.e. _reloading_loop
    assert stack[0].function == "_reloading_loop"
    # The second element is the caller of the first, i.e. reloading
    assert stack[1].function == "reloading"
    # The third element is the loop which called reloading.
    loop_frame_info: inspect.FrameInfo = stack[2]
    filename: str = loop_frame_info.filename
    # If we are running in Jupyter Notebook then the filename
    # of the current notebook is stored in the __session__ variable.
    if ".ipynb" in loop_frame_info.frame.f_globals.get("__session__", ""):
        filename = str(loop_frame_info.frame.f_globals.get("__session__"))
    # Replace filename with Jupyter Notebook file name
    # if we are in a Jupyter Notebook session.
    loop_object = get_loop_code(loop_frame_info,
                                None,
                                filename,
                                interactive_exception)

    if isinstance(loop_object, ForLoop):
        assert isinstance(seq, Iterable)
        execute_for_loop(seq,
                         loop_frame_info,
                         filename,
                         interactive_exception)
    elif isinstance(loop_object, WhileLoop):
        execute_while_loop(loop_frame_info,
                           filename,
                           interactive_exception)

    # If there is a third element, then it is the scope which called the loop.
    # If this is the main scope, then all is good. Howver, if we are in a
    # scope within the main scope then it is only possible to modify
    # variables in this scope since Python 3.13.
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
    elif (len(stack) > 3 and
          loop_frame_info.frame.f_locals.get("__name__", "") != "__main__"):
        variables = ", ".join(
                    f'"{k}"' for k in loop_frame_info.frame.f_locals.keys())
        log.warning(f"Variable(s) {variables} in reloaded loop were not "
                    "exported to the scope which called the reloaded loop "
                    f'initially defined at line {loop_frame_info.lineno} in '
                    f'file "{filename}".')
    return []


def get_decorator_name_or_none(decorator_node):
    if hasattr(decorator_node, "id"):
        return decorator_node.id
    elif hasattr(decorator_node, "attr"):
        return decorator_node.attr
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
    if "reloading" in decorator_names:
        # Find index of "reloading" decorator
        reloading_index = decorator_names.index("reloading")
        fwod.decorator_list = fwod.decorator_list[reloading_index + 1:]
    function_without_decorator = fwod
    return function_without_decorator


# Source: https://stackoverflow.com/questions/34570992/
class Parentage(ast.NodeTransformer):
    # current parent (module)
    parent = None

    def visit(self, node: Any):
        # set parent attribute for this node
        node.parent = self.parent
        # This node becomes the new parent
        self.parent = node
        # Do any work required by super class
        node = super().visit(node)
        # If we have a valid node (ie. node not being removed)
        if isinstance(node, ast.AST):
            # update the parent, since this may have been transformed
            # to a different node by super
            self.parent = getattr(node, "parent")
        return node


def get_node_id(node) -> str:
    path = ""
    while node.parent:
        path = node.__class__.__name__+("." if path else "")+path
        node = node.parent
    return path


class Function:
    def __init__(self,
                 function_name: str,
                 function_frame_info: inspect.FrameInfo,
                 ast_module: ast.Module,
                 filename: str,
                 node_id: str):
        self.ast = ast_module
        self.id = node_id
        self.name = function_name
        caller_locals = function_frame_info.frame.f_locals
        caller_globals = function_frame_info.frame.f_globals
        # Copy locals to avoid exec overwriting the decorated function with
        # the new undecorated function.
        caller_locals_copy = caller_locals.copy()
        caller_globals_copy = caller_globals.copy()
        # Variables that are local to the calling scope
        # are global to the function.
        caller_globals_copy.update(caller_locals_copy)
        compiled_body = compile(ast_module, filename=filename, mode="exec")
        exec(compiled_body, caller_globals_copy, caller_locals_copy)
        self.function = caller_locals_copy[function_name]


def get_function_object(function_frame_info: inspect.FrameInfo,
                        function: Callable,
                        reloaded_file_ast: ast.Module,
                        filename: str,
                        function_id: Union[None, str] = None) -> Function:
    """
    Traverse AST of the entire reloaded file in a search for the
    function (minus the reloading decorator) which is reloaded.
    """
    qualname = function.__qualname__
    function_name = qualname.split(".")[-1]

    candidate = None
    Parentage().visit(reloaded_file_ast)
    relevant_nodes = []
    for node in ast.walk(reloaded_file_ast):
        if (isinstance(node, ast.FunctionDef) and
           node.name == function.__name__):
            relevant_nodes.append(node)
            if function_id is None:
                # If we don't have an ID, then it is because this is the
                # first time we get the function object. In this case, we
                # can assume that the function object and the AST are in sync.
                # That is, if the line numbers match then it's all good.
                if node.lineno == function.__code__.co_firstlineno:
                    candidate = node
                    break
                # Okay, so the line numbers don't match exactly. This could be
                # because of decorators. Check if the function is decorated
                # for reloading and that the line numbers are plausible.
                node_l = node.lineno
                function_l = function.__code__.co_firstlineno
                if all(["reloading" in [get_decorator_name_or_none(decorator)
                       for decorator in node.decorator_list],
                       node_l > function_l,
                       node_l - function_l <= len(node.decorator_list)]):
                    candidate = node
                    break
            # If the node IDs match then its a sure thing.
            if get_node_id(node) == function_id:
                candidate = node
                break
    if candidate is None and len(relevant_nodes) == 1:
        candidate = relevant_nodes[0]
    elif candidate is None and len(relevant_nodes) > 1:
        raise ReloadingException(
            f'File "{filename}" contains '
            f'{len(relevant_nodes)} definitions of function '
            f'"{function_name}" and it is not possible '
            f'to determine which to reload.')
    # Select the candidate node which is closest to function_frame_info
    if not candidate:
        raise ReloadingException(
            f'Unable to reload function "{function_name}" '
            f'in file "{filename}". '
            'The function might have been renamed or the '
            'decorator might have been removed.'
        )
    function_id = get_node_id(candidate)
    function_node = strip_reloading_decorator(candidate)
    function_node_ast = ast.Module([function_node], type_ignores=[])
    return Function(function.__name__,
                    function_frame_info,
                    function_node_ast,
                    filename,
                    function_id)


def get_reloaded_function(function_frame_info: inspect.FrameInfo,
                          function: Callable,
                          filename: str,
                          function_id: Union[None, str],
                          interactive_exception: bool) -> Function:
    reloaded_file_ast: ast.Module = parse_file_until_successful(
        filename,
        interactive_exception
    )
    return get_function_object(function_frame_info,
                               function,
                               reloaded_file_ast,
                               filename,
                               function_id)


def _reloading_function(function: Callable,
                        interactive_exception: bool) -> Callable:
    stack: List[inspect.FrameInfo] = inspect.stack()
    # The first element on the stack is the caller of inspect.stack()
    # That is, this very function.
    assert stack[0].function == "_reloading_function"
    index = 2
    if stack[1].function == "reloading":
        pass
    elif stack[1].function == "wrap" and stack[2].function == "reloading":
        index = 3
    # The third/fourth element or later is the function which called reloading.
    # Assume it's the third.
    function_frame_info: inspect.FrameInfo = stack[index]
    # Look to see if theres a better frame in the stack.
    for frame_info in stack[index:]:
        names_global = set(frame_info.frame.f_globals.keys())
        names_local = set(frame_info.frame.f_locals.keys())
        variables = names_local | names_global
        if all([function.__name__ in variables,
                frame_info.filename == function.__code__.co_filename]):
            function_frame_info = frame_info
    filename: str = function.__code__.co_filename
    # If we are running in Jupyter Notebook then the filename
    # of the current notebook is stored in the __session__ variable.
    if ".ipynb" in function_frame_info.frame.f_globals.get("__session__", ""):
        filename = str(function_frame_info.frame.f_globals.get("__session__"))

    file_stat: int = os.stat(filename).st_mtime_ns
    function_object = get_reloaded_function(function_frame_info,
                                            function,
                                            filename,
                                            None,
                                            interactive_exception)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        nonlocal file_stat, function_object
        # Reload code if possibly modified
        file_stat_: int = os.stat(filename).st_mtime_ns
        if file_stat != file_stat_:
            log.info(f'Function "{function.__name__}" initially defined at '
                     f'line {function_frame_info.lineno} '
                     f'of file "{filename}" has been reloaded.')
            ast_before = function_object.ast
            function_object = get_reloaded_function(function_frame_info,
                                                    function,
                                                    filename,
                                                    function_object.id,
                                                    interactive_exception)
            ast_after = function_object.ast
            log.debug(get_diff_text(ast_before, ast_after))
            file_stat = file_stat_
        while True:
            try:
                return function_object.function(*args, **kwargs)
            except Exception:
                handle_exception(filename, interactive_exception)

    return wrapped
