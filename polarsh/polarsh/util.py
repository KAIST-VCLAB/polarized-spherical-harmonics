from __future__ import annotations
import os, sys, psutil, platform, functools, string, inspect
from typing import Sequence, Union, IO, Optional, Callable, Any
from time import time
from datetime import datetime
from pathlib import Path
from textwrap import indent

import numpy as np

########################################
### Python support
########################################

def consistent_type(param_list: Sequence) -> bool:
    types = {type(param) for param in param_list}
    assert len(types) == 1
    return types.pop()

def consistent_value(param_list: Sequence) -> bool:
    values = set(param_list)
    assert len(values) == 1
    return values.pop()

def tuple_add(tup: tuple, inc:int):
    res = np.array(tup) + inc
    return tuple(res)

########################################
### String support
########################################

def str_trunc(string: str, len_max: Optional[int]=30) -> str:
    string = str(string)
    len_max = max(len_max, 10)
    mid = "..."
    length = len(string)
    if length <= len_max:
        return string
    else:
        lmid = len(mid)
        i1 = (len_max - lmid)//2
        i2 = len_max-i1-lmid
        return string[:i1] + mid + string[-i2:]

'''
`type2str` and `arg2str` are tested in `test_helpers.ipynb`.
'''
def type2str(typ) -> str:
    res = typ.__module__
    if res == "builtins":
        res = ""
    else:
        res += "."
    res += typ.__name__
    return res

def arg2str(arg) -> str:
    typ = type(arg)
    str_full = str(arg)
    len_max1 = 50
    len_max2 = 70
    str_res = type2str(typ)
    
    if typ in [set, list, tuple, dict]:
        str_res += f" len: {len(arg)}\n"
        str_res += f"\t{str_trunc(str_full, len_max=len_max2)}"
        
    elif typ == np.ndarray:
        str_res += f" {arg.dtype}[{str(arg.shape)[1:-1]}]\n"
        str_full = str_full.replace('\n', ';')
        str_res += f"\t{str_trunc(str_full, len_max=len_max2)}"
        
    else:
        str_res += f" {str_trunc(str_full, len_max=len_max1)}"
    
    if typ == np.ndarray:
        str_res += f"\n\tmax:\t{arg.max()}"
        str_res += f"\n\tmin:\t{arg.min()}"
        str_res += f"\n\tabsmin:\t{np.abs(arg).min()}"
        str_res += f"\n\tmean:\t{arg.mean()}"
        n_nans = np.isnan(arg).sum()
        str_res += f"\n\tNaNs:\t{n_nans} ({n_nans/arg.size*100 :.2f}%)"
    return str_res

def row2str_short(row: Union[list, np.ndarray]) -> str:
    """
    Examples:
        >>> row2str_short([1, 2])
        '1, 2'
        >>> row2str_short(np.zeros(2))
        '0.,0.'
    """
    if isinstance(row, (list, tuple)):
        return str(row)[1:-1]
    elif isinstance(row, np.ndarray):
        return np.array2string(row, separator=',')[1:-1]
    else:
        raise TypeError(f"Unsupported type: {type(row)}.") 

def format_complex(val: complex, format: str) -> str:
    '''
    If there is white spaces before or after the formatting character in `format`,
    the result have spaces before and after '+'.
    Examples
        > @TODO
    '''
    plus = " + " if format.lstrip() != format else "+"
    i    = " i"  if format.rstrip() != format else "i"
    format = format.strip()
    return format%(val.real) + plus + format%(val.imag) + i

def printlog(text: str, file: IO, end="\n"):
    file.write(text + end)
    print(text, end=end)

def python_memory() -> str:
    '''
    Return memory used by the current python process
    in a string, such as "227.07MiB"
    '''
    process = psutil.Process(os.getpid())
    byte = process.memory_info().rss

    order = int(np.log(byte) / np.log(1024))
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    if order >= len(units):
        order = len(units)-1
    return f"{byte/(1024**order):.2f}{units[order]}"

class ArgFormatter:
    """
    Examples
    --------
    >>> def foo(x, y=0, /, z=1, *args, u=2, **kargs):
    >>>     print(f"[foo] {x=}, {y=}, {z=}, {u=}")
    >>>     return x+y
    >>> formatter = ArgFormatter(foo, "x:{x}, y:{y}, z:{z}, u:{u}")
    >>> foo(10, 20, 30, 1, 1, u=40)
    >>> print(formatter(10, 20, 30, 1, 1, u=40))
    [foo] x=10, y=20, z=30, u=40
    x:10, y:20, z:30, u:40
    """
    def __init__(self, func: Callable, format_text: str):
        self.func = func
        self.argspec = inspect.getfullargspec(func)
        self.arg_list = self.argspec.args + self.argspec.kwonlyargs

        assert isinstance(format_text, str), TypeError(f"Invalid type {type(format_text)=}")
        for _, fname, _, _ in string.Formatter().parse(format_text):
            if (fname is not None) and (fname not in self.arg_list):
                raise KeyError(f"The format {fname} is not found in arguments of given function:\n"
                                                f"{format = }\nArguments of {self.func.__name__}: {str(self.argspec)[12:-1]}")
        self.format_text = format_text

        self.format_dict_default = dict()
        if self.argspec.defaults is not None:
            self.format_dict_default.update({arg: val for arg, val in zip(self.argspec.args[-len(self.argspec.defaults):], self.argspec.defaults)})
        if self.argspec.kwonlydefaults is not None:
            self.format_dict_default.update(self.argspec.kwonlydefaults)
    
    def __call__(self, *args, **kargs) -> str:
        format_dict = self.format_dict_default.copy()
        format_dict.update({self.argspec.args[i]: val for i,val in enumerate(args) if i < len(self.argspec.args)})
        format_dict.update(kargs)
        return self.format_text.format(**format_dict)

########################################
### Decorators
########################################
def deco_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        print(f"\n### [Function start] {func.__name__}")
        t = time()
        result = func(*args, **kargs)
        print(f"### [Function end] time: {time()-t:.6f} seconds.")
        return result
    return wrapper

def deco_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        print("\n"+"#"*30)
        print(f"### {func.__name__}")
        print("#"*30)
        return func(*args, **kargs)
    return wrapper

def deco_log(outdir_position: int):
    def deco_return(func):
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            t0 = time()
            outdir = args[outdir_position]
            Path(outdir).mkdir(exist_ok=True)
            log_file = os.path.join(outdir, f"log_{func.__name__}.txt")
            with open(log_file, 'a') as f:
                now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{'#'*5} {now_string}")
                f.write(f" from {platform.node()} {'#'*5}\n")
                
                f.write(f"co_filename: {func.__code__.co_filename}\n")
                f.write(f"py:func:{func.__name__}\n")
                if args:
                    f.write(f"Positional arguments:\n")
                    for arg in args:
                        str_item = indent(arg2str(arg), '\t')
                        f.write(f"{str_item}\n")
                if kargs:
                    f.write(f"Keyword arguments:\n")
                    for key in kargs:
                        str_item0 = f"{key}: {arg2str(kargs[key])}"
                        str_item1 = indent(str_item0, '\t') 
                        f.write(f"{str_item1}\n")
            try:
                res = func(*args, **kargs)
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write("ERROR occured!\n")
                    msg = indent(f"{type2str(type(e))}: {str(e)}", '\t')
                    f.write(msg + "\n\n")
                raise e
            else:
                with open(log_file, 'a') as f:
                    f.write("Return:\n")
                    str_item = indent(arg2str(res), '\t')
                    f.write(f"{str_item}\n")
                    
                    f.write("Elapsed time (w/ this logging):\n")
                    dt = time() - t0
                    f.write(f"\t{dt:.4f} sec\n\n")
                return res
        return wrapper            
    return deco_return


def flip_arg(func: Callable) -> Callable:
    """Expected usage is for `cached` defined below."""
    def wrapper(*args, **kargs):
        return func(args[1], args[0], *args[2:], **kargs)
    return wrapper


class cached:
    msg_cache_hit = "# Read cached `%s` object from file: %s"
    msg_no_cache = "# Start to computation due to no cached file: %s"
    msg_done = "# Finished to compute `%s` object and saved in the cache file."

    def __init__(self,
                 filename_format: Union[str, Path],
                 func_read:       Callable[[str], Any],
                 func_write:      Callable[[str, Any], Any],
                 quiet:           Optional[bool] = True,
                 tictoc:          Optional[Union[bool, Tictoc]] = False
                ):
        """
        Decorator for cached evaluation
        Parameters:
            filename: str, file name for cached data
        """
        if not quiet:
            tictoc = True
        if isinstance(tictoc, bool):
            if tictoc == True:
                tictoc = Tictoc()
        elif not isinstance(tictoc, Tictoc):
            raise TypeError(f"Invalid type of the argumnet: {type(tictoc) = }")
        self.filename_format = str(filename_format)
        self.tictoc = tictoc
        self.quiet = bool(quiet)

        self.func_read = func_read
        self.func_write = func_write

    def __call__(self, func: Callable) -> Callable:
        formatter = ArgFormatter(func, self.filename_format)
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            filename = formatter(*args, **kargs)
            if "%" in filename:
                path_for_check = Path(filename % 0) # small hack for Stokes images
            else:
                path_for_check = Path(filename)

            if path_for_check.exists():
                res = self.func_read(filename)
                if not self.quiet:
                    print(self.msg_cache_hit % (type(res), filename))
            else:
                if not self.quiet:
                    print(self.msg_no_cache % filename)
                
                if self.tictoc is False:
                    res = func(*args, **kargs)
                else:
                    with self.tictoc:
                        res = func(*args, **kargs)
                
                self.func_write(filename, res)
                if not self.quiet:
                    print(self.msg_done % (type(res), filename))
            return res
        return wrapper
    

    
########################################
### Tictoc
########################################
class Tictoc:
    def __init__(self, 
                 msg_format: Optional[str] = "Elapsed time is %.4f seconds.",
                 msg_start: Optional[str] = None,
                 file: Optional[Union[IO, Sequence[IO]]] = sys.stdout):
        if not isinstance(file, (tuple, list)):
            file = [file]
        self.file_list = file
        if "%" not in msg_format:
            msg_format = msg_format + ": %.4f seconds."
        self.msg_format = msg_format
        self.msg_start = msg_start
        self.recorded = 0
    def __enter__(self):
        if not self.msg_start is None:
            self.print(self.msg_start)
        self.t0 = time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.recorded = time() - self.t0
        self.print(self.msg_format % self.recorded)

    def __call__(self, func: Callable) -> Callable:
        """ As a decorator """
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            with self:
                res = func(*args, **kargs)
            return res
        return wrapper

    def print(self, *args, sep: Optional[str]=" ", end: Optional[str]="\n"):
        for file in self.file_list:
            print(*args, sep=sep, end=end, file=file)