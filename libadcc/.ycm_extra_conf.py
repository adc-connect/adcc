import os

# This file is loosely based upon the file
# cpp/ycm/.ycm_extra_conf.py from the youcompleteme daemon process
# available on github:
# https://github.com/Valloric/ycmd/blob/master/cpp/ycm/.ycm_extra_conf.py


# These are the compilation flags that will be used in case there's no
# compilation database set (by default, one is not set).
flags = [
    # Warnings: For a very detailed discussion about this
    # see the following stackexchange post:
    # https://programmers.stackexchange.com/questions/122608#124574
    '-Werror',
    '-Wall',
    '-Wextra',
    '-pedantic',
    "-Wnon-virtual-dtor",
    "-Woverloaded-virtual",
    "-Wshadow",
    "-Wold-style-cast",
    "-Wcast-align",
    "-Wconversion",
    "-Wuseless-cast",
    "-Wsign-conversion",
    "-Wmisleading-indentation",
    "-Wduplicated-cond",
    "-Wduplicated-branches",
    "-Wlogical-op",
    "-Wnull-dereference",
    "-Wdouble-promotion",
    "-Wformat=2",
    '-fexceptions',  # Generate unwind information
    '-DDEBUG',       # Compile as debug
    '-std=c++11',    # and c++ 11
    '-x', 'c++',     # Treat .h header files as c++
    # Include other libraries and show errors and
    # warnings within them
    # To suppress errors shown here, use "-isystem"
    # instead of "-I"
    '-isystem', '~/.local/include',
    '-isystem', '/usr/local/include',
    # Explicit clang includes:
    '-isystem', '/usr/include/c++/v1',
]


DIRECTORY_OF_THIS_SCRIPT = os.path.dirname(os.path.abspath(__file__))
SOURCE_EXTENSIONS = ['.cpp', '.cxx', '.cc', '.c', '.C']


def MakeRelativePathsInFlagsAbsolute(flags, working_directory):
    if not working_directory:
        return list(flags)
    new_flags = []
    make_next_absolute = False
    path_flags = ['-isystem', '-I', '-iquote', '--sysroot=']
    for flag in flags:
        new_flag = flag

        if make_next_absolute:
            make_next_absolute = False
            if not flag.startswith('/'):
                new_flag = os.path.join(working_directory, flag)

        for path_flag in path_flags:
            if flag == path_flag:
                make_next_absolute = True
                break

            if flag.startswith(path_flag):
                path = flag[len(path_flag):]
                new_flag = path_flag + os.path.join(working_directory, path)
                break

        if new_flag:
            new_flags.append(new_flag)
    return new_flags


def IsHeaderFile(filename):
    extension = os.path.splitext(filename)[1]
    return extension in ['.h', '.hxx', '.hpp', '.hh']


def FlagsForFile(filename, **kwargs):
    relative_to = DIRECTORY_OF_THIS_SCRIPT
    final_flags = MakeRelativePathsInFlagsAbsolute(flags, relative_to)

    return {
        'flags': final_flags,
        'do_cache': True
    }
