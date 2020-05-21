import inspect
import os.path
from pathlib import Path

def output_path(prefix, input_file, suffix):
    # use filename of calling function to specify output directory
    caller = inspect.stack()[1][1]
    output_path = os.path.join('outputs',
                os.path.split(caller)[0],
                path_to_filename(caller))
    # make sure directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    filename = prefix + path_to_filename(input_file) + suffix
    return os.path.join(output_path, filename)

def path_to_filename(path):
    return os.path.splitext(os.path.split(path)[1])[0]

def read_seq(filename, remove_lines=0, print_progress=0):
    with open(filename, mode='rb') as file:
        
        for _ in range(remove_lines):
            _ = file.readline()
        char = file.read(1)
        i = 0
        while char != b'':
            if char != b'\n':
                i += 1
                yield char
                if print_progress > 0 and i % print_progress == 0:
                    print('read entries: ' + str(i), end='\r')
            char = file.read(1)
