import inspect
import os.path
from pathlib import Path

# if suffix is None, return a directory
# if input_file is None, only name using prefix/suffix
# if directory is not None, append to end of path but before filename
def output_path(prefix, input_file=None, suffix=None, directory=None):
    # use filename of calling function to specify output directory
    caller = inspect.stack()[-1][1]
    out_path = os.path.join('outputs',
                os.path.split(caller)[0],
                path_to_filename(caller))
    if not directory is None:
        out_path = os.path.join(out_path, directory)
    if input_file is None:
        name = prefix
    else:
        name = prefix + path_to_filename(input_file)
    if suffix is None:  # assume this is directory
        out_path = os.path.join(out_path, name)
        # make sure directory exists
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if not suffix is None:  # assume this is file
        out_path = os.path.join(out_path, name + suffix)
    return out_path


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


def parse_variant_line(line):
    split_str = line.split()
    position = int(split_str[1])
    alleles = int(split_str[2])
    major_allele, major_allele_freq = b'X', 0.0
    for i in range(4, alleles + 4):
        allele = split_str[i].split(":")
        allele_type = allele[0]
        freq = float(allele[-1])
        if freq > major_allele_freq and len(allele_type) == 1:
            major_allele = allele_type.encode()
            major_allele_freq = freq
    return position, major_allele, major_allele_freq


def read_variants(filename, remove_lines=1):
    with open(filename, mode='r') as file:
        lines = file.readlines()
        for line in lines[remove_lines:]:
            yield parse_variant_line(line)


def read_variants_as_seq(filename):
    variants = read_variants(filename)
    cur_pos = 0
    for position, major_allele, freq in variants:
        for _ in range(cur_pos, position):
            yield b'N'
        yield major_allele
        cur_pos = position


def iterate_regions_as_seq(dataframe):
    cur_pos = 0
    for _, row in dataframe.iterrows():
        start_pos = row['start']
        end_pos = row['end']
        print(row, start_pos, end_pos)
        assert start_pos <= end_pos
        for _ in range(cur_pos, start_pos):
            yield False
        for _ in range(start_pos, end_pos + 1):
            yield True
        if cur_pos <= end_pos:
            cur_pos = end_pos + 1
