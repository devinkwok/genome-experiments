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
        for i in range(cur_pos, position):
            yield b'N'
        yield major_allele
        cur_pos = position
