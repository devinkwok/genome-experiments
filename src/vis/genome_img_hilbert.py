import sys
sys.path.append('./src/')

import math
import os

import numpy as np
from PIL import Image

import k_mers
import seq_util.io


rgb_map = {
    b'a': 0xffb80000,
    b'c': 0xff5bb800,
    b't': 0xff00b6b8,
    b'g': 0xff5f00b8,
    b'A': 0xffff6969,
    b'C': 0xffb3ff69,
    b'T': 0xff69fdff,
    b'G': 0xffb769ff,
    b'N': 0xff000000,
    b'N': 0xff000000,
}


def make_img_from_variants(filename):
    with open(filename, 'r') as file:
        last_line = file.readlines()[-1]
    print(filename)
    n_bases, _, _ = seq_util.io.parse_variant_line(last_line)
    seq = seq_util.io.read_variants_as_seq(filename)
    out_path = seq_util.io.output_path('var_img_', filename, '.png')
    make_img(n_bases, seq, 0xffffffff, out_path)



def make_img_from_sequence(filename):
    # this is an approximate size to guide the image composition
    n_bases = os.stat(filename).st_size
    seq = seq_util.io.read_seq(filename)
    out_path = seq_util.io.output_path('seq_img_', filename, '.png')
    # default value is black
    make_img(n_bases, seq, 0xff000000, out_path)


def make_img(n_bases, seq, background_color, out_path):
    iters = int(math.ceil(math.log(n_bases, 4)))
    width = 2**iters
    
    img_array = np.zeros((width, width), dtype=np.uint32)
    coords = hilbert_curve(iters)

    for base, (x, y) in zip(seq, coords):
        img_array[x, y] = rgb_map.get(base, background_color)

    genome_img = Image.fromarray(img_array, mode='RGBA')
    genome_img.save(out_path, format="PNG")


# direction is 0, 1, 2, or 3
# iteration is recursion depth
# x, y are min corner
def hilbert_curve(iter, direction=0, x=0, y=0):
    next_iter = iter - 1
    d = 2 ** next_iter
    if iter == 0:
        yield x, y
    elif direction == 0: # recursively 
        yield from hilbert_curve(next_iter, direction=3, x=x, y=y)
        yield from hilbert_curve(next_iter, direction=0, x=x, y=y+d)
        yield from hilbert_curve(next_iter, direction=0, x=x+d, y=y+d)
        yield from hilbert_curve(next_iter, direction=1, x=x+d, y=y)
        
    elif direction == 1: # recursively 
        yield from hilbert_curve(next_iter, direction=2, x=x+d, y=y+d)
        yield from hilbert_curve(next_iter, direction=1, x=x, y=y+d)
        yield from hilbert_curve(next_iter, direction=1, x=x, y=y)
        yield from hilbert_curve(next_iter, direction=0, x=x+d, y=y)
        
    elif direction == 2: # recursively 
        yield from hilbert_curve(next_iter, direction=1, x=x+d, y=y+d)
        yield from hilbert_curve(next_iter, direction=2, x=x+d, y=y)
        yield from hilbert_curve(next_iter, direction=2, x=x, y=y)
        yield from hilbert_curve(next_iter, direction=3, x=x, y=y+d)
        
    elif direction == 3: # recursively 
        yield from hilbert_curve(next_iter, direction=0, x=x, y=y)
        yield from hilbert_curve(next_iter, direction=3, x=x+d, y=y)
        yield from hilbert_curve(next_iter, direction=3, x=x+d, y=y+d)
        yield from hilbert_curve(next_iter, direction=2, x=x, y=y + d)


def test_hilbert_curve(n):
    a = hilbert_curve(n)
    img_array = np.zeros((2**n, 2**n))
    for idx, (x, y) in enumerate(a):
        img_array[x, y] = idx
    img = Image.fromarray(np.uint8(img_array*5), mode='RGBA')
    img.save('vis_test_hilbert_curve.png')


def test_img_save():
    a = np.array([[0xffb80000, 0xff5bb800], [0xff00b6b8, 0xff5f00b8]], dtype=np.uint32)
    img = Image.fromarray(a, mode='RGBA')
    print(img.getpixel((0,0)))
    print(img.getpixel((0,1)))
    print(img.getpixel((1,0)))
    print(img.getpixel((1,1)))
    img.save('vis_test_img_save.png')


if __name__ == '__main__':
    make_img_from_sequence('data/ref_genome/test.fasta')
    make_img_from_sequence('data/ref_genome/chr22.fa')
    make_img_from_variants('data/vcf/test_freq.frq')
    make_img_from_variants('data/vcf/chr22_freq.frq')
