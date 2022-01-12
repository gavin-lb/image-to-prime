import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from argparse import ArgumentParser
from sympy.ntheory import isprime
from textwrap import wrap
from math import log
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
from time import perf_counter


def count_pixels(text):
    im = Image.new('1', (20, 20), 0)
    draw = ImageDraw.Draw(im)
    draw.text((1, 1), text, 1)
    return sum(im.getdata())

def image_to_ascii(im, char_set, invert=False):
    weights = {char: count_pixels(char) for char in char_set}
    chars = sorted(weights, key=weights.get, reverse=not invert)
    arr = np.array(im.convert('L'))
    bins = np.linspace(0, 255, num=len(chars))
    inds = np.digitize(arr, bins, right=True)
    return np.array(chars, dtype='U1')[inds]
    
def image_to_num(im, invert):
    arr = image_to_ascii(im, char_set='0123456789', invert=invert)
    if arr[0, 0] == '0':
        arr[0, 0] = '1'
    return int(''.join(arr.flatten()))

def check_num(num, flag):
    if not flag.is_set() and isprime(num):
        flag.set()
        return num

def find_near_prime(num):
    start = perf_counter() 
    step = int(log(num))  # by PNT we expect a prime within log(num) numbers
    print(f'Starting search for prime using {cpu_count()} processes...')
    with Pool(cpu_count()) as pool:
        func = partial(check_num, flag=Manager().Event())
        while True:
            prime = set(pool.map(func, range(num, num+step)))
            if prime == {None}:
                num += step
                print(f'Did not find a prime amongst the {step} numbers tested, trying another {step}.')
            else:
                print(f'Found prime in {perf_counter() - start:.2f} seconds.')
                prime, = prime - {None}
                return prime

    
if __name__ == '__main__':
    parser = ArgumentParser(description='Image to prime: converts the given image into an ascii-art prime number')
    parser.add_argument('-p','--path', help='Path to image file', required=True)
    parser.add_argument('-s','--size', help='Size to rescale the image to, given as space separated ints for width and height', nargs=2, type=int)
    parser.add_argument('-i','--inverted', help='If this flag is given the image will be inverted (for use on dark backgrounds)', action='store_true', default=False)
    args = parser.parse_args()
    
    im = Image.open(args.path)
    if args.size:
        im = im.resize(args.size)

    im = ImageEnhance.Contrast(im).enhance(3)
    im = ImageEnhance.Sharpness(im).enhance(3)
    
    num = image_to_num(im, args.inverted)
    prime = find_near_prime(num)
    print(*wrap(str(prime), im.size[0]), sep='\n')
