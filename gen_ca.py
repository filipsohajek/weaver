#!/usr/bin/env python3
import numpy as np
import sys

cpsels = [
    (1, 5), 
    (2, 6), 
    (3, 7), 
    (4, 8), 
    (0, 8), 
    (1, 9), 
    (0, 7), 
    (1, 8), 
    (2, 9), 
    (1, 2), 
    (2, 3), 
    (4, 5), 
    (5, 6), 
    (6, 7), 
    (7, 8), 
    (8, 9), 
    (0, 3), 
    (1, 4), 
    (2, 5), 
    (3, 6),
    (4, 7),
    (5, 8),
    (0, 2),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
]
def gen_ca(psel_1, psel_2):
     psel_mask = (1 << psel_1) | (1 << psel_2)
     print(psel_mask)
     g1_mask = 0b1000000100
     g2_mask = 0b1110100110
     reg_mask = 0b1111111111
     g1 = 0b1111111111
     g2 = 0b1111111111
     for i in range(1023):
         g2i = (g2 & psel_mask).bit_count() & 0x1
         yield g2i ^ (g1 >> 9)
         g1 = ((g1 << 1) & reg_mask) | ((g1 & g1_mask).bit_count() & 0x1)
         g2 = ((g2 << 1) & reg_mask) | ((g2 & g2_mask).bit_count() & 0x1)

if __name__ == "__main__":
    prn = int(sys.argv[1]) - 1
    chips = np.array(list(gen_ca(*cpsels[prn])))
    chips = np.tile(chips, 2)
    packed_chips = np.packbits(chips)
    print("\n".join(map(hex, packed_chips)))
