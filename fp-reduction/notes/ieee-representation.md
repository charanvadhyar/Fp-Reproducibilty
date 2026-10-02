Ieee reprseantation of a floating point number
number = -13.625
A:sign is negative so sign bit is 1
B: integer part to binary
13 by repeated division by 2, reading remainders bottom to top:
13 ÷ 2 = 6 r 1
 6 ÷ 2 = 3 r 0
 3 ÷ 2 = 1 r 1
 1 ÷ 2 = 0 r 1
 remiander from bottom to top so 13 is 1101
 C:fractional part to binary
Multiply the fraction by 2 repeatedly; each time, the integer part that pops out (0 or 1) is the next bit, read top to bottom:
0.625 × 2 = 1.25  → bit 1   (keep .25)
0.25  × 2 = 0.5   → bit 0   (keep .5)
0.5   × 2 = 1.0   → bit 1   (keep .0 — done)
read downward
so 0.625 is 101

D normalize
Move the binary point so one nonzero digit (always 1) sits to its left:

1101.101  =  1.101101 × 2³
E: exponent is 3 beacuse poitn moved 3 places

Sign = 1 (from step 1).

Exponent = actual 3 + bias 127 = 130 = 10000010.

Mantissa = the digits after the leading "1." → 101101, padded on the right to 23 bits:

10110100000000000000000

 sign │ exponent │ mantissa
  1   │ 10000010 │ 10110100000000000000000

  1 10000010 10110100000000000000000

  All the formats that will be used in the prohject and their bits and bias

  
Format	Exp bits (e)	    2^(e−1) − 1	  Bias
fp64	11              	2¹⁰ − 1	      1023
fp32	8	                2⁷ − 1	      127
fp16	5	                2⁴ − 1	      15
bf16	8	                2⁷ − 1	      127
tf32	8	                2⁷ − 1	      127