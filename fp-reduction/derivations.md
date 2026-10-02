1 Derivation:
a= 16777216 b=1, c=1 
GROUPING 1: (a + b) + c

  a + b = 16777216 + 1 = ?
     ULP at 16777216 = 2^24-23 = 2
     true result 16777217 is half ULP above 16777216
     halfway line is at 16777217 
     16777217 is on the halfway line
     tie? if so, which neighbour has even last bit? two nieghbors 16777216 : last bit 0 → EVEN  
16777218 : last bit 1 → ODD
Ties-to-even picks the even one → 16777216.

     → a+b rounds to  16777216

  (that result) + c = 16777216 + 1 = same situation
     → rounds to  16777216

  GROUPING 1 ANSWER = 16777216


GROUPING 2: a + (b + c)

  b + c = 1 + 1 = ?
     ULP at magnitude 2 = ______  (tiny)
     does this round at all? ___no___
     → b+c = 2______

  a + (that) = 16777216 + 2______ = 16777218
     is the true result on a grid mark or between? ___on___
     → rounds to __167777218____

  GROUPING 2 ANSWER = 16777218


COMPARE:  grouping 1 = 16777216  grouping 2 = 16777218   equal? No