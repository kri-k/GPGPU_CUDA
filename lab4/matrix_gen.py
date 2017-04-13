import sys
import random

DIAG = 2
FILLER = 1

MIN_FACTOR = -100
MAX_FACTOR = 100

FACTOR_RANDOM_FUNC = random.uniform
# FACTOR_RANDOM_FUNC = random.randint

ELEM_FORMAT = '{:e}'
# ELEM_FORMAT = '{:d}'

if len(sys.argv) < 2:
    print('specify number of equations')
    sys.exit(1)
try:
    n = int(sys.argv[1])
    if len(sys.argv) > 2:
        fout = open(sys.argv[2], 'w')
    else:
        fout = sys.stdout
        sys.stdout.close = lambda: None
except Exception as e:
    print(e)
    sys.exit(1)

print = lambda *args, f=print, **kwargs: f(*args, file=fout, **kwargs)

order_list = random.sample(range(n), n)

l = [(i, FACTOR_RANDOM_FUNC(MIN_FACTOR, MAX_FACTOR)) for i in range(n)]

row_mul = {}
row_mul.update(random.sample(l, random.randint(0, n)))
col_mul = {}
col_mul.update(random.sample(l, random.randint(0, n)))

solution = tuple(range(-9, 10))
answer = []

print(n)
for row in order_list:
    cur_answ = 0
    for col in range(n):
        elem = DIAG if row == col else FILLER
        if row in row_mul.keys():
            elem *= row_mul[row]
        if col in col_mul.keys():
            elem *= col_mul[col]
        cur_answ += elem * solution[col % len(solution)]
        print(ELEM_FORMAT.format(elem), end=' ')
    print()
    answer.append(cur_answ)

for i in answer:
    print(ELEM_FORMAT.format(i), end=' ')

fout.close()
