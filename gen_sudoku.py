s = "min: z;\n"
rows = range(0, 9)
cols = range(0, 9)
grids = range(0, 9)
values = range(1, 10)
for row in rows:
    for col in cols:
        for value in values:
            s += f"x_{row}_{col}_{value} + "
        s = s[:-2] + "= 1;\n"

for col in cols:
    for value in values:
        for row in rows:
            s += f"x_{row}_{col}_{value} + "
        s = s[:-2] + "= 1;\n"
    
for row in rows:
    for value in values:
        for col in cols:
            s += f"x_{row}_{col}_{value} + "
        s = s[:-2] + "= 1;\n"

for i0 in range(1, 4):
    for j0 in range(1, 4):
        for value in values:
            s += f"x_{3*i0-2}_{3*j0-2}_{value} + "
        s = s[:-2] + "= 1;\n"
for row in rows:
    for col in cols:
        for value in values:
            s += f"int x_{row}_{col}_{value};\n"
s = s[:-1]
with open("instances/sudoku.lp", "w") as f:
    f.write(s)