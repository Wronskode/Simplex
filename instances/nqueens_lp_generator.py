for n in range(8, 26):
    filetxt = ""
    filetxt += "min: z;\n"
    for j in range(1, n+1):
        s = ""
        for i in range(1, n+1):
            s = s + "q_"+str(i)+"_"+str(j)+" + "
        s = s[0:len(s)-3]
        s = s + " = 1;\n"
        filetxt += s

    for i in range(1, n+1):
        s = ""
        for j in range(1, n+1):
            s = s + "q_"+str(i)+"_"+str(j)+" + "
        s = s[0:len(s)-3]
        s = s + " = 1;\n"
        filetxt += s

    for d in range(-(n-2), n-1):
        s = ""
        for i in range(1, n+1):
            j = i - d
            if 1 <= j <= n:
                s += f"q_{i}_{j} + "
        if s:
            s = s[:-3]
            s += " <= 1;\n"
            filetxt += s

    for d in range(3, 2*n+1):
        s = ""
        for i in range(1, n+1):
            j = d - i
            if 1 <= j <= n:
                s += f"q_{i}_{j} + "
        if s:
            s = s[:-3]
            s += " <= 1;\n"
            filetxt += s

    for j in range(1, n+1):
        for i in range(1, n+1):
            filetxt += "q_"+str(i)+"_"+str(j)+" <= 1;\n"

    for j in range(1, n+1):
        for i in range(1, n+1):
            filetxt += "int q_"+str(i)+"_"+str(j)+";\n"

    with open(str(n)+"queens.lp", "w") as f:
        f.write(filetxt)