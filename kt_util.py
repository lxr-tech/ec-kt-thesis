
def get_triple_list(path):
    tri_s, data = [], open(path, 'r')
    con_s, pro_s, ans_s = [], [], []
    for lineID, line in enumerate(data):
        line = '[' + line.strip() + ']'
        if lineID % 4 == 1:
            pro_s = eval(line)
        elif lineID % 4 == 2:
            con_s = eval(line)
        elif lineID % 4 == 3:
            ans_s = eval(line)
            tri_s.append((pro_s, con_s, ans_s))
    data.close()
    return tri_s

