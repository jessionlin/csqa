def edit_distance(lst1, lst2):
    """
    0 插入
    1 删除
    2 替换
    3 无操作
    """
    m = [[i+j for i in range(len(lst2)+1)] for j in range(len(lst1)+1)]
    c = [[3 for i in range(len(lst2)+1)] for j in range(len(lst1)+1)]

    for i in range(len(lst1)+1):
        c[i][0] = 0
    for j in range(len(lst2)+1):
        c[0][j] = 1

    for i in range(1, len(lst1)+1):
        for j in range(1, len(lst2)+1):
            if lst1[i-1] == lst2[j-1]:
                d = 0
            else:
                d = 1
            m[i][j] = min(m[i-1][j]+1, m[i][j-1]+1, m[i-1][j-1]+d)

            if m[i][j] == m[i-1][j] + 1:
                c[i][j] = 0
            elif m[i][j] == m[i][j-1] + 1:
                c[i][j] = 1
            elif d == 1:
                c[i][j] = 2

    return m, c


def diff_seq(lst1, lst2):
    x = len(lst1)
    y = len(lst2)
    m, c = edit_distance(lst1, lst2)
    res_seq = [[0 for i in lst1], [0 for i in lst2]]
    while True:
        if c[x][y] == 0:
            res_seq[0][x-1] = 1
            x = x - 1
        elif c[x][y] == 1:
            res_seq[1][y-1] = 1
            y = y - 1
        elif c[x][y] == 2:
            res_seq[0][x-1] = 1
            res_seq[1][y-1] = 1
            x = x - 1
            y = y - 1
        else:
            x = x - 1
            y = y - 1

        if x == y == 0:
            break
    return res_seq


def reverse(arr):
    if len(arr) < 1:
        return []
    ret = []
    for item in arr:
        ret.append(1-item)
    return ret

def diff_seq_three(lst1, lst2, lst3):
    diff12 = diff_seq(lst1, lst2)
    diff23 = diff_seq(lst2, lst3)
    diff13 = diff_seq(lst1, lst3)
    
    x = len(diff12)
    diff1 = [diff13[i] + diff12[i] for i in range(x)]
    y = len(diff21)
    diff2 = [diff23[i] + diff21[i] for i in range(y)]
    z = len(diff32)
    diff3 = [diff31[i] + diff32[i] for i in range(z)]
    
    # x = len(diff12)
    # diff1 = [diff13[i] + diff12[i] - diff13[i] * diff12[i] for i in range(x)]
    # y = len(diff21)
    # diff2 = [diff23[i] + diff21[i] -  diff23[i] * diff21[i] for i in range(y)]
    # z = len(diff32)
    # diff3 = [diff31[i] + diff32[i] - diff31[i] * diff32[i]  for i in range(z)]
    return reverse(diff1), reverse(diff2), reverse(diff3)

if __name__ == '__main__':
    s1, s2 = diff_seq('ABCDE', 'QACD')
    print(s1, s2)
