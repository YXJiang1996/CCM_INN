import re

points = []
dic = {}

def selectType(point):
    zeros_idx = ''
    for i in range(6):
        if point[i] != 0:
            zeros_idx += str(i)
    if zeros_idx not in dic.keys():
        dic[zeros_idx] = len(points)
        points.append([])
    points[dic[zeros_idx]].append(point)

# 读取数据点, 点分类
with open('ccm2flux.log') as f:
    content = f.readlines()
    for line in content:
        if re.match('######## Test Sample 1 ########', line):
            break
        if re.match(r'\[.+\]', line):
            print(line)
            s = line[1:-1]
            # 分出六个数字
            group = re.findall(r'\d+.\d*(?= *)', s)
            point = (float(group[0]),
                     float(group[1]),
                     float(group[2]),
                     float(group[3]),
                     float(group[4]),
                     float(group[5]))
            # 选择类别
            selectType(point)
# acc = 0
# for k in dic.keys():
#     acc += len(points[dic[k]])
#     print('{}: {}'.format(k, len(points[dic[k]])))
# print(acc)

# 维度压缩
pressed_points = []
for k in dic.keys():
    print(k)
    group = list(map(lambda x: [x[int(s)] for s in k], points[dic[k]]))
    pressed_points.append(group)

# 开始画分布和分桶
bucket = 0.1
rec = []
cur = 0
for ppt in pressed_points:
    n = len(ppt[0])
    rec.append({})
    for p in ppt:
        content = ''
        for i in range(n):
            # 得到桶数
            num = int(p[i] / bucket)
            content += str(num) + ','
        content = content[:-1]
        current = rec[cur]

        # 将得到数值放入字典, 并计算桶中点数和点的累加求和
        if content not in current.keys():
            current[content] = [1, p]
        else:
            current[content][0] += 1
            current[content][1] = list(map(lambda x, y: x + y, current[content][1], p))
    rec[cur] = dict(sorted(rec[cur].items(), key=lambda item: item[0]))
    cur += 1

# 再求平均值即可
for i in range(len(pressed_points)):
    current = rec[i]
    for key in current.keys():
        rec[i][key][1] = list(map(lambda x: x / rec[i][key][0], rec[i][key][1]))
# print(sorted(rec[i].items(), key=lambda it: -it[1][0]))

