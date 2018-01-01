import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')

x = data[:, 0]
y = data[:, 1]
sp.sum(sp.isnan(y))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]
plt.scatter(x, y)
plt.title('Web traffice over the last month')
plt.xlabel('Time')
plt.ylabel('Hits/hour')
plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()


def error(f, x, y):
    return sp.sum((f(x) - y)**2)

frac = 0.3
split_idx = int(frac * len(x))
shuffled = sp.random.permutation(list(range(len(x))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])

xt = x[train]
yt = y[train]

# 一维线性方程
fp1, residuals, rank, sv, rcond = sp.polyfit(xt, yt, 1, full=True)
print("Model parameters: %s" % fp1)
# print(res)
f1 = sp.poly1d(fp1)
print("Error d=1: %f" % error(f1, xt, yt))
fx = sp.linspace(0, xt[-1], 1000)  # 生成x值来作图
plt.plot(fx, f1(fx), lineWidth=4)
plt.legend(['d=%i' % f1.order], loc='upper left')

# 二维方程
fp2 = sp.polyfit(xt, yt, 2)
f2y = sp.poly1d(fp2)
print("Error d=2: %f" % error(f2y, xt, yt))
fx = sp.linspace(0, xt[-1], 1000)  # 生成x值来作图
plt.plot(fx, f2y(fx), lineWidth=4)

# 十阶方程
fp10 = sp.polyfit(x, y, 10)
f10y = sp.poly1d(fp10)
print("Error d=10: %f" % error(f10y, x, y))
fx = sp.linspace(0, x[-1], 1000)  # 生成x值来作图
plt.plot(fx, f10y(fx), lineWidth=4)
plt.legend(['d=%i' % f10y.order, 'd=%i' %
            f2y.order, 'd=%i' % f1.order], loc='upper right')

# 使用两条直线
inflection = 3.5 * 7 * 24  # 计算拐点的小时数
inflection = int(inflection)
xa = x[:inflection]  # 拐点之前的数据
ya = y[:inflection]
xb = x[inflection:]  # 之后的数据
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fax = sp.linspace(0, x.size / 2, 500)
fbx = sp.linspace(x.size / 2, x[-1], 500)
# plt.plot(fax, fa(fax), lineWidth=4)
# plt.legend(['d=%i' % fa.order], loc='upper left')
# plt.plot(fbx, fb(fbx), lineWidth=4)
# plt.legend(['fb=%i' % fb.order, 'fa=%i' % fa.order], loc='upper right')
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error inflection=%f" % (fa_error + fb_error))

# plt.show()

# 计算达到100000每小时访问量需要的时间
reached_max = fsolve(f2y - 100000, 800) / (7 * 24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
