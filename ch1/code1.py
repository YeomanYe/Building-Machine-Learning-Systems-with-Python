import scipy as sp
import matplotlib.pyplot as plt
data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')
# print(data[:10])
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

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
# print(res)
f1 = sp.poly1d(fp1)
print(error(f1, x, y))
fx = sp.linspace(0, x[-1], 1000)  # 生成x值来作图
plt.plot(fx, f1(fx), lineWidth=4)
plt.legend(['d=%i' % f1.order], loc='upper left')
plt.show()
