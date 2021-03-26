# In[1]:
import numpy as np
import matplotlib.pyplot as plt

# plt.plot()只有一个输入列表或数组时，参数被当作Y轴，X轴以索引自动生成
plt.plot([3, 1, 4, 5, 2])
plt.ylabel('Grade')
# plt.savefig()将输出图形存储为文件，默认PNG格式，可以通过dpi修改输出质量
# plt.savefig('test',dpi=600)
plt.show()

# plt.plot(x,y)当有两个以上参数时，按照X轴和Y轴顺序绘制数据点
plt.plot([0, 2, 4, 6, 8], [3, 1, 4, 5, 2])
plt.ylabel('Grade')
plt.axis([-1, 10, 0, 6])  # 设置横纵坐标尺度 先x后y
plt.show()

'''
plt.subplot 分割绘图区域
(nrows,ncols,plot_number)
对应 横轴块数，纵轴块数， 第几个区域
123
456   表示区域对应的编号
789
plt.subplot(3,2,4) ==plt.subplot(324)
'''


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


a = np.arange(0, 5, 0.2)
plt.subplot(211)
plt.plot(a, f(a))
plt.subplot(2, 1, 2)
plt.plot(a, np.cos(2 * np.pi * a), 'r--')
plt.show()
# In[2]:
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

'''
plt.plot(x,y,format_String,*kwargs)
∙ x : X轴数据，列表或数组，可选
∙ y : Y轴数据，列表或数组
∙format_string: 控制曲线的格式字符串，可选
∙ **kwargs : 第二组或更多(x,y,format_string)
. 当绘制多条曲线时，各条曲线的x不能省略
'''
a = np.arange(10)
plt.plot(a, a * 1.5, a, a * 2.5, a, a * 3.5, a, a * 4.5)
plt.show()

'''
=============================================================
∙ format_string: 控制曲线的格式字符串，可选
           由  颜色字符、风格字符  和  标记字符  组成
           颜色字符说明颜色字符说明
           'b' 蓝色                      'm' 洋红色magenta
           'g' 绿色                      'y' 黄色
           'r' 红色                      'k' 黑色
           'c' 青绿色cyan                'w' 白色
           '#008000' RGB某颜色           '0.8' 灰度值字符串
=============================================================
风格字符说明
             '‐' 实线           '‐‐' 破折线
             '‐.' 点划线         ':' 虚线
             '' ' ' 无线条
=============================================================
标记字符说明
'.' 点标记          '1' 下花三角标记            'h' 竖六边形标记
',' 像素标记(极小点) '2' 上花三角标记            'H' 横六边形标记
'o' 实心圈标记      '3' 左花三角标记            '+' 十字标记
'v' 倒三角标记         '4' 右花三角标记         'x' x标记
'^' 上三角标记         's' 实心方形标记         'D' 菱形标记
'>' 右三角标记         'p' 实心五角标记         'd' 瘦菱形标记
'<' 左三角标记         '*' 星形标记            '|' 垂直线标记
=============================================================
'''

plt.plot(a, a * 1.5, 'go-', a, a * 2.5, 'rx', a, a * 3.5, '*', a, a * 4.5, 'b-.')
plt.show()

# In[3]:
import matplotlib.pyplot as plt
import numpy as np

'''
pyplot并不默认支持中文显示，需要rcparams修改字体实现
========================================================
'font.family' 用于显示字体的名字
'font.style' 字体风格，正常'normal'或斜体'italic'
'font.size' 字体大小，整数字号或者'large'、'x‐small'
========================================================
      中文字体的种类
rcParams['font.family']
中文字体        说明
'SimHei'       中文黑体
'Kaiti'        中文楷体
'LiSu'         中文隶书
'FangSong'     中文仿宋
'YouYuan'      中文幼圆
'STSong'       华文宋体
========================================================
'''

plt.rcParams['font.family'] = 'YouYuan'  # 'SimHei'是黑体
plt.plot([3, 1, 4, 5, 2])
plt.ylabel(r"纵轴（值）")
plt.show()

'''


matplotlib.rcParams['font.family']='STSong'
matplotlib.rcParams['font.size']=20
a=np.arange(0.0,5.0,0.02)
plt.xlabel('横轴：时间')
plt.ylabel('纵轴：振幅')
plt.plot(a,np.cos(2*np.pi*a),'r--')
plt.show()


plt.xlabel('横轴：时间',fontproperties='SimHei',fontsize=20)
plt.ylabel('纵轴：振幅',fontproperties='SimHei',fontsize=20)
plt.plot(a,np.cos(2*np.pi*a),'bh')
plt.show()

'''
# In[4]:
import numpy as np
import matplotlib.pyplot as plt

'''
pylot文本显示函数
函数                说明
plt.xlabel()       对X轴增加文本标签
plt.ylabel()       对Y轴增加文本标签
plt.title()        对图形整体增加文本标签
plt.text()         在任意位置增加文本
plt.annotate()     在图形中增加带箭头的注解
'''

a = np.arange(0.0, 5.0, 0.02)
plt.plot(a, np.cos(2 * np.pi * a), 'r--')

plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
# $y=cos(2\pi x)$  Latex格式的文本
plt.title(r'正弦波实例 $y=cos(2\pi x)$', fontproperties='SimHei', fontsize=25)
plt.text(2, 1, r'\mu=100$', fontsize=15)

plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()

# plt.annotate(s, xy=arrow_crd, xytext=text_crd, arrowprops=dict)
plt.plot(a, np.cos(2 * np.pi * a), 'r--')

plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2\pi x)$', fontproperties='SimHei', fontsize=25)
plt.annotate(r'$\mu=100$', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.1, width=2))
plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()

# In[5]:
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

'''
函数                                        说明
plt.plot(x,y,fmt,…)                       绘制一个坐标图
plt.boxplot(data,notch,position)          绘制一个箱形图
plt.bar(left,height,width,bottom)         绘制一个条形图
plt.barh(width,bottom,left,height)        绘制一个横向条形图
plt.polar(theta, r)                       绘制极坐标图
plt.pie(data, explode)                    绘制饼图
==============================================================================
plt.psd(x,NFFT=256,pad_to,Fs)             绘制功率谱密度图
plt.specgram(x,NFFT=256,pad_to,F)         绘制谱图
plt.cohere(x,y,NFFT=256,Fs)               绘制X‐Y的相关性函数
plt.scatter(x,y)                          绘制散点图，其中，x和y长度相同
plt.step(x,y,where)                       绘制步阶图
plt.hist(x,bins,normed)                   绘制直方图  bins为直方图的个数
==============================================================================
plt.contour(X,Y,Z,N)                      绘制等值图
plt.vlines()                              绘制垂直图
plt.stem(x,y,linefmt,markerfmt)           绘制柴火图
plt.plot_date()                           绘制数据日期
'''

# 饼图=========================================================================
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.show()

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

# 直方图=======================================================================
np.random.seed(0)
mu, sigma = 100, 20  # 均值和标准差
a = np.random.normal(mu, sigma, size=100)
plt.hist(a, 20, normed=1, histtype='stepfilled', facecolor='b', alpha=0.75)
plt.title('histogram')
plt.show()

np.random.seed(0)
mu, sigma = 100, 20  # 均值和标准差
a = np.random.normal(mu, sigma, size=100)
plt.hist(a, 40, normed=1)
plt.title('histogram')
plt.show()

# 极坐标===========面向对象绘制极坐标=============================================
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='polar')
#            left  height  width
bars = ax.bar(theta, radii, width=width, bottom=0.0)

for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)
plt.show()

# 散点图=======================================================================
fig, ax = plt.subplots()
ax.plot(10 * np.random.randn(100), 10 * np.random.randn(100), 'o')
ax.set_title('Simple Scatter')
plt.show()
