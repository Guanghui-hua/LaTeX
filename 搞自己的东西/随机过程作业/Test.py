import numpy as np

pi = np.array([.25, .25, .25, .25])
A = np.array([
    [0,  1,  0, 0],
    [.4, 0, .6, 0],
    [0, .4, 0, .6],
    [0, 0, .5, .5]])
B = np.array([
    [.5, .5],
    [.3, .7],
    [.6, .4],
    [.8, .2]])

def get_data_with_dist(dist):
    r = np.random.rand()
    print(r)
    for i, p in enumerate(dist):
        if r < p: return i
        r -= p

def generate(T: int):
        '''
        根据给定的参数生成观测序列
        T: 指定要生成数据的数量
        '''
        z = get_data_with_dist(pi)    # 根据初始概率分布生成第一个状态
        x = get_data_with_dist(B[z])  # 生成第一个观测数据
        result = [x]
        for _ in range(T-1):        # 依次生成余下的状态和观测数据
            z = get_data_with_dist(A[z])
            x = get_data_with_dist(B[z])
            result.append(x)
        return result

generate(3)


while True:
    try:
        x = int( input("请输入一个整数："))
        print(x)
    except ValueError:
        print("值异常")
    except :
        print("Error!")
    else:
        print("输入正确！")
        break

try:
    x,y = eval(input("请输入两个数，以逗号分隔："))
    z = x/y
    print(z)
except ValueError:
    print("...ValueError")
else:
    print("输入正确！")
finally:
    print("执行完毕！")


def test(x:int):
    print(x)

test("123")
test(23)
test("Hello")


#%%
print("test")
print("Hello world")
#%%
for _ in range(10):
    print("Hello world")
# %%
