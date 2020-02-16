import numpy as np
import matplotlib.pyplot as plt
import math
#老虎机个数
number_of_bandits=10
number_of_buttons = 10
#老虎机的臂数
number_of_arms=10
#尝试数
number_of_pulls=10000
#eps
epsilon=0.3
#最小的decay
min_temp = 0.1
#衰减率
decay_rate=0.999

def pick_arm(q_values,counts,strategy,success,failure):
	global epsilon
	#随机返回一个臂
	if strategy=="random":
		return np.random.randint(0,len(q_values))
	#贪心算法,每次都收益最大的那个臂
	if strategy=="greedy":
		best_arms_value = np.max(q_values)
		#返回收益最大的臂的位置，并随机返回一个臂
		best_arms = np.argwhere(q_values==best_arms_value).flatten()
		return best_arms[np.random.randint(0,len(best_arms))]
	#加epsilon,egreedy中，epsilon不变，egreedy_decay，epsilon变化
	if strategy=="egreedy" or strategy=="egreedy_decay": 
		if  strategy=="egreedy_decay": 
			epsilon=max(epsilon*decay_rate,min_temp)
		if np.random.random() > epsilon:
			best_arms_value = np.max(q_values)
			best_arms = np.argwhere(q_values==best_arms_value).flatten()
			return best_arms[np.random.randint(0,len(best_arms))]
		else:
			return np.random.randint(0,len(q_values))
	#ucb,按照ucb公式，算每个臂的收益,取最大的收益的臂
	if strategy=="ucb":
		total_counts = np.sum(counts)
		q_values_ucb = q_values + np.sqrt(np.reciprocal(counts+0.001)*2*math.log(total_counts+1.0))
		best_arms_value = np.max(q_values_ucb)
		best_arms = np.argwhere(q_values_ucb==best_arms_value).flatten()
		return best_arms[np.random.randint(0,len(best_arms))]
	#thompson,利用beta分布选择臂
	if strategy=="thompson":
		sample_means = np.zeros(len(counts))
		for i in range(len(counts)):
			sample_means[i]=np.random.beta(success[i]+1,failure[i]+1)
		return np.argmax(sample_means)


fig = plt.figure()
ax = fig.add_subplot(111)
for st in ["greedy","random","egreedy","egreedy_decay","ucb","thompson"]:

	#定义数组：行-bandit按钮的个数，列-实验的次数
	best_arm_counts = np.zeros((number_of_buttons,number_of_pulls))

	#对于老虎机的每个按钮来说
	for i in range(number_of_buttons):
		#随机一个老虎机的按钮的收益w，保存最大收益
		arm_means = np.random.rand(number_of_buttons)
		best_arm = np.argmax(arm_means)
		#初始化按钮的收益
		q_values = np.zeros(number_of_buttons)
		#初始化按钮的按动次数
		counts = np.zeros(number_of_buttons)
		#初始化按钮成功的次数
		success=np.zeros(number_of_buttons)
		#初始化按钮失败的次数
		failure=np.zeros(number_of_buttons)
		
		#对于每次实验
		for j in range(number_of_pulls):
			#根据不同的策略，选择臂a
			a = pick_arm(q_values,counts,st,success,failure)
			
			#当前臂a的收益
			reward = np.random.binomial(1,arm_means[a])
			#臂的次数+1
			counts[a]+=1.0
			#更新当前臂的收益
			q_values[a]+= (reward - q_values[a])/counts[a]  ###???
			#记录成功的收益
			success[a]+=reward
			#记录失败的收益
			failure[a]+=(1-reward)
			#更新best_arm_counts[i][j]
			best_arm_counts[i][j] = counts[best_arm]*100.0/(j+1)
		epsilon=0.3

	#横纵坐标
	ys = np.mean(best_arm_counts,axis=0)
	xs = range(len(ys))
	ax.plot(xs, ys,label = st)

plt.xlabel('Steps')
plt.ylabel('Optimal pulls')

plt.tight_layout()
plt.legend()
plt.ylim((0,110))
plt.show()        



