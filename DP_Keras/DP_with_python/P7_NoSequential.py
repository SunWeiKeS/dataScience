# In[7-1]: P201 用函数式API实现双输入问答模型
'''
Sequential 模型，默认网络只有一个输入和一个输入，
且网络是层的线性堆叠
'''
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# 文本输入是一个长度可变整数序列，可以对其进行命名
text_input = Input(shape=(None,), dtype='int32', name='text')

# 将输入嵌入长度为64的向量
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)

# 利用LSTM将向量编码为单个向量
encoded_text = layers.LSTM(32)(embedded_text)

# 对问题进行相同的处理 //使用不同层的实例
question_input = Input(shape=(None,), dtype='int32', name='question')

embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 将编码后的问题和文本连接起来
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

# 在上面添加一个softmax分类器
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# In[7-2]: P202 将数据输入到多输入模型中
import numpy as np
import keras

num_samples = 1000
max_length = 100

# 生成虚拟的Numpy数据
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
# 回答是one hot编码不是整数
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

# 使用输入组成的列表来拟合
model.fit([text, question], answers, epochs=10, batch_size=128)

# 使用输入组成的字典来拟合
# model.fit({'text':text,'question':question},answers,epochs=100,batch_size=128)


# In[7-3]: 用函数式API实现一个三输出模型

from keras.models import Model
from keras import layers
from keras import Input

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 输出层都具有名称
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

# In[7-4]: 多输出模型的编译选项：多重损失
'''
训练这种模型需要能够对网络的各个头指定不同的损失函数，
例如年龄预测是标量回归任务、性别预测是二分类任务，两者需要不同的训练过程，
但是梯度下降要求将一个标量最小化，所以为了能够训练模型，我们必须将这些损失合并为单个标量
合并不同损失最简单的方法就是对所有损失求和
'''
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
'''
model.compile(optimizer='rmsprop',loss={'age':'mse',
										'income':'categorical_crossentropy',
										'gender':'binary_crossentropy'})
'''
# In[7-5]: 多输出模型的编译选项：损失加权
'''
严重不平衡的损失贡献会导致模型表示针对单个损失值最大的任务进行优化，而不考虑其他任务优化
当损失值具有不同的取值范围，我们应该为每个损失值分配不同大小的重要性
'''
model.compile(optimizer='rmsprop',
			  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
			  loss_weights=[0.25, 1., 10.])
'''
model.compile(optimizer='rmsprop',
			  loss={'age': 'mse',
					'income': 'categorical_crossentropy',
					'gender': 'binary_crossentropy'},
			  loss_weights={'age': 0.25,
							'income': 1.,
							'gender': 10.})
'''

# In[7-6]: 将数据输入到多输出模型中
'''
# 假设age_targets,income_targets,gender_targets是numpy数组
model.fit(posts,[age_targets,income_targets,gender_targets],
	epochs=10,batch_size=64)

model.fit(posts,
	{'age':age_targets,
	'income':income_targets,
	'gender':gender_targets},
	epochs=10,batch_size=64)
'''
