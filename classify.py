from datetime import date
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def get_data(str1, str2):
	months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
	def get_tweets(str):
		dates = [] # format YYYY-MM-DD
		actions = 0
		def str_to_num(str):
			if ('K' in str):
				return 1000 * float(str[:-1])
			else:
				return float(str)
		with open(str, 'r') as file:
			for line in file:
				line = line.strip().split(' ')
				if (len(line) == 2): # Mo DD
					today = date.today()
					dates.append(date(today.year, months[line[0]], int(line[1])))
				elif (len(line) == 3): # DD Mo Yr
					dates.append(date(int(line[2]), months[line[1]], int(line[0])))
				elif (len(line) > 3): # reply, retweet, like
					actions = list(map(str_to_num, line))
		dates = np.array(dates, dtype='datetime64')
		actions = np.array(actions).reshape((int(len(actions)/3), 3))
		return dates, actions
	def get_prices(str):
		dates = []
		prices = []
		with open(str, 'r') as file:
			file.readline()
			for line in file:
				line = line.strip().split(',')
				dates.append(line[0])
				for i in range(1, 7): # open, hi, low, close, adj, vol
					prices.append(line[i])
		dates = np.array(dates, dtype='datetime64')
		prices = np.array(prices).reshape((int(len(prices)/6)), 6)
		return dates, prices

	tweets = get_tweets(str1)
	prices = get_prices(str2)

	data = prices[1]
	tweet_data = np.zeros((data.shape[0], 3))

	for i, tweet_date in enumerate(tweets[0]):
		for j, market_date in enumerate(prices[0]):
			if (tweet_date == market_date):
				tweet_data[j] += tweets[1][i]
				break

	data = np.hstack((data, tweet_data))

	dates = prices[0]
	prices = 0
	tweets = 0
	tweet_data = 0

	 # 9 dimensions: op, hi, lo, close, adj, vol, replies, retweets, likes
	data = np.reshape(data, (int(data.shape[0]/5), 5, 9)) # separated into market week statistics
	data = np.asarray(data, dtype='d')
	after = data[1:, 4, 4].flatten()
	before = data[:-1, 4, 4].flatten()
	difference = np.subtract(after, before)
	labels = np.clip(np.sign(difference), 0, 1) # weekly adjusted closing price difference. 1: rise, -1: fall

	# normalize training data
	for i in range(0, data.shape[0]):
		for j in range(0, 9):
			week_norm = np.linalg.norm(data[i, :, j])
			if week_norm == 0:
				week_norm = 1
			for k in range(0, 5):
				data[i, k, j] = data[i, k, j] / week_norm

	return dates, data, labels

dnkn = get_data('dnkn.txt', 'DNKN.csv') # dunkin' data

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras import regularizers

model = Sequential()
# filter 4, kernel size 1, input dim 9 = 4 x (9 weights + 1 bias)
model.add(Conv1D(4, 1, input_shape=(5, 9), activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid', bias_regularizer=regularizers.l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer='nadam')

model.fit(dnkn[1][:-1], dnkn[2], batch_size=16, epochs=512)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

dates, data, labels = get_data('sbux.txt', 'SBUX.csv') # starbucks data
predicted = model.predict(data[:-1])

# plot setup
days = mdates.DayLocator()
months = mdates.MonthLocator()
date_format = mdates.DateFormatter('%Y-%m-%d')

fig, ax = plt.subplots()

week_dates = np.array(np.reshape(dates[:len(dates)-(len(dates)%5)], (int(len(dates)/5), 5))[:, 4]).flatten()

ax.plot(week_dates[1:], np.clip(labels, 0.49, 0.51)) # plot truth

# ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_minor_locator(days)

# bound years
date_min = np.datetime64(dates[0])
date_max = np.datetime64(dates[-1])
ax.set_xlim(date_min, date_max)

def price(x):
	return ('$%1.2f' % x)

ax.format__xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

fig.autofmt_xdate()

# plot predicted volume
predict_dates = []
for i, d in enumerate(dates):
	if (i%7 == 0):
		predict_dates.append(d)
ax.scatter(week_dates[1:], np.clip(predicted, 0.49, 0.51), c='r') # 0.49, 0.51
plt.show()

# augmented results: 51/61
# unaugmented results: 34/61
# just tweets: 49/61