import numpy as np

# generate a list w/ all the columns
cols = ['Time', 'Amount', 'Class']
for i in range(1,29): # 1 to 28
	colname = "V"+str(i)
	cols.append(colname)

# pick 5 random cols
chosencols = np.random.choice(cols, 5, replace=False)
print(chosencols)