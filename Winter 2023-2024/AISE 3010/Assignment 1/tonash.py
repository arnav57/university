empID = input('Please enter your employee ID:')
option = input('''Please enter the order option:
               1) 600 tix for $100
               2) 250 tix for $50\n''')

# cast option to int
option = int(option)

# emptId nad option arae str bny efault
if (option not in [1,2,3,4]):
    print('Please enter a valid option number')
else:
    print('You have selected option: ', option)

# number of orders per option
option1 = 0
option2 = 0

if (option == 1):
    option1 = option1 + 1
elif (option == 2):
    option2 = option2 + 1

# calculate rev per option
r01 = 60 * option1 

# report part
print('There are ', option1, ' orders for option 1 - total revenue: ', r01)
print('There are ', option2, ' orders for option 2')