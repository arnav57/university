import openpyxl as pxl
import matplotlib.pyplot as plt
from math import sqrt

workbook = pxl.load_workbook('Lab_1_3333.xlsx')
sheet = workbook.active

# acquire If, V, I
If = []
V = [80.3, 90.7, 100.7, 110.4, 120, 129.8]
I = []
# If
max = 6
for i, cell in enumerate(sheet[2]):
    if i == 0:
        continue
    if i > max:
        break
    else:
        If.append(cell.value)
# I
max = 6
for i, cell in enumerate(sheet[12]):
    if i == 0:
        continue
    if i > max:
        break
    else:
        I.append(cell.value)

# calc pf = Pin / sqrt(3) * V_ll * I_l
vll = []
il = []
p1 = [231.0,	212.9,	194.4,	175.7,	155.5,	136.8,	118.3,	98.0,	79.4,	57.4,	37.2]
p2 = [226.3,	207.6,	187.4,	167.3,	148.7,	130.2,	111.5,	91.3,	72.6,	52.4,	33.8]
pin = []
pf = []

for i in range(1,12):
    vll.append(sheet[16][i].value)
    il.append(sheet[18][i].value)

for i in range(len(p1)):
    pin.append(p1[i] + p2[i])
    pf.append(pin[i] / (sqrt(3) * vll[i] * il[i] ))

# create plots
plt.plot(V, If, 'ro')
plt.plot(V, If, 'b')
plt.title("V vs If")
plt.ylabel('V [V]')
plt.xlabel('If [A]')
plt.savefig('V vs If.png')

plt.clf()

plt.plot(I, If, 'ro')
plt.plot(I, If, 'b')
plt.title("I vs If")
plt.ylabel('I [A]')
plt.xlabel('If')
plt.savefig('I vs If.png')

plt.clf()

plt.plot(V, If, 'ro')
plt.plot(V, If, 'b')
plt.title("V vs If")
plt.ylabel('V [V]')
plt.xlabel('If [A]')
plt.savefig('V vs If.png')

plt.clf()
if2 = [0.35,	0.4,	0.45,	0.5,	0.55,	0.6,	0.65,	0.7,	0.75,	0.8,	0.85]

plt.plot(il, if2, 'ro')
plt.plot(il, if2, 'b')
plt.title("I vs If (Synchronization)")
plt.ylabel('I [A]')
plt.xlabel('If [A]')
plt.savefig('I vs If sync.png')

plt.clf()

plt.plot(pf, if2, 'ro')
plt.plot(pf, if2, 'b')
plt.title("pf vs If (Synchronization)")
plt.ylabel('pf')
plt.xlabel('If [A]')
plt.savefig('pf vs If sync.png')