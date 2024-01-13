import scipy.signal as signal
import matplotlib.pyplot as plt

# define system
TYPE = 'tf' # 'zpk' or 'tf'
zeroes, poles, gain = [-3, -5], [0, -2, -4], [8/150]
num, den = [1,5], [1,4,16,0,0]

if TYPE == 'zpk':
    sys = signal.ZerosPolesGain(zeroes,poles,gain)
if TYPE == 'tf':
    sys = signal.TransferFunction(num,den)

w, mag, phase = signal.bode(sys)

plt.figure()
# Bode Mag Plot
plt.semilogx(w, mag, 'b')
plt.title('Bode Magnitude Plot')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Gain (dB)')
plt.grid(True,'both')


plt.figure()
# Bode Phase Plot
plt.semilogx(w,phase, 'r')
plt.title('Bode Phase Plot')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase Shift (deg)')
plt.grid(True,'both')

plt.show()

