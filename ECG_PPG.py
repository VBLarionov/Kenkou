import numpy as np
import scipy.signal as ss
import scipy.interpolate as si
import matplotlib.pyplot as plt
import pandas as pd

ppg_data = np.load('second_ppg.npy')
ecg_data = np.load('second_ecg.npy')


# functions ---------------------------------------------------------------------------------------

# calculate median
# better option: statistics.median - O(N) complexity instead of O(NlogN)
def CalculateMedian(array):
    array.sort()
    return array[array.size//2]


# PPG processing

# find PPG isoline points (points of waves start)
def FindPPGPeaks(ppg_data):
    
    # approx PPG peaks - calculate smoothed 1st derivative to amplify waves onsets and find its peaks
    ppg_der = 100*ss.savgol_filter(ppg_data, 3, 2, 1)  # using Savitzky-Golay filter
    ppg_peaks, _ = ss.find_peaks(ppg_der, distance = 20)

    # PPG isoline points - move from each found peak to the left while 1st derivative is positive
    ppg_isoline = ppg_peaks.copy()
    for n in range(0,ppg_peaks.size):
        pos = ppg_peaks[n]
        while pos > 0 and ppg_der[pos] > 0:
            pos -= 1
        ppg_isoline[n] = pos + 1

    # some additional correction
    for n in range(0,ppg_isoline.size):
        if ppg_data[ppg_isoline[n]-1] < ppg_data[ppg_isoline[n]]:
            ppg_isoline[n] -= 1

    return ppg_isoline

# mark bad PPG waves,
# returns array with flags, if nth wave is bad
# plus also return additional parameters: median waves amplitude (needed for output)
def MarkBadPPGWaves(ppg_data, ppg_isoline):
    # calculate parameters for each PPG-wave
    ppg_waves_slope = np.zeros(ppg_isoline.size, dtype = int)   # time from start to half amplitude
    ppg_waves_width = np.zeros(ppg_isoline.size, dtype = int)   # PPG wave width (RR)
    ppg_waves_max = np.zeros(ppg_isoline.size)                  # PPG wave max value
    ppg_waves_min = np.zeros(ppg_isoline.size)                  # PPG wave min value
    for n in range(0,ppg_isoline.size):
        if n < ppg_isoline.size - 1:
            ppg_waves_width[n] = ppg_isoline[n + 1] - ppg_isoline[n]
            a = ppg_data[ppg_isoline[n]:ppg_isoline[n+1]]
            ppg_waves_max[n] = np.max(ppg_data[ppg_isoline[n]:ppg_isoline[n+1]])
            ppg_waves_min[n] = np.min(ppg_data[ppg_isoline[n]:ppg_isoline[n+1]])
            ppg_waves_slope[n] = ppg_isoline[n]
            while ppg_waves_slope[n] < ppg_isoline[n + 1] and ppg_data[ppg_waves_slope[n]] < ppg_waves_max[n] / 2:
                ppg_waves_slope[n] += 1
            ppg_waves_slope[n] = ppg_waves_slope[n] - ppg_isoline[n]

    # calculate medians of parameters
    ppg_waves_slope_med = CalculateMedian(ppg_waves_slope.copy())
    ppg_waves_width_med = CalculateMedian(ppg_waves_width.copy())
    ppg_waves_max_med = CalculateMedian(ppg_waves_max.copy())

    # calculate smoothed RR-trend (rolling window median filter)
    pd_arr = pd.DataFrame(ppg_waves_width)
    pd_arr = pd_arr.rolling(window = 5).median()
    ppg_waves_width_rmed = pd_arr.to_numpy().reshape(-1)
    ppg_waves_width_rmed[np.isnan(ppg_waves_width_rmed)] = 0

    # find bad waves
    ppg_waves_bad = np.zeros(ppg_isoline.size)
    for n in range(0,ppg_isoline.size):
        if (ppg_waves_max[n] > ppg_waves_max_med * 2 or ppg_waves_max[n] < ppg_waves_max_med * 0.2 or
            ppg_waves_min[n] < -ppg_waves_max_med * 0.1 or
            ppg_waves_slope[n] > ppg_waves_slope_med * 3 or
            ppg_waves_width[n] > 1.8 * ppg_waves_width_rmed[n] or ppg_waves_width[n] < 0.5 * ppg_waves_width_rmed[n]):
            ppg_waves_bad[n] = 1

    # close wholes of 1 or 2 good between bad
    last_bad = -1
    for n in range(0,ppg_isoline.size-1):
        if ppg_waves_bad[n] > 0:
            last_bad = n
        if ppg_waves_bad[n+1] == 1 and ppg_waves_bad[n] == 0 and n + 1 - last_bad <= 3:
            ppg_waves_bad[n] = 1
            if n > 1:
                ppg_waves_bad[n - 1] = 1        

    return ppg_waves_bad, ppg_waves_max_med  # is nth wave is bad, median amplitude


# ECG processing


# ECG peak detector: calculate ECG peaks and isoline points
def FindECGPeaks(ecg_data):
    
    # using rolling window variance of smoothed first derivative
    # windows width set up corresponding to width of QRS complex
    # such procedure will effectively amplify QRS complexes and supress P,T-waves
    ecg_der = ss.savgol_filter(ecg_data, 25, 4, 1)  # Savitskiy-Golay smoothed 1st derivative
    #ecg_2der = ss.savgol_filter(ecg_data, 25, 4, 2)
    pd_arr = pd.DataFrame(ecg_der)
    pd_arr = pd_arr.rolling(window = 100).std()  # rolling window std
    ecg_crit = pd_arr.to_numpy().reshape(-1)
    ecg_crit[np.isnan(ecg_crit)] = 0
    ecg_crit = np.square(ecg_crit)  # std to variance
    ecg_peaks, _ = ss.find_peaks(ecg_crit, distance = 600)
    peaks_values = np.zeros(ecg_peaks.size)

    # specify peaks: calculate adaptive threshold using already found peaks and using it recalculate new
    for n in range(0,ecg_peaks.size):
        peaks_values[n] = ecg_crit[ecg_peaks[n]]
    # better option: statistics.median - O(N) complexity instead of O(NlogN)
    peaks_values.sort()
    peaks_level = peaks_values[ecg_peaks.size//2]
    ecg_peaks, _ = ss.find_peaks(ecg_crit, distance = 600, height = [peaks_level / 4, peaks_level * 4])
    peaks_values = np.zeros(ecg_peaks.size)

    # ECG isoline points between P and QRS waves: just move left while variance is high
    ecg_isoline = ecg_peaks.copy()
    for n in range(0,ecg_peaks.size):
        pos = ecg_peaks[n]
        max_pos = pos
        max_val = ecg_crit[max_pos]
        while pos > 0 and pos > max_pos - 150 and (ecg_crit[pos] > max_val/120) and not(ecg_crit[pos] < max_val/30 and ecg_crit[pos] > ecg_crit[pos - 5] * 0.9):
            pos -= 1
        ecg_isoline[n] = max(0, pos - 20)

    return ecg_peaks, ecg_isoline   # return found peaks and isoline points


# calculate exact positions of R-peaks
def SpecifyECGRPeaks(ecg_data, ecg_isoline):
    # QRS max & min for each complex
    QRS_max = np.zeros(ecg_isoline.size)
    QRS_min = np.zeros(ecg_isoline.size)
    for n in range(0,ecg_isoline.size):
        QRS_max[n] = np.max(ecg_data[ecg_isoline[n]:ecg_isoline[n]+200])
        QRS_min[n] = np.min(ecg_data[ecg_isoline[n]:ecg_isoline[n]+200])

    # calculate medians
    # better option: statistics.median - O(N) complexity instead of O(NlogN)
    QRS_max.sort()
    QRS_max_total = QRS_max[ecg_isoline.size//2]
    QRS_min.sort()
    QRS_min_total = QRS_min[ecg_isoline.size//2]


    # specify R-peaks using maximums or minimums
    Rpeaks = np.zeros(ecg_isoline.size, dtype = int)
    for n in range(0,ecg_isoline.size):
        QRS = ecg_data[ecg_isoline[n]:ecg_isoline[n]+200]
        if abs(QRS_max_total) > abs(QRS_min_total):
            Rpeaks[n] = ecg_isoline[n] + QRS.argmax()
        else:
            Rpeaks[n] = ecg_isoline[n] + QRS.argmin()

    return Rpeaks  # return array with exact R-peaks positions




# main code ---------------------------------------------------------------------------------------



# lowpass filter PPG using 4th order Butterworth filter
b,a = ss.butter(N=4, Wn=10, fs=30, btype='lowpass')
ppg_data = ss.lfilter(b, a, ppg_data)

# calculate PPG isoline points
ppg_isoline = FindPPGPeaks(ppg_data)

# highpass PPG filter: construct smooth spline using isoline points, substract it
# this kind of filter doesn't affect useful PPG signal (which is rather low freq itself)
ppg_spline = si.CubicSpline(ppg_isoline, ppg_data[ppg_isoline])
ppg_trend = ppg_spline(np.linspace(0, ppg_data.size, ppg_data.size))
ppg_data = ppg_data - ppg_trend

# mark bad PPG-waves
ppg_waves_bad, MaxTh = MarkBadPPGWaves(ppg_data, ppg_isoline)

# convert flags of bad waves to signal channel (for output)
ppg_waves_bad_sig = np.zeros(ppg_data.size)
for n in range(0,ppg_waves_bad.size-1):
    if ppg_waves_bad[n] > 0:
        ppg_waves_bad_sig[ppg_isoline[n]:ppg_isoline[n+1]] = 1

# ECG low-pass filtering
# using usual 4th order lowpass Butterworth filter with cutoff freq = 20 Hz
# (it cuts a little QRS-peaks, but in this task it is not critical)
# (better option - adaptive wavelet shrinkage based LP filter
b,a = ss.butter(N=4, Wn=20, fs=1000, btype='lowpass')
ecg_data = ss.lfilter(b, a, ecg_data)

ecg_peaks, ecg_isoline = FindECGPeaks(ecg_data)

# ECG highpass filter
# adaptive filter - constructing 3rd order spline using isoline points
# this filter doesn't affect useful ECG signal (especially ST-segment)
ecg_spline = si.CubicSpline(ecg_isoline, ecg_data[ecg_isoline])
ecg_trend = ecg_spline(np.linspace(0, ecg_data.size, ecg_data.size))
ecg_trend[0:ecg_isoline[1]] = ecg_trend[ecg_isoline[1]]
ecg_trend[ecg_isoline[-1]:ecg_trend.size] = ecg_trend[ecg_isoline[-1]]
ecg_data = ecg_data - ecg_trend

# calculate exact positions of R-peaks
Rpeaks = SpecifyECGRPeaks(ecg_data, ecg_isoline)


# Synchronize two signals
RR_ECG = np.zeros(ecg_peaks.size, dtype = int)  # ECG peaks in terms of PPG sample rate
for n in range(0,ecg_peaks.size):
    dec_pos = Rpeaks[n] / 1000 * 30
    RR_ECG[n] = round(dec_pos)

RR_PPG = np.zeros(ppg_isoline.size, dtype = int)  # only good PPG peaks
good_ppg_peaks = 0
for n in range(0,ppg_isoline.size):
    if ppg_waves_bad[n] == 0:
        RR_PPG[good_ppg_peaks] = ppg_isoline[n]
        good_ppg_peaks += 1
RR_PPG = RR_PPG[0:good_ppg_peaks]

# find best superimposition
min_loss = ecg_data.size  # minnimal error values
min_shift = 0  # remembered best shift (in terms of PPG channel)
for shift in range(0, round(ecg_data.size / 1000 * 30) - ppg_data.size):
    loss = 0
    ecg_peak = 0
    for ppg_peak in range(5, RR_PPG.size):
        # current PPG peak
        curr_ppg_peak = shift + RR_PPG[ppg_peak]
        # find closest ECG peak
        while ecg_peak < RR_ECG.size - 1 and RR_ECG[ecg_peak+1] < curr_ppg_peak:
            ecg_peak += 1
        if ecg_peak >= RR_ECG.size-1:
            loss += min(abs(RR_ECG[ecg_peak] - curr_ppg_peak), abs(RR_ECG[ecg_peak-1] - curr_ppg_peak))
        else:
            loss += min(abs(RR_ECG[ecg_peak+1] - curr_ppg_peak), abs(RR_ECG[ecg_peak] - curr_ppg_peak))

    if loss < min_loss:
        min_loss = loss
        min_shift = shift

# calculate part of bad PPG waves
BadPPGWaves = 0
for n in range(0,ppg_isoline.size):
    if ppg_waves_bad[n] == 1:
        BadPPGWaves += 1
QualityIndex = BadPPGWaves / ppg_isoline.size

# output processed ECG signal with peaks
plt.rcParams['figure.figsize'] = [50, 25]
plt.plot(ecg_data, label = 'ECG')
plt.plot(Rpeaks, ecg_data[Rpeaks], "xr", label = 'R-peaks')
leg = plt.legend()

# cut possibly huge values of PPG in the begining
first_good_wave = 1
while first_good_wave < 5 and ppg_waves_bad[first_good_wave] == 1:
    first_good_wave = first_good_wave + 1
BadPos = ppg_isoline[first_good_wave + 1]
while BadPos > 0 and ppg_data[BadPos] < MaxTh * 4 and ppg_data[BadPos] > -MaxTh:
    BadPos -= 1

if BadPos > 1:
    ppg_data[0:BadPos] = 0

# output processed PPG signal with isoline points
fig = plt.figure()
#plt.set_title('Percent of good waves = ' + str(int(QualityIndex * 100)), fontsize = 22)
plt.plot(ppg_data, label='PPG')
plt.plot(ppg_isoline, ppg_data[ppg_isoline], "xr", label = 'PPG waves start')
plt.plot(ppg_waves_bad_sig * MaxTh, label='Bad waves: '+ str(int(QualityIndex * 100)) + '%')
leg = plt.legend()

# prepare overlaid signals
#ppg_data[0:100] = 0
ppg_isoline = min_shift + ppg_isoline
ppg_isoline = np.round(ppg_isoline * 1000 / 30)
ppg_isolinei = ppg_isoline.astype(int)
min_shift = round(min_shift * 1000 / 30)
PPg_new = 300*ss.resample(ppg_data, (int)(ppg_data.size * 1000/30))
PPg_new1 = ecg_data.copy()
PPg_new1[0:min_shift] = 0
PPg_new1[min_shift:min_shift+PPg_new.size] = PPg_new
PPg_new1[min_shift+PPg_new.size:PPg_new1.size] = 0

# output overlaid signals
fig = plt.figure()
plt.plot(ecg_data, label = 'ECG')
plt.plot(PPg_new1, label = 'PPG')
plt.plot(Rpeaks, ecg_data[Rpeaks], "xr", label = 'R-peaks')
plt.plot(ppg_isolinei, PPg_new1[ppg_isolinei], "og", label = 'PPG waves start')
leg = plt.legend()

plt.show() 

i = 0
