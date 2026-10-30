# Acquisition
FS = 1000  # sampling frequency (Hz)

# Accelerometer
ACC_COLUMNS_HUB = ["x", "y", "z"]
ACC_COLUMNS_DHD = ["gACCx", "gACCy", "gACCz"]
ACC_COLUMNS = ACC_COLUMNS_HUB + ACC_COLUMNS_DHD

# Force / displacement
FORCE_COLUMN = "FORCE"
DISPLACEMENT_COLUMN = "DISPLACEMENT"

# EMG
EMG_Extensor = "EMG_Extensor"
EMG_Flexor = "EMG_Flexor"
EMG_Extensor_filt = "EMG_Extensor_filt"
EMG_Flexor_filt = "EMG_Flexor_filt"
EMG_CHANNELS = [EMG_Extensor, EMG_Flexor]