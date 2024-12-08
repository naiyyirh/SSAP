import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt  

matplotlib.use("Agg")

# list of notes and their corresponding frequencies
notes_to_frequencies = [
    ("C0", 16.35), ("D0", 18.35), ("E0", 20.60), ("F0", 21.83), ("G0", 24.50), ("A0", 27.50), ("B0", 30.87),
    ("C1", 32.70), ("D1", 36.71), ("E1", 41.20), ("F1", 43.65), ("G1", 49.00), ("A1", 55.00), ("B1", 61.74),
    ("C2", 65.41), ("D2", 73.42), ("E2", 82.41), ("F2", 87.31), ("G2", 98.00), ("A2", 110.00), ("B2", 123.47),
    ("C3", 130.81), ("D3", 146.83), ("E3", 164.81), ("F3", 174.61), ("G3", 196.00), ("A3", 220.00), ("B3", 246.94),
    ("C4", 261.63), ("D4", 293.66), ("E4", 329.63), ("F4", 349.23), ("G4", 392.00), ("A4", 440.00), ("B4", 493.88),
    ("C5", 523.25), ("D5", 587.33), ("E5", 659.26), ("F5", 698.46), ("G5", 783.99), ("A5", 880.00), ("B5", 987.77),
    ("C6", 1046.50), ("D6", 1174.66), ("E6", 1318.51), ("F6", 1396.91), ("G6", 1567.98), ("A6", 1760.00), ("B6", 1975.53),
    ("C7", 2093.00)
]

# converting the list into a NumPy array of the frequencies for easier computation
freq_array = np.array([freq for _, freq in notes_to_frequencies])

# function to find the closest note a frequency is to (specifically the index of that note in notes_to_frequencies)
def closest_note_index(freq):
    # notes_array - freq: subtracts the pitch frequency from each frequency in freq_array, giving an array of differences between freq and each note's frequency
    # np.argmin(): returns the index of the smallest value in the array (so the index where the difference between the note's frequency and the pitch frequency is the smallest)
    return np.argmin(np.abs(freq_array - freq))

# function to segment the array of pitches into intervals where the note changes
def segment_pitches(pitch_array):
    # empty list to store the intervals
    intervals = []

    # getting the index of the closest note in notes_to_frequencies
    current_note_index = closest_note_index(pitch_array[0])

    # the index of the beginning of the 1st interval is 0
    start_index = 0
    
    # iterating through the pitch array starting from the 2nd pitch
    for i in range(1, len(pitch_array)):
        # getting the index of the closest note in notes_to_frequencies for the current pitch
        note_index = closest_note_index(pitch_array[i])
        # checking if the note has changed (if the note index is different)
        if note_index != current_note_index:  
            # appending the interval from the start_index
            # to the previous index, including the note
            intervals.append((start_index, i - 1, notes_to_frequencies[current_note_index][0]))
            # changing the current note index to the index of the new note
            current_note_index = note_index
            # changing the start index to the current index
            start_index = i
    
    # appending the last interval
    intervals.append((start_index, len(pitch_array) - 1, notes_to_frequencies[current_note_index][0]))
    
    return intervals

# function to calculate the middle 50% and its median for each note
def middle_50_median(pitches, intervals):
    # initializing an empty list to store the median of the middle 50% for each interval
    medians = []
    
    # looping through each interval 
    for interval in intervals:
        # unpacking the interval tuple into the start index, end index, and note 
        start, end, note = interval  
        
        # getting the values from the pitch array corresponding to this interval
        values = pitches[start:end+1]
        
        # sorting the values in ascending order
        sorted_values = np.sort(values)
        
        # the total number of values in the current interval
        n = len(sorted_values)
        
        # finding the index for the 25th percentile (start of the middle 50%)
        lower_idx = int(0.25 * n)
        
        # finding the index for the 75th percentile (end of the middle 50%)
        upper_idx = int(0.75 * n)
        if upper_idx == 0:
            upper_idx += 1
        
        # extracting the middle 50% of the sorted values (from the 25th to 75th percentile)
        middle_50 = sorted_values[lower_idx:upper_idx]
        
        # the median of the middle 50% values
        median_of_middle_50 = np.median(middle_50)
        # appending the median of the middle 50% to the medians list
        medians.append(median_of_middle_50)
    
    # returning the list medians for the middle 50% of each interval
    return medians

# function to perform the cents calculation
def calculate_cents(user_pitches, reference_pitches):
    # performing the calculation: 1200 * log2(user_pitches / reference_pitches)
    result = 1200 * np.log2(np.array(np.maximum(user_pitches, reference_pitches)) / np.array(np.minimum(user_pitches, reference_pitches)))
    return result

# function to analyze pitch in an audio file
def analyze_pitch(file_path, is_user, sr=None):
    # loading the audio file with librosa
    audio_data = librosa.load(file_path, sr=44100)[0]

    # finding the intervals where there is sound
    sound_intervals = librosa.effects.split(audio_data, top_db=20)
    # checking if there is exactly 1 interval
    if len(sound_intervals) != 1:
        return None, None, None, "You need exactly 1 sound."

    # extracting pitches and magnitudes using librosa.pyin
    pitches = librosa.pyin(
        audio_data, 
        # maximum frequency for pitch detection
        fmin=librosa.note_to_hz('C2'),  
        # minimum frequency for pitch detection
        fmax=librosa.note_to_hz('C7'),  
        # sampling rate
        sr=44100  
    )[0]

    # replacing NaN values in pitches with 0
    pitches = np.array([0 if np.isnan(x) else x for x in pitches])

    # performing short-time Fourier transform on the audio data
    stft_result = librosa.stft(audio_data, n_fft=2048, hop_length=512)

    # getting the magnitude of the STFT result
    magnitudes = np.abs(stft_result)

    # transposing the magnitudes array for easier manipulation
    magnitudes = list(np.transpose(magnitudes))

    # converting the magnitude values to decibels
    decibels = np.array([librosa.amplitude_to_db(x) for x in magnitudes])

    # taking the maximum decibel value for each frame
    decibels = [max(x) for x in decibels]

    # applying a threshold to filter out low-magnitude signals
    pitches = np.array([0 if decibels[i] < 5 else pitches[i] for i in range(len(pitches))])

    # generating times corresponding to pitches
    times = librosa.times_like(pitches, sr=44100)

    # if analyzing the user audio
    if is_user: 

        # creating a figure 
        plt.figure(figsize=(10, 6))

        # plotting the times (x-axis) vs pitches (y-axis) and labeling the line as "F0 (Hz)"
        plt.plot(times, pitches, label="F0 (Hz)")

        # labeling the x-axis as "Time (s)"
        plt.xlabel("Time (s)")

        # labeling the y-axis as "Frequency (Hz)"
        plt.ylabel("Frequency (Hz)")

        # setting the title of the plot to "User Pitch (F0) Over Time"
        plt.title("Pitch Contour (F0) Over Time")

        # displaying the legend, showing the label for the line
        plt.legend()

        # adding a grid to the plot for easier reading of values
        plt.grid()

        # saving the figure to a file 
        plt.savefig(file_path[:-4] + "_F0_graph.png")

        # closing the figure to free up memory
        plt.close()

    return times, pitches, decibels, None

def analyze_four_pitches(file_path, is_user, sr=44100, top_db=20):
    # loading the audio file with librosa
    audio_data, sr = librosa.load(file_path, sr=sr)

    # analyzing the pitches from the audio clip as a whole

    # extracting pitches and magnitudes using librosa.pyin
    all_pitches = librosa.pyin(
        audio_data, 
        # maximum frequency for pitch detection
        fmin=librosa.note_to_hz('C2'),  
        # minimum frequency for pitch detection
        fmax=librosa.note_to_hz('C7'),  
        # sampling rate
        sr=44100  
    )[0]

    # generating times corresponding to pitches
    all_times = librosa.times_like(all_pitches, sr=44100)
    
    # finding the intervals where there is sound as a list of tuples
    sound_intervals = librosa.effects.split(audio_data, top_db=top_db)

    # checking if there are exactly 4 intervals
    if len(sound_intervals) != 4:
        return "You need exactly 4 clearly separated sounds.", None, None, None

    # list to store pitches for each seggment
    results = []
    
    # analyzing each sound interval
    for interval in sound_intervals:
        # the starting and ending index of each interval
        start_idx, end_idx = interval
        # extracting the segment
        interval = None
        if end_idx == len(audio_data) - 1:
            interval = audio_data[start_idx:]
        else:
            interval = audio_data[start_idx:end_idx+1]
        
        # extracting pitches using pYIN
        pitches = librosa.pyin(
            interval, 
            fmin=librosa.note_to_hz('C2'),  
            fmax=librosa.note_to_hz('C7'),  
            sr=sr
        )[0]
        
        # replacing NaN values in pitches with 0
        pitches = np.array([0 if np.isnan(x) else x for x in pitches])

        # performing short-time Fourier transform on the segment
        stft_result = librosa.stft(interval, n_fft=2048, hop_length=512)

        # getting the magnitude of the STFT result
        magnitudes = np.abs(stft_result)

        # converting the magnitude values to decibels
        decibels = np.array([librosa.amplitude_to_db(x) for x in magnitudes])
        
        # taking the maximum decibel value for each frame
        max_decibels = [max(x) for x in decibels]

        # applying a threshold to filter out low-magnitude signals
        filtered_pitches = np.array([0 if max_decibels[i] < 5 else pitches[i] for i in range(len(pitches))])

        # generating times corresponding to pitches
        times = librosa.times_like(pitches, sr=sr)
        
        # adding the pitches of the segment to results
        results.append(filtered_pitches)

    #if analyzing the user audio
    if is_user:

        ###################################################################################################################

        # plotting F0 over time with breaks where F0 is 0

        # setting the figure size for the plot
        plt.figure(figsize=(10, 5))  
        # plotting with steps to show breaks at F0=0
        plt.plot(all_times, all_pitches, drawstyle='steps-post')  

        # setting labels for the x and y axes
        # labeling the x-axis as "Time (seconds)"
        plt.xlabel("Time (seconds)")  
        # labeling the y-axis as "F0 (Hz)"
        plt.ylabel("F0 (Hz)")  
        # title of the plot
        plt.title("Fundamental Frequency (F0) over Time")  

        # displaying a legend for each interval
        plt.legend()
        # adding a grid to make the plot easier to read  
        plt.grid()  

        # saving the plot to a PNG file 
        plt.savefig(file_path[:-4] + "_F0_graph.png") 
        # closing the plot to free up memory 
        plt.close()  

        ###################################################################################################################

        ###################################################################################################################

        # calculating and plot the amplitude envelope

        # performing short-time Fourier transform on the audio data to get the magnitudes
        stft_result = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitudes = np.abs(stft_result)
        # getting the largest magnitude in time frame
        max_magnitudes = np.max(magnitudes, axis=0)
        # getting the largest magnitude
        max_magnitude = np.max(max_magnitudes)
        # creating a new array by dividing each element by the largest magnitude (so that the amplitude ranges from 0 to 1)
        amplitudes = max_magnitudes / max_magnitude
        # generating times corresponding to magnitudes
        all_amp_times = librosa.times_like(stft_result, sr=44100)

        plt.figure(figsize=(10, 5))
        plt.plot(all_amp_times, amplitudes, label='Amplitude Envelope', color='orange', linewidth=0.5)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Amplitude Envelope over Time")
        plt.legend()
        plt.grid()
        plt.savefig(file_path[:-4] + "_Amplitude_Envelope.png")
        plt.close()

        ###################################################################################################################

        ###################################################################################################################

        # calculating the first derivative of the amplitude envelope
        amplitude_derivative = np.diff(amplitudes)
        # creating the corresponding time array for the derivative (the length would be 1 less because it's the change in values)
        times_derivative = all_amp_times[:-1] 

        plt.figure(figsize=(10, 5))
        plt.plot(times_derivative, amplitude_derivative, label='1st Derivative of Amplitude Envelope', color='green', linewidth=0.5)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude Envelope Derivative")
        plt.title("1st Derivative of Amplitude Envelope over Time")
        plt.legend()
        plt.grid()
        plt.savefig(file_path[:-4] + "_Amplitude_Envelope_Derivative.png")
        plt.close()

        ###################################################################################################################

    return None, results, all_times, all_pitches

# function to calculate accuracy between reference and user pitches
def calculate_accuracy(reference_pitches, user_pitches):
    # identifying indices where the reference pitches are non-silent
    non_silent_indices_ref = np.where(reference_pitches > 0)
    # identifying indices where the user pitches are non-silent
    non_silent_indices_user = np.where(user_pitches > 0)
    # extracting non-silent pitches from reference and user arrays
    reference_pitches = reference_pitches[non_silent_indices_ref]
    user_pitches = user_pitches[non_silent_indices_user]
    # if there are no non-silent pitches, it means they didn't make a sound
    if len(user_pitches) == 0:
        return "You need to make a sound.", None
    if len(reference_pitches) == 0:
        reference_pitches = np.array([0])

    # getting the intervals of the each note
    reference_intervals = segment_pitches(reference_pitches)
    user_intervals = segment_pitches(user_pitches)
    # getting the median F0 of each note
    reference_pitches = middle_50_median(reference_pitches, reference_intervals)
    user_pitches = middle_50_median(user_pitches, user_intervals)
    # finding the minimum length between the reference and user pitch arrays
    min_length = min(len(reference_pitches), len(user_pitches))

    # interpolating to match lengths if necessary
    if len(reference_pitches) > len(user_pitches):
        reference_pitches = np.interp(
            np.linspace(0, len(reference_pitches)-1, min_length), 
            np.arange(len(reference_pitches)), 
            reference_pitches
        )
    else:
        user_pitches = np.interp(
            np.linspace(0, len(user_pitches)-1, min_length), 
            np.arange(len(user_pitches)), 
            user_pitches
        )

    cents = calculate_cents(user_pitches, reference_pitches)

    # variable to store the note (if the deviation was within 1 semitone, to use for the comfortable range)
    note = None
    # if the deviation was within 1 semitone
    if sum(cents)/len(cents) <= 100:
        note = notes_to_frequencies[closest_note_index(user_pitches[0])][0]
    # converting cents to accuracy percentages
    accuracy_percentages = 100 * np.exp(-0.001 * cents)
    # calculating the average accuracy
    average_accuracy = np.mean(accuracy_percentages)

    return None, average_accuracy, note

# function to calculate the average accuracy over 4 segments
def average_accuracy(reference_pitches_per_sound, user_pitches_per_sound):    
    # the accuracies of each pitch
    accuracies = []
    notes = []
    for i in range(4):
        # calculating accuracy for each segment
        _, segment_accuracy, note = calculate_accuracy(reference_pitches_per_sound[i], user_pitches_per_sound[i])
        accuracies.append(segment_accuracy)
        # getting note for each segment
        notes.append(note)
    # returning the average accuracy
    return np.mean(accuracies), notes

# function to find the nearest musical note for a given frequency
def frequency_to_nearest_note(frequency):
    # returning "Silent" if the frequency is 0
    if frequency == 0:
        return "Silent"
    
    # initializing the minimum difference to infinity
    min_diff = float('inf')
    # initializing the closest note as an empty string
    closest_note = ""
    
    # iterating over each note and its frequency
    for note, note_frequency in notes_to_frequencies:
        # calculating the absolute difference between the given frequency and note frequency
        diff = abs(frequency - note_frequency)
        # if this difference is smaller than the current minimum difference
        if diff < min_diff:
            # updating the minimum difference
            min_diff = diff  
            # updating the closest note
            closest_note = note  
    
    # returning the closest note
    return closest_note