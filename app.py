from flask import Flask, request, jsonify, render_template, send_from_directory, session
import numpy as np
import librosa
import os
import matplotlib
import matplotlib.pyplot as plt  

matplotlib.use("Agg")

# creating the Flask application 
app = Flask(__name__)

# setting the app's folders where uploaded files will be stored
participant = 1
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], "p" + str(participant))):
    participant += 1
app.config['UPLOAD_FOLDER'] = os.path.join(app.config["UPLOAD_FOLDER"], "p" + str(participant))
ONE_PITCH_UPLOAD_FOLDER = os.path.join(app.config["UPLOAD_FOLDER"], "one_pitch_uploads")
FOUR_PITCH_UPLOAD_FOLDER = os.path.join(app.config["UPLOAD_FOLDER"], "four_pitch_uploads")

# password for session data
app.secret_key = "secret_key"

# paths to the reference audio file
one_pitch_reference_file = "static/one_pitch.wav"
four_pitch_reference_file = "static/four_pitch.wav"
# lists that hold the file paths of reference audio files for each trial
one_pitch_reference_files = []
four_pitch_reference_files = []
# adding the file paths to the lists
for i in range(1, 4):
    one_pitch_reference_files.append("static/one_pitch/trial_" + str(i) + "_reference.wav")
    four_pitch_reference_files.append("static/four_pitch/trial_" + str(i) + "_reference.wav")

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
    # np.argmin(...): returns the index of the smallest value in the array (so the index where the difference between the note's frequency and the pitch frequency is the smallest)
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
    # converting cents to accuracy percentages
    accuracy_percentages = 100 * np.exp(-0.001 * cents)
    # calculating the average accuracy
    average_accuracy = np.mean(accuracy_percentages)

    return None, average_accuracy

# function to calculate the average accuracy over 4 segments
def average_accuracy(reference_pitches_per_sound, user_pitches_per_sound):    
    accuracies = []
    
    for i in range(4):
        # calculating accuracy for each segment
        segment_accuracy = calculate_accuracy(reference_pitches_per_sound[i], user_pitches_per_sound[i])[1]
        accuracies.append(segment_accuracy)
    
    # returning the average accuracy
    return np.mean(accuracies)

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

# function to display index.html for the main page
@app.route('/')
def index():
    # setting the current trial to 1 when the user first visits the page
    session['current_trial'] = 1
    # creating a variable to store the sum of the one pitch accuracies 
    session['one_pitch_accuracy_sum'] = 0
    # creating a variable to store the number of completed one pitch trials
    session['completed_one_pitch_trials'] = 0
    # creating a variable to store the sum of the four pitch accuracies 
    session['four_pitch_accuracy_sum'] = 0
    # creating a variable to store the number of completed four pitch trials
    session['completed_four_pitch_trials'] = 0
    # showing the webpage
    return render_template('Home Page.html')

@app.route('/get_trial')
def get_trial():
    trial_file_path = ""
    # if the number of completed one pitch trial results is less than the length of the list of one pitch reference files 
    # (meaning we haven't completed all of the one pitch trials)
    if session['completed_one_pitch_trials'] < len(one_pitch_reference_files):
        # getting the file path for the audio clip of the one pitch trial they are currently on
        trial_file_path = one_pitch_reference_files[session.get('current_trial') - 1]
    # if the number oc completed one pitch trial results is greater than or equal to the length of the list of one pitch reference files
    # (meaning we have completed all of the one pitch trials)
    else:
        # getting the file path for the audio clip of the four pitch trial they are currently on
        trial_file_path = four_pitch_reference_files[session.get('current_trial') - 1]
    # incrementing the trial number
    session['current_trial'] += 1
    return jsonify({'trial_file_path': trial_file_path})

@app.route('/upload/<int:trial_num>', methods=['POST'])
def upload_audio(trial_num):
    # checking if the request object contains data labeled 'audio'
    if '1 Pitch' not in request.files and '4 Pitches' not in request.files:
        return jsonify({'error': 'No valid keys provided in request object.'}), 400
    file = None
    # getting the pitch data from the request object
    if '1 Pitch' in request.files:    
        file = request.files['1 Pitch']
    else:  
        file = request.files['4 Pitches']
    reference_pitches = None
    user_pitches = None
    accuracy = None
    if '1 Pitch' in request.files:
        # making sure the trial number is valid
        if trial_num > len(one_pitch_reference_files):
            return jsonify({'error': 'Invalid trial number.'}), 400

        # saving the user's audio file with a unique name in the folder where we store trial files
        file_path = os.path.join(ONE_PITCH_UPLOAD_FOLDER, f'user_trial_{trial_num}.wav')
        file.save(file_path)
        
        # the user's F0 values over time as a file with a unique name in the folder where we store trial files
        F0_file_path = os.path.join(ONE_PITCH_UPLOAD_FOLDER, f'user_trial_{trial_num}_F0_values.txt')

        # summary output file
        summary_file_path = os.path.join(ONE_PITCH_UPLOAD_FOLDER, 'summary.txt')

        # analyzing the reference pitch from the trial's reference audio file 
        reference_times, reference_pitches, reference_decibels, reference_error_message = analyze_pitch(one_pitch_reference_files[trial_num - 1], False)
        # analyzing the pitch from the user's audio
        user_times, user_pitches, user_decibels, error_message = analyze_pitch(file_path, True)
        # deleting the file if an error occured and displaying an error message 
        if error_message:
            os.remove(file_path)
            return jsonify({'error': error_message})

        # calculating how well the user imitated the reference pitch by comparing the two pitch analyses
        error_message, accuracy = calculate_accuracy(reference_pitches, user_pitches)
        # deleting the files if an error occured and displaying an error message 
        if error_message:
            os.remove(file_path)
            os.remove(file_path[:-4] + "_F0_graph.png")
            return jsonify({'error': error_message})
        
        # opening the F0 file in write mode
        with open(F0_file_path, 'w') as f:
            # looping through each time and corresponding pitch value
            for time, pitch in zip(user_times, user_pitches):
                # writing the time (formatted to 3 decimal places) and pitch (formatted to 2 decimal places) to the file
                f.write(f"{time:.3f} {pitch:.2f}\n")

        # getting index of the target note onset
        reference_onset_idx = np.nonzero(reference_pitches)[0][0]
            
        # getting index of the user note onset
        user_onset_idx = np.nonzero(user_pitches)[0][0]

        # getting the intervals of each note
        reference_intervals = segment_pitches(reference_pitches)
        user_intervals = segment_pitches(user_pitches)
        # getting the median F0 of each note
        reference_pitches_medians = middle_50_median(reference_pitches, reference_intervals)
        user_pitches_medians = middle_50_median(user_pitches, user_intervals)
        # getting the mean F0 across the entire audio clips
        reference_mean_F0 = sum(reference_pitches_medians)/len(reference_pitches_medians)
        user_mean_F0 = sum(user_pitches_medians)/len(user_pitches_medians)
        # mean cents
        cents = 1200 * np.log2(user_mean_F0 / reference_mean_F0)
        # absolute value of centst
        abs_cents = abs(cents)
        # error flag
        error_flag = 0
        if abs_cents > 50:
            error_flag = 1

        # if it's the 1st trial, overwrite the file if there is already one there
        mode = 'a'
        if trial_num == 1:
            mode = 'w'
        # opening the summary file 
        with open(summary_file_path, mode) as f:
            f.write(f"{file_path[26:]} 1 {reference_times[reference_onset_idx]:.2f} {user_times[user_onset_idx]:.2f} {reference_mean_F0:.2f} {user_mean_F0:.2f} {cents:.2f} {abs_cents:.2f} {error_flag} \n")
        
        # if everything went well, adding the accuracy score for this trial to the session's one pitch accuracy sum and incrementing the completed count
        session['one_pitch_accuracy_sum'] += accuracy
        session['completed_one_pitch_trials'] += 1
        # if we have completed all of the one pitch trials
        if session['current_trial'] == len(one_pitch_reference_files) + 1:
            # resetting the trial number to 1 (for the four pitch trials)
            session['current_trial'] = 1
            # returning the average accuracy
            return jsonify({'average_accuracy': session['one_pitch_accuracy_sum']/session['completed_one_pitch_trials']})
        
    if '4 Pitches' in request.files:
        # making sure the trial number is valid
        if trial_num > len(four_pitch_reference_files):
            return jsonify({'error': 'Invalid trial number.'}), 400

        # saving the user's audio file with a unique name in the folder where we store trial files.
        file_path = os.path.join(FOUR_PITCH_UPLOAD_FOLDER, f'user_trial_{trial_num}.wav')
        file.save(file_path)

        # the user's F0 values over time as a file with a unique name in the folder where we store trial files
        F0_file_path = os.path.join(FOUR_PITCH_UPLOAD_FOLDER, f'user_trial_{trial_num}_F0_values.txt')

        # summary output file
        summary_file_path = os.path.join(FOUR_PITCH_UPLOAD_FOLDER, 'summary.txt')

        # analyzing the reference pitch from the trial's reference audio file 
        reference_error_message, reference_pitches, all_reference_times, all_reference_pitches = analyze_four_pitches(four_pitch_reference_files[trial_num - 1], False)
        
        # analyzing the pitch from the user's audio
        error_message, user_pitches, all_times, all_pitches = analyze_four_pitches(file_path, True)

        # deleting the files if an error occured and displaying an error message 
        if error_message:
            os.remove(file_path)
            os.remove(file_path[:-4] + "_F0_graph.png")
            return jsonify({'error': error_message})
        
        # replacing NaN values in all_pitches with 0
        all_pitches = np.array([0 if np.isnan(x) else x for x in all_pitches])
        # replacing NaN values in all_reference_pitches with 0
        all_reference_pitches = np.array([0 if np.isnan(x) else x for x in all_reference_pitches])

        # opening the F0 file in write mode
        with open(F0_file_path, 'w') as f:
            # looping through each time and corresponding pitch value
            for time, pitch in zip(all_times, all_pitches):
                # writing the time (formatted to 3 decimal places) and pitch (formatted to 2 decimal places) to the file
                f.write(f"{time:.3f} {pitch:.2f}\n")

        # getting indices of the note onsets
        reference_onsets = np.where((all_reference_pitches[:-1] == 0) & (all_reference_pitches[1:] != 0))[0] + 1
        # checking if the first element is the start of a segment 
        if all_reference_pitches[0] != 0:
            reference_onsets = np.insert(reference_onsets, 0, 0)
        user_onsets = np.where((all_pitches[:-1] == 0) & (all_pitches[1:] != 0))[0] + 1
        # checking if the first element is the start of a segment 
        if all_pitches[0] != 0:
            user_onsets = np.insert(user_onsets, 0, 0)

        # for each sound
        for i in range(4):
        
            # getting the intervals of each note
            reference_intervals = segment_pitches(reference_pitches[i])
            user_intervals = segment_pitches(user_pitches[i])
            # getting the median F0 of each note
            reference_pitches_medians = middle_50_median(reference_pitches[i], reference_intervals)
            user_pitches_medians = middle_50_median(user_pitches[i], user_intervals)
            # getting the mean F0 across the entire audio clips
            reference_mean_F0 = sum(reference_pitches_medians)/len(reference_pitches_medians)
            user_mean_F0 = sum(user_pitches_medians)/len(user_pitches_medians)
            # mean cents
            cents = 1200 * np.log2(user_mean_F0 / reference_mean_F0)
            # absolute value of centst
            abs_cents = abs(cents)
            # error flag
            error_flag = 0
            if abs_cents > 50:
                error_flag = 1

            # if it's the 1st sound of the 1st trial, overwrite the file if there is already one there
            mode = 'a'
            if trial_num == 1 and i == 0:
                mode = 'w'
            # opening the summary file 
            with open(summary_file_path, mode) as f:
                f.write(f"{file_path[27:]} {i + 1} {all_reference_times[reference_onsets[i]]:.2f} {all_times[user_onsets[i]]:.2f} {reference_mean_F0:.2f} {user_mean_F0:.2f} {cents:.2f} {abs_cents:.2f} {error_flag} \n")
        
        # calculating the accuracy between the reference and user pitches
        accuracy = average_accuracy(reference_pitches, user_pitches)
        # if everything went well, adding the accuracy score for this trial to the session's four pitch accuracy sum and incrementing the completed count
        session['four_pitch_accuracy_sum'] += accuracy
        session['completed_four_pitch_trials'] += 1
        # if we have completed all of the four pitch trials
        if session['current_trial'] == len(four_pitch_reference_files) + 1:
            # returning the average accuracy
            return jsonify({'average_accuracy': session['four_pitch_accuracy_sum']/session['completed_four_pitch_trials']})
        
    # returning the accuracy score for this trial
    return jsonify({'accuracy': accuracy})

# running the Flask application
if __name__ == '__main__':
    # changing the current directory to the directory this file is in
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)
    # creating the upload folders if they don't exist
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
        os.makedirs(ONE_PITCH_UPLOAD_FOLDER)
        os.makedirs(FOUR_PITCH_UPLOAD_FOLDER)
    # starting the Flask development server
    app.run(debug=True)