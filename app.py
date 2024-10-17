from flask import Flask, request, jsonify, render_template, send_from_directory, session
import numpy as np
import librosa
import os

# creating the Flask application 
app = Flask(__name__)

# setting the app's folders where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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

# function to analyze pitch in an audio file
def analyze_pitch(file_path, sr=None):
    # loading the audio file with librosa
    audio_data = librosa.load(file_path, sr=44100)[0]

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

    return times, pitches, decibels

def analyze_four_pitches(file_path, sr=44100, top_db=20):
    # loading the audio file with librosa
    audio_data, sr = librosa.load(file_path, sr=sr)
    
    # finding the intervals where there is sound as a list of tuples
    sound_intervals = librosa.effects.split(audio_data, top_db=top_db)

    # checking if there are exactly 4 intervals
    if len(sound_intervals) != 4:
        return "You need exactly 4 clearly separated sounds.", None

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

    return None, results

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
    # calculating the absolute differences between reference and user pitches
    differences = np.abs(reference_pitches - user_pitches)
    # converting differences to accuracy percentages
    accuracy_percentages = 100 * np.exp(-0.1 * differences)

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

        # saving the user's audio file with a unique name in the folder where we store trial files.
        file_path = os.path.join(ONE_PITCH_UPLOAD_FOLDER, f'user_trial_{trial_num}.wav')
        file.save(file_path)
        # analyzing the reference pitch from the trial's reference audio file 
        reference_pitches = analyze_pitch(one_pitch_reference_files[trial_num - 1])[1]
        # analyzing the pitch from the user's audio
        user_pitches = analyze_pitch(file_path)[1]

        # calculating how well the user imitated the reference pitch by comparing the two pitch analyses
        error_message, accuracy = calculate_accuracy(reference_pitches, user_pitches)
        # deleting the file if an error occured and displaying an error message 
        if error_message:
            os.remove(file_path)
            return jsonify({'error': error_message})
        
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

        # analyzing the reference pitch from the trial's reference audio file 
        reference_pitches = analyze_four_pitches(four_pitch_reference_files[trial_num - 1])[1]
        
        # analyzing the pitch from the user's audio
        error_message, user_pitches = analyze_four_pitches(file_path)

        # deleting the file if an error occured and displaying an error message 
        if error_message:
            os.remove(file_path)
            return jsonify({'error': error_message})
        
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