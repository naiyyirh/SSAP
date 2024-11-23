from flask import Flask, request, jsonify, render_template, session
import numpy as np
import os
import matplotlib
from audio_analysis import closest_note_index, segment_pitches, middle_50_median, calculate_cents, analyze_pitch, analyze_four_pitches, calculate_accuracy, average_accuracy, frequency_to_nearest_note

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
            f.write(f"{file_path} 1 {reference_times[reference_onset_idx]:.2f} {user_times[user_onset_idx]:.2f} {reference_mean_F0:.2f} {user_mean_F0:.2f} {cents:.2f} {abs_cents:.2f} {error_flag} \n")
        
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
                f.write(f"{file_path} {i + 1} {all_reference_times[reference_onsets[i]]:.2f} {all_times[user_onsets[i]]:.2f} {reference_mean_F0:.2f} {user_mean_F0:.2f} {cents:.2f} {abs_cents:.2f} {error_flag} \n")
        
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