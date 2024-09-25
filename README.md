# SSAP

## Overview

This project is a Flask web application designed for analyzing and comparing pitch accuracy between user-uploaded audio and reference audio. The application allows users to upload either a single pitch or a sequence of four pitches, which are then processed to evaluate the pitch accuracy against predefined reference pitches.

The app uses **librosa** for pitch detection and **NumPy** for numerical computations. The pitch accuracy is calculated by comparing the user's audio against reference recordings, returning an accuracy score that measures how closely the user's pitches match the reference.

## Features

- **Single Pitch Analysis**: Users can record an audio clip containing a single pitch, which is compared to a reference pitch file. The app returns an accuracy score based on the comparison.
- **Four Pitch Sequence Analysis**: Users can record an audio clip containing four distinct pitch sounds. The app checks that the clip contains exactly four sounds, compares the pitches to a reference file, and returns the overall accuracy for all four pitches.
- **Pitch Detection**: Uses **librosa's pYIN** algorithm for pitch detection, allowing the identification of the closest musical note.
- **Real-time Feedback**: Provides accuracy results for the user-uploaded audio in real time.

## Folder Structure

- `uploads/`: This folder stores the uploaded audio files.
  - `one_pitch_uploads/`: Stores user audio files for single pitch analysis.
  - `four_pitch_uploads/`: Stores user audio files for four-pitch analysis.
  
- `static/`: Contains reference audio files used for comparison.
  - `one_pitch.wav`: Reference file for single pitch.
  - `four_pitch.wav`: Reference file for four-pitch sequence.

## How It Works

1. **Audio Upload**:
   - The user records an audio clip containing either one pitch or four pitches.
   - The audio is saved in the appropriate upload folder.

2. **Pitch Analysis**:
   - For single pitch, the app extracts the pitch from the reference file and compares it to the pitch extracted from the user audio clip.
   - For four-pitch sequences, the app ensures the clip contains exactly four distinct sound intervals before comparing each sound to the corresponding pitch in the reference file.

3. **Accuracy Calculation**:
   - The pitch accuracy is calculated by comparing the user-uploaded pitch to the reference pitch using a difference-based formula.
   - The app returns the percentage of pitch accuracy.

### API Endpoints

- **Home Page** (`/`): Displays the homepage where users can record audio clips.
- **Upload File** (`/upload`, `POST`): Accepts audio files and returns pitch accuracy.

## Dependencies

- **Flask**: Python web framework.
- **NumPy**: Library for numerical operations.
- **librosa**: Python library for audio processing and music information retrieval.
- **os**: Module for interacting with the file system.
