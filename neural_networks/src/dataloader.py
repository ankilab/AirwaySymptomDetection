import librosa
import numpy as np
from skimage.transform import resize
from utils.Helper import get_labels_array, scale_minmax
import pandas as pd
import os
import subprocess
from tqdm.notebook import tqdm
from scipy.io.wavfile import read
from numba import njit
import flammkuchen as fl


class DataLoader:
    def __init__(self, params, nb_classes):
        # initialize some class variables
        self.nb_classes = nb_classes
        self.signal_type = params.signal_type
        self.data_dir = params.data_dir
        self.label_dir = params.label_dir
        self.sample_length = params.sample_length
        self.window_shift = params.window_shift
        self.n_mels = params.n_mels
        self.n_fft = params.n_fft
        self.hop_length = params.hop_length
        self.train_data = params.train_data
        self.val_data = params.val_data
        self.test_data = params.test_data

        # check if data_dir and label_dir exist
        if not os.path.isdir(self.data_dir):
            raise ValueError("Couldn't find the data directory: " + str(self.data_dir))
        if not os.path.isdir(self.label_dir):
            raise ValueError("Couldn't find the label directory: " + str(self.label_dir))

    def get_train_val_data(self):
        (X_train, Y_train) = self.__get_mcgill_old_data('train')
        (X_val, Y_val) = self.__get_mcgill_old_data('val')

        return (X_train, Y_train), (X_val, Y_val)

    def get_test_data(self):
        (X_test, Y_test) = self.__get_mcgill_old_data('test')

        return X_test, Y_test

    def get_data_paired(self):
        """
        Creates paired audio/NSA samples to train autoencoder.
        Returns:
        Paired audio- and NSA- samples.
        """
        (trainA, trainB) = self.__get_mcgill_old_data_paired(["P1_1", "P1_2", "P1_3", "P2_1", "P2_2", "P2_3",
                                                              "P3_1", "P3_2", "P3_3", "P4_1", "P4_2", "P4_3"])

        (valA, valB) = self.__get_mcgill_old_data_paired(["P5_1", "P5_2", "P5_3"])
        (testA, testB) = self.__get_mcgill_old_data_paired(["P6_1", "P6_2", "P6_3"])
        return (trainA, trainB), (valA, valB), (testA, testB)

    # ==============================================================================
    # =            McGill old dataset (Rainbow Passage scripted)                   =
    # ==============================================================================
    def __get_mcgill_old_data_paired(self, data_split):
        """
        Get pre-processed data using the Rainbow Passage dataset for autoencoder training.
        Args:
            data_split: Defines which participants are used for the training set.

        Returns:
        Paired audio- and NSA-samples.
        """
        trainA, trainB = [], []
        for data_name in data_split:
            print("\nCurrently loading data: " + data_name)
            annotations = fl.load(self.label_dir + data_name + ".coughai")

            data_audio, sr = librosa.load(self.data_dir + "MIC\\" + data_name + ".wav", sr=44100)
            data_nsa, sr = librosa.load(self.data_dir + "NSA\\" + data_name + ".wav", sr=44100, mono=False)
            data_nsa = np.asfortranarray(data_nsa[0])

            self.signal_type = "MIC"
            imagesA, _ = self.__get_all_spectrograms(data_audio, sr, annotations)

            self.signal_type = "NSA"
            imagesB, _ = self.__get_all_spectrograms(data_nsa, sr, annotations)

            if len(imagesA) > len(imagesB):
                imagesA = imagesA[:len(imagesB)]
            else:
                imagesB = imagesB[:len(imagesA)]

            trainA = [*trainA, *imagesA]
            trainB = [*trainB, *imagesB]

        return np.asarray(trainA), np.asarray(trainB)

    def __get_mcgill_old_data(self, data_usage):
        """
        Get pre-processed data using the Rainbow Passage dataset.
        Args:
            data_usage: 'train', 'val' or 'test'.

        Returns:
        Mel-spectrograms and the associated label.
        """
        x, y = [], []

        if data_usage == 'test':
            data_split = self.test_data
        elif data_usage == 'val':
            data_split = self.val_data
        elif data_usage == 'train':
            data_split = self.train_data
        else:
            raise ValueError('Data usage must be "train", "val" or "test".')

        # data_split contains the different Participants, e.g., [P1_1, P1_2, etc.]
        for data_name in data_split:
            print("\nCurrently loading data: " + data_name)

            # load annotations
            annotations = fl.load(self.label_dir + data_name + ".coughai")

            #  load data from .wav files
            if self.signal_type == "NSA":
                data, sr = librosa.load(self.data_dir + self.signal_type + "\\" + data_name + ".wav", sr=44100,
                                        mono=False)
                data = np.asfortranarray(data[0])
            else:
                data, sr = librosa.load(self.data_dir + self.signal_type + "\\" + data_name + ".wav", sr=44100)

            # generate Mel-spectrograms
            images, labels = self.__get_all_spectrograms(data, sr, annotations)

            x = [*x, *images]
            y = [*y, *labels]

        x = np.array(x)
        y = np.array(y)
        return x, y

    def __get_all_spectrograms(self, data, sampling_rate, annotations, without_no_event=False):
        """
        Creates the mel-spectrograms and associated labels from the input signal.
        Args:
            data: 1-D signal in time-domain.
            sampling_rate: Sampling rate of the 1-D signal.
            annotations: Dictionary containing annotations.
            without_no_event: 'True' if the neural network should be trained with only three classes ('cough',
                              'dry swallow' and 'throat clear').
        Returns:
        Mel-spectrograms and the associated label.
        """

        # get the lag/time_shift from saved in the data annotations dictionary
        if self.signal_type == "NSA":
            time_shift = annotations['shift']
        else:
            time_shift = 0

        # save all event beginnings, endings and event-type (e.g., 'cough') in variables
        if not annotations['annotation'].empty:
            events_beginnings = annotations['annotation']['From']
            events_endings = annotations['annotation']['To']
            events = annotations['annotation']['Event']
        else:
            events = []

        # create return-lists
        images, labels = [], []

        # iterate over the whole 1D-signal
        for sample_start in tqdm(range(time_shift, len(data), self.window_shift)):  # iterate over the whole signal
            array_labels = np.zeros(self.nb_classes)
            found_event = False

            sample_end = sample_start + self.sample_length
            # iterate over all annotated events and check if the current window matches an event
            for j in range(len(events)):
                if self.nb_classes == 4 and events[j] == 'Speech':
                    continue
                event_begin = events_beginnings[j] + time_shift
                event_end = events_endings[j] + time_shift

                if event_begin < sample_end < event_end:
                    inlier_length = sample_end - event_begin
                    # found event if more than 30% of the window is part of the event
                    if inlier_length / self.sample_length > 0.3:
                        found_event = True
                elif event_end < sample_end:
                    no_event_length = sample_end - event_end
                    # found event if less than 70% of the window is no event (equal to more than 30% event)
                    if no_event_length / self.sample_length < 0.7:
                        found_event = True

                if found_event:
                    array_labels = get_labels_array(array_labels, events[j])
                    break

            if found_event is False:
                if without_no_event:  # skip spectrogram creation if no event shall not be created
                    continue
                # no event / silence
                if self.nb_classes == 4:
                    array_labels = get_labels_array(array_labels, "No event")
                elif self.nb_classes == 5:  # if 'silence' should be a separate class and not included into 'no event'
                    array_labels = get_labels_array(array_labels, "Silence")

            # create Mel-spectrogram and append it to the list
            window = data[sample_start:sample_end]
            if len(window) < self.sample_length:
                continue
            else:
                spec = self.__get_spectrogram_image(window, sr=sampling_rate)

                images.append(spec)
                labels.append(array_labels)

        return images, labels

    def __get_spectrogram_image(self, y, sr, fmax=16384):
        """
        Calculation of one mel-frequency spectrogram.
        Args:
            y: 1-D signal.
            sr: 1-D signal sampling rate.
            fmax: upper frequency bound for Mel-spectrogram calculation

        Returns:
        One single Mel-spectrogram image (flipped and normalized between -1 and 1.)
        """
        img = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft,
                                             hop_length=self.hop_length, fmax=fmax, htk=True)
        img = np.log(img + np.finfo(float).eps)

        img = scale_minmax(img, -1.0, 1.0)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom of the image

        img = resize(img, (self.n_mels, 64))
        return img

    # ==============================================================================
    # =                          McGill new dataset (Harry Potter)                 =
    # ==============================================================================
    def get_mcgill_new_data(self, path, without_no_event=False):
        """
        Loads the dataset "New_Data" recorded by McGill university.
        Args:
            path: Path to the folder containing the data.
            without_no_event: 'True' if the neural network should be trained with only three classes ('cough',
                                'dry swallow' and 'throat clear').
        Returns:
        All spectrograms and the associated labels.
        """
        x_train, y_train, x_test, y_test = [], [], [], []
        for file in os.listdir(os.path.join(path, str(self.signal_type))):
            print(f"Currently loading: {file}")

            # skip files that have a bad quality
            if file.__contains__('_bad'):
                continue

            # load 1-D audio signals and annotations
            data, sr = librosa.load(os.path.join(path, str(self.signal_type), file), sr=44100)
            annotations = fl.load(os.path.join(path, 'Annotations', file[:-4:]) + '.coughai')

            # create all Mel-spectrograms
            images, labels = self.__get_all_spectrograms(data, sr, annotations, without_no_event=without_no_event)

            # participants '007', '008', and '009' used for the test set
            if "007" in file or "008" in file or "009" in file:
                x_test = [*x_test, *images]
                y_test = [*y_test, *labels]
            else:
                x_train = [*x_train, *images]
                y_train = [*y_train, *labels]

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        return (x_train, y_train), (x_test, y_test)

    # ==============================================================================
    # =                          Cough database                                    =
    # ==============================================================================
    def get_cough_database_data(self, path, n_samples=1000):
        """
        Creates Mel-spectrograms from the COUGHVID crowd-sourcing dataset
        (https://zenodo.org/record/4048312#.YcB4_8eZNnI) to evaluate our trained neural network.
        Args:
            path: Path where to find the downloaded COUGHVID dataset.
            n_samples: Maximum number of how many samples shall be created from the whole dataset.

        Returns:
        All Mel-spectrograms from different cough recordings.
        """

        # convert ".webm/.ogg" files to ".wav"
        print("Converting files to .wav")
        self.__convert_cough_database_files(path)

        print("Load data as spectrograms")
        X_test = []
        original_signal = []
        from utils.params import Params
        with tqdm(total=n_samples) as pbar:
            for num, file in enumerate(os.listdir(path)):
                if file.__contains__(".wav"):
                    # load .json file to check the probability determined by the authors of the COUGHVID dataset
                    # --> take only files with a cough probability higher than 98%
                    # Params class is used here since it is just an interface for accessing .json files
                    metadata = Params(os.path.join(path, file[:-4:]) + ".json")
                    if float(metadata.cough_detected) < 0.98:
                        continue
                    else:
                        sr, data = read(os.path.join(path, file))
                        data = np.array(data, dtype='float32')

                        # get starting point where the cough itself starts
                        starts = self.__get_cough_event_start(x=data)
                        if len(starts) == 0:
                            continue
                        else:
                            start = starts[0]

                        # load audio 1-D signal
                        data, _ = librosa.load(os.path.join(path, file), sr=sr)
                        sample_end = start + self.sample_length
                        window = data[start:sample_end]
                        if len(window) >= self.sample_length:
                            spec = self.__get_spectrogram_image(window, sr=sr)
                            X_test.append(spec)
                            original_signal.append(window)
                            pbar.update(1)
                        if len(X_test) >= n_samples:
                            return np.array(X_test), original_signal

        print("Finished: Iterated over all .wav files")
        return np.array(X_test), original_signal  # Returning Y_test not needed since all recordings contain cough anyway

    @staticmethod
    def __convert_cough_database_files(folder):
        """
        Convert files from .webm and .ogg to .wav
        (Note: installaton of ffmpeg required --> conda install -c conda-forge ffmpeg)
        Args:
            folder: Path to COUGHVID database and with the metadata_compiled.csv file.
        """
        df = pd.read_csv(folder + 'metadata_compiled.csv')
        names_to_convert = df.uuid.to_numpy()
        for name in tqdm(names_to_convert):
            if os.path.isfile(folder + name + '.wav'):
                continue
            elif os.path.isfile(folder + name + '.webm'):
                subprocess.call(["ffmpeg", "-i", folder + name + ".webm", folder + name + ".wav"], shell=True)
            elif os.path.isfile(folder + name + '.ogg'):
                subprocess.call(["ffmpeg", "-i", folder + name + ".ogg", folder + name + ".wav"], shell=True)
            else:
                print("Error: No file name {0}".format(name))

    def __get_cough_event_start(self, x, window=5000, energy_cutoff=8000):
        """
        Determine the onset of the cough event (i.e., ignore the silent or noisy sections before the onset of
        the actual event) using the rolling standard deviation.
        Args:
            x: 1-D audio signal containing a cough at some point.
            window: Size of the sliding window.
            energy_cutoff: Energy cutoff to determine the starting point of the cough event.
        Returns:
        Th starting point of the cough event within the 1-D audio signal.
        """
        rstd = get_rolling_std(x=x, window=window)
        ndiff = np.diff((rstd > energy_cutoff).astype(np.float32))
        start_point = np.where(ndiff == 1)[0]

        return start_point


@njit
def get_rolling_std(x, window=1000):
    """
    Calculates the rolling standard-deviation of a given input signal.
    Args:
        x: 1-D audio signal.
        window: Size of the sliding window.

    Returns:
    Rolling standard-deviation.
    """
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(window, x.shape[0]):
        y[i] = x[i - window:i].std()
    return y


