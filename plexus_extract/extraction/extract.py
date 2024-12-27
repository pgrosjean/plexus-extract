##########
# Author: Parker Grosjean
##########

###########################################
###########################################
## Importing Dependencies
###########################################
###########################################

from typing import List, Tuple, Union
from tqdm import tqdm
import os
from argparse import ArgumentParser

import numpy as np
import torch
import zarr
import nd2reader as nd2
import skimage

from scipy.optimize import curve_fit
from cellpose import models as cpm
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from oasis.functions import deconvolve

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


###########################################
###########################################
## Defining Functions
###########################################
###########################################


def expand_mask(mask_da):
    """
    This expands an int 2D mask into a binary 3D mask tensor.

    Parameters
    ----------
        mask_da: np.ndarray
            2D int mask

    Returns
    -------
        bg_mask: np.ndarray
            Background mask.
        exp_mask>: np.ndarray
            Expanded 3D bool mask.
    """
    bg_mask = np.array((mask_da == 0)).astype('float')
    exp_mask = []
    for m_num in sorted(np.unique(mask_da))[1:]:
        mask = (mask_da == m_num).astype('float')
        exp_mask.append(mask)
    exp_mask = np.array(exp_mask)
    return bg_mask, exp_mask


def find_mask_cellpose(image: np.ndarray,
                       cp_model: cpm.CellposeModel,
                       diameter: float = 14.41,
                       flow_thresh: float = 0.6,
                       cellprob_thresh: float = -0.1) -> np.ndarray:
    """
    This function segments live cell nuclei.
    
    Parameters
    ----------
    image : np.ndarray
        Image to be segmented.
    
    Returns
    -------
    mask_cp : np.ndarray
        The corresponding single cell nuclei mask.
    """
    # https://cellpose.readthedocs.io/en/latest/settings.html
    mask_cp, _, _, = cp_model.eval(image,
                                     diameter=diameter,
                                     channels=[0,0], 
                                     min_size=200,
                                     channel_axis=0,
                                     flow_threshold=flow_thresh,
                                     cellprob_threshold=cellprob_thresh, 
                                     normalize=True,
                                     resample=False)
    return mask_cp


def normalize_signal(sig_vec):
    """
    Normalize a list of signal arrays to a range of 0 to 1.

    Parameters
    ----------
    sig_vec : list of ndarray
        A list containing signal arrays. Each array in the list should be one-dimensional.

    Returns
    -------
    ndarray
        A numpy array of normalized signal arrays where each signal is scaled to the range of 0 to 1.
        If any signal in the input is a constant array, it will be shifted by 1.

    Notes
    -----
    This function normalizes each array in the input list by first subtracting the minimum value of the array,
    ensuring the signal starts from zero, and then divides by the maximum value to scale the signal to between 0 and 1.
    If the maximum value after subtracting the minimum is 0 (indicating a constant array), the signal is incremented by 1.
    """
    norm_sigs = []
    for signal in sig_vec:
        signal = signal - np.amin(signal)
        if np.amax(signal) == 0:
            signal = signal + np.ones_like(signal)
        signal = signal / np.amax(signal)
        norm_sigs.append(signal)
    norm_sigs = np.array(norm_sigs)
    return norm_sigs


def normalize_background_signal(signal: np.ndarray):
    """
    Normalize a single background signal array and return the minimum and maximum values.

    Parameters
    ----------
    signal : ndarray
        A one-dimensional numpy array representing the signal to be normalized.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - The normalized signal array, scaled to a range from 0 to 1.
        - The minimum value of the original signal.
        - The maximum value of the original signal after subtracting the minimum value.

    Notes
    -----
    The function normalizes the input signal by first subtracting its minimum value, thus shifting the signal to start from zero,
    and then dividing by its new maximum value to scale it to between 0 and 1.
    """
    sig_min = np.amin(signal)
    signal = signal - sig_min
    sig_max = np.amax(signal)
    signal = signal/sig_max
    return signal, sig_min, sig_max


def collate_files(base_path: str, file_suffix: str='.tif') -> List[str]:
    """
    This function collates all files with a specified file suffix
    given a base directory.
    
    Parameters
    ----------
    base_path : str
        The path to the base directory for files to be collated.
    file_suffix : str
        The suffix for the files to collate.
    
    Returns
    -------
    foi_list : List[str]
        The files of interest in list form.
    """
    foi_list = [] # files of interest list
    for path, directories, files in os.walk(base_path):
        for file in files:
            if file.endswith(file_suffix):
                foi_list.append(os.path.join(path, file))
    foi_list = sorted(foi_list)
    return foi_list


class ND2Image:
    """
    This class is used to read in ND2 nuclei images aqcuired on a Nikon scope.

    Paramters
    ---------
    fname: str
        ND2 filename of interest for reading to numpy array.

    Attributes
    ----------
        nd2_obj:  ND2Reader Class Instance
            ND2 Reader class instnace for specified nd2 file.
    """
    def __init__(self, fname):
        self.nd2_obj = nd2.ND2Reader(fname)
        if len(self.nd2_obj.metadata['z_levels']) == 0:
            self.nd2_obj.metadata['z_levels'] = range(1)
        if len(self.nd2_obj.metadata['channels']) == 0:
            self.nd2_obj.metadata['channels'] = range(1)


    def get_image_array(self):
        """
        This function generates the image array from the nd2 file.

        Returns
        -------
        np.ndarray
             A numpy array with the shape [fov_num, y_dim, x_dim]
        """
        nd2_obj = self.nd2_obj
        nd2_obj.iter_axes = 'v'
        parser = nd2_obj._parser
        num_roi = nd2_obj.sizes['v']
        try:
            num_channels = nd2_obj.sizes['c']
        except KeyError:
            num_channels = 1
        
        if num_channels == 1:
            frame_count = 0
            while frame_count < num_roi:
                full_im = []
                for _ in range(num_roi):
                    full_im.append(np.array(parser.get_image(frame_count)))
                    frame_count += 1
                full_im = np.array(full_im)
        else:
            full_im = []
            frame_count = 0
            for _ in range(num_roi):
                inner_im = []
                for _ in range(num_channels):
                    inner_im.append(np.array(parser.get_image(frame_count)))
                    frame_count += 1
                full_im.append(np.array(inner_im))
            full_im = np.array(full_im)
            full_im = full_im[:, 0, :, :]
        return full_im

    def close(self):
        """
        This function closes the nd2reader object and file.
        """
        self.nd2_obj.close()


class ND2Video:
    """
    This class enables reading in an ND2 File from a Nikon scope into a
    numpy array, which will be used for all downstream processing.

    Parameters
    ----------
        fname (str): The filenmae of the ND2 file to read into a numpy array.

    Attributes
    ----------
        nd2_obj: ND2Reader Class Instance
             ND2 Reader class instance for specified nd2 file.
        parser: Parser Class Instance
             Parser used to parse through raw ND2 file.
        frame_index: dict 
            Dictionary pointing from fov to index.
        fs: int 
            Sampling frequency of video.
    """
    def __init__(self, fname):
        nd2_obj = nd2.ND2Reader(fname)
        self.nd2_obj = nd2_obj
        axis_sizes = nd2_obj.sizes
        self.parser = nd2_obj._parser
        if len(self.nd2_obj.metadata['z_levels']) == 0:
            self.nd2_obj.metadata['z_levels'] = range(1)
        if len(self.nd2_obj.metadata['channels']) == 0:
            self.nd2_obj.metadata['channels'] = range(1)
        self._x = axis_sizes['x']
        self._y = axis_sizes['y']
        self._t = axis_sizes['t']
        try:
            self._v = axis_sizes['v']
        except KeyError:
            self._v = 1
        self.frame_index = {}
        acq_times = np.array([x for x in nd2_obj._parser._raw_metadata.acquisition_times])
        self.fs = int(np.round(1/np.diff(acq_times)[0]))
        # setting frame index
        for fov in range(self._v):
            num_frames = self._t
            fov_start = fov*num_frames
            fov_end = (fov+1)*num_frames
            self.frame_index[fov] = np.arange(fov_start, fov_end)

    def get_fov(self, fov_num):
        """
        Return a timeseries movie for a specified field of view number.

        Parameters
        ----------
        fov_num : int
            The number corresponding to the field of view to return.

        Returns
        -------
        ndarray
            A numpy array of the timeseries corresponding to the specified field of view,
            with shape [time_frames, y_dim, x_dim].
        """
        timeseries = np.array([self.parser.get_image(frame_num) for frame_num in self.frame_index[fov_num]])
        return timeseries
    
    def get_specific_fov(self, fov_num, frame_num):
        """
        Retrieve a specific frame from a specific field of view.

        Parameters
        ----------
        fov_num : int
            The number corresponding to the field of view.
        frame_num : int
            The frame number within the specified field of view.

        Returns
        -------
        ndarray
            A numpy array of the specified frame, typically with dimensions corresponding to [y_dim, x_dim].
        """
        frame_idx = self.frame_index[fov_num][frame_num]
        frame = np.array(self.parser.get_image(frame_idx))
        return frame

    def get_full_video(self):
        """
        Return the entire video consisting of all fields of view and all time frames.

        Returns
        -------
        ndarray
            A numpy array containing the entire video, shaped [fov, time_frames, y_dim, x_dim].
        """
        full_video = np.array([self.get_fov(key) for key in self.frame_index.keys()])
        return full_video
    
    def get_frames(self, frame_num):
        """
        Retrieve a specific frame from all fields of view.

        Parameters
        ----------
        frame_num : int
            The frame number to retrieve across all fields of view.

        Returns
        -------
        ndarray
            A numpy array containing the specified frame from all fields of view,
            shaped [num_fov, 1, y_dim, x_dim].
        """
        frames = np.array([self.get_specific_fov(key, frame_num) for key in self.frame_index.keys()])
        return frames  # shape: [num_fov, 1, y, x]
          
    def close(self):
        """
        This function closes the nd2reader object and file.
        """
        self.nd2_obj.close()
    
    
def reset_label_numbers(arr):
    """
    This function resets the label numbers in a integer labeled array.
    
    Args:
        arr (array-like): Labeled array with missing values.
    
    Returns:
        arr (array-like): Relabeled array.
    """
    c = 1
    unique_sorted = sorted(np.unique(arr))[1:]
    arr_copy = arr.copy()
    for l_num in unique_sorted:
        bool_ind = arr == l_num
        arr_copy[bool_ind] = c
        c += 1
    return arr_copy
    

def flat_field_correction(video, flat_field_image):
    """
    Apply flat field correction to the video.
    
    Parameters
    ----------
    video: np.ndarray
        numpy array of shape [frames, y, x]
    flat_field_image: np.ndarray
        numpy array of shape [y, x]

    Returns
    -------
    Corrected video: np.ndarray
    """
    return video / flat_field_image[np.newaxis, :, :]


def generate_colormap_for_segmentation(label_mask):
    """
    Generate a colormap for a given segmentation mask.
    
    Parameters
    ----------
    label_mask : ndarray
        ndimage label mask.

    Returns
    -------
    colormap : matplotlib.colors.ListedColormap
        Colormap suitable for matplotlib.imshow.
    """
    
    # Find the number of unique labels in the mask
    unique_labels = np.unique(label_mask)
    num_labels = len(unique_labels)
    
    # Generate random colors for each label
    colors = np.random.rand(num_labels, 3)
    colors[0] = [0, 0, 0]
    
    # Create a colormap with the generated colors
    colormap = mcolors.ListedColormap(colors)
    
    return colormap


def filter_signals_from_background(signals, background, threshold=1):
    """
    Filter the signals based on the mean difference from the background signal.

    Parameters
    ----------
    signals : ndarray
        2D array of shape [num_cells, time] representing signals from different cells over time.
    background : ndarray
        1D array of shape [time] representing the background signal curve.
    threshold : float
        Minimum mean difference required for a signal to be considered different from the background.

    Returns
    -------
    mask : ndarray
        mask to use in order to get filtered signal array.
    """
    signals = normalize_signal(signals)
    for signal in signals:
        plt.plot(signal)
    plt.plot(background, linestyle='dashed', linewidth=4, c='k')
    plt.show()
    
    # Calculate mean differences
    mean_differences = np.mean(np.abs(signals - background), axis=1)
    plt.hist(mean_differences, bins=100)
    plt.show()
    
    # Filter based on threshold
    mask = mean_differences > threshold
    
    return mask


def unique_filename(filename):
    """
    Ensure a filename is unique.

    If the filename already exists, append an incrementing number to the end
    of the filename (before the file extension) until a unique filename is found.

    Parameters
    ----------
    filename : str
        Original filename.

    Returns
    -------
    str
        Unique filename.
    """
    basename, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{basename}_{counter}{ext}"
        counter += 1
    return filename


def extract_nd2_signal(video_file: str, 
                        ff_image: str,
                        cp_model: cpm.CellposeModel,
                        cp_model_nuclei: cpm.CellposeModel,
                        nuclei_file: Union[str,None] = None,
                        verbose: bool = False,
                        background_noise_threshold: float = 0.04,
                        stimulation_window: Union[float, None] = None) -> Tuple[np.ndarray,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:
    """
    This function extracts out the calcium imaging signal.
    
    Parameters
    ----------
    video_file : str
        The .nd2 file to extract the signal from.
    ff_image : str
        The flatfield image to use for flatfield correction.
    nuclei_file : Union[None, str], optional
        The .nd2 file with dapi nuclear information.
    verbose : bool, optional
        If True, will plot out additional information.
    background_noise_threshold : float, optional
        The threshold for filtering out background noise.
    stimulation_window : Union[float, None], optional
        The window of time to consider for stimulation.
        
    Returns
    -------
    all_signals : np.ndarray
        The signal arrays of shape [num_fov, num_cell, video_frame]
        where fov means the field of view
    """
    # Setting up the lists to store the signals
    all_signals = []
    bg_signals = []
    all_masks = []
    images = []
    nuclei_bool = []

    vid_obj = ND2Video(video_file)
    vids = vid_obj.get_full_video()
    if nuclei_file is not None:
        im_obj = ND2Image(nuclei_file)
        dapi_ims = im_obj.get_image_array()
    else:
        dapi_ims = [None]*vids.shape[0]
    
    for fov_vid, fov_dapi_im in zip(vids, dapi_ims):
        # Perform flat field correction
        fov_vid_ff_correct = flat_field_correction(fov_vid, ff_image)
        # Setting up the summed intensity projection 
        # of the video for segmentation
        sip_gcamp = np.sum(fov_vid_ff_correct, axis=0)
        sip_gcamp_cp = sip_gcamp - np.amin(sip_gcamp)
        sip_gcamp_cp = (sip_gcamp_cp/np.amax(sip_gcamp_cp))
        sip_gcamp_cp = sip_gcamp_cp / np.exp(sip_gcamp_cp)
        sip_gcamp_cp = sip_gcamp_cp - np.amin(sip_gcamp_cp)
        sip_gcamp_cp = sip_gcamp_cp / np.amax(sip_gcamp_cp)
        sip_gcamp_cp = sip_gcamp_cp * 255
        # Extracting the mask using the cellpose model
        mask = find_mask_cellpose(sip_gcamp_cp, cp_model)
        mask = skimage.morphology.remove_small_objects(mask, min_size=20)
        mask = reset_label_numbers(mask)

        # For filtering nuclei
        if fov_dapi_im is not None:
            # correct mask to only include masks with nuclei
            # dapi_im_for_cp = (fov_dapi_im/np.amax(fov_dapi_im))*255
            image = fov_dapi_im
            image = image - np.amin(image)
            image = image / np.max(image)
            image = image / np.exp(image)
            image = image - np.amin(image)
            image = image / np.max(image)
            image = image * 255

            dapi_mask = find_mask_cellpose(image,
                                           cp_model_nuclei,
                                           diameter=9.1,
                                           flow_thresh=0.6,
                                           cellprob_thresh=-0.3)
            cond = dapi_mask > 0
            # Getting rid of all nuclei that are not at least 75 contained in neuron mask
            dapi_mask_overlap = dapi_mask * (mask > 0)  # Getting rid of all nuclei not in neuron
            real_objects = np.unique(dapi_mask_overlap)
            for obj in range(np.amax(dapi_mask)):
                nuclei_size = np.sum(dapi_mask == obj)
                overlap_size = np.sum(dapi_mask_overlap == obj)
                if obj not in real_objects and overlap_size/nuclei_size < 0.75:
                    dapi_mask[dapi_mask == obj] = 0  # Get rid of nuclei not in neurons
            dapi_mask = reset_label_numbers(dapi_mask)
            cond = dapi_mask > 0
            masked_cond = cond * mask
            nuclei_positive_cells = np.unique(masked_cond)
            if np.amax(mask) > 0 and nuclei_positive_cells.shape[0] > 0:
                nuc_pos_bool = np.zeros((np.amax(mask),))
                for obj in nuclei_positive_cells[1:]:
                    nuc_pos_bool[obj-1] = 1
                nuclei_bool.append(nuc_pos_bool)
            else:
                nuclei_bool.append(None)
        
        if np.amax(mask) == 0:
            all_signals.append(None)
            bg_signals.append(None)
            all_masks.append(None)
            images.append(None)
        else:
            # Extracting the signal
            vid_flat = fov_vid.reshape(fov_vid.shape[0], -1) # shape: [frame, y*x]
            if stimulation_window is not None:
                assert stimulation_window*2 <= vid_flat.shape[0], f"Stimulation window {stimulation_window} is too large for the video of shape: {vid_flat.shape[0]}."
                vid_flat_first = vid_flat[:stimulation_window, :] # shape: [stimulation_window, y*x]
                vid_flat_last = vid_flat[-stimulation_window:, :] # shape: [stimulation_window, y*x]
                vid_flat = np.concatenate([vid_flat_first, vid_flat_last], axis=0) # shape: [2*stimulation_window, y*x]
            bg_tensor, mask_tensor = expand_mask(mask) # shape: [cell, y, x]
            mask_flat = mask_tensor.reshape(mask_tensor.shape[0], -1) # shape: [cell, y*x]
            bg_flat = bg_tensor.reshape(-1) # shape: [y*x]
            signal = np.dot(mask_flat, vid_flat.T) # shape: [cell, frame]
            cell_size = np.sum(mask_flat, axis=-1) # shape: [cell]
            # Normalizing by cell size
            signal_norm = signal.T/cell_size
            signal_raw = signal_norm.T # shape: [cell, frame]
            signal_bg = np.dot(bg_flat, vid_flat.T)/bg_flat.sum() # shape: [frame]

            ########## currently removing for new background correction test
            # signal_bg, bg_min, bg_max = normalize_background_signal(signal_bg)

            # Fitting the exponential decay to the background signal
            # bg_curve = fit_exponential_decay(signal_bg)
            # bg_curve_scaled = (bg_curve*bg_max) + bg_min
            # Filtering signals based on background
            # filter_cond = filter_signals_from_background(signal_raw,
            #                 bg_curve,
            #                 threshold=background_noise_threshold)
            # signal = signal_raw - np.expand_dims(bg_curve_scaled, 0)
            # signal = (signal.T - np.min(signal, axis=1)).T

            # Creating a boolean mask for the filtered signals
            # is_active = np.zeros(signal.shape[0])
            # is_active[filter_cond] = 1
            # activity_bool.append(is_active)
            # signal = signal[filter_cond]

            # saving all the arrays for saving
            all_signals.append(signal_raw)
            all_masks.append(mask)
            bg_signals.append(signal_bg)
            images.append([sip_gcamp, fov_dapi_im])
            # plotting results if requested
            if verbose:
                plt.figure()
                plt.imshow(fov_dapi_im, cmap='gray')
                plt.imshow(cond, alpha=0.2, cmap='inferno')
                plt.title('Neuronal Nuclei')
                plt.show()
                plt.imshow(sip_gcamp/np.amax(sip_gcamp), cmap='gray')
                plt.imshow(mask, cmap=generate_colormap_for_segmentation(mask), alpha=0.1)
                plt.title('Masked Neurons')
                plt.show()
                plt.figure()
                plt.plot(signal_bg)
                # plt.plot(bg_curve)
                plt.legend(['background signal', 'background curve'])
                plt.title('Traces')
                plt.show()

    return_dict = {'signals': all_signals,
                   'background': bg_signals,
                   'masks': all_masks,
                   'images': images,
                   'nuclei_positive': nuclei_bool,
                   }
    return return_dict
    

def extract_peak_parameters(signal, prominence=40, width=[5,600], verbose=False):
    """
    Extract peak parameters from a signal.

    Parameters
    ----------
    signal : ndarray
        1D array representing the signal to extract peaks from.
    prominence : float
        Minimum prominence of peaks.
    width : list
        Minimum and maximum width of peaks.

    Returns
    -------
    peaks_parameters : ndarray
        Array containing peak parameters for each cell.
    """
    peaks_parameters = np.zeros([signal.shape[0],signal.shape[1], 5]) 
    offset = 1.1
    if verbose:
        plt.figure(figsize=(10,10), dpi=100)
    for cell in range(0,signal.shape[0]):
        peaks, peak_props = find_peaks(signal[cell,:], prominence=40, width=[5,600])
        peaks_parameters[cell,peaks,0] = 1
        isi = []
        for i in range(1,len(peaks)):
            isi.append(peaks[i] - peaks[i-1])
        isi.append(0)
        peaks_parameters[cell,peaks,1] = peak_props['prominences']
        peaks_parameters[cell,peaks,2] = peak_props['widths']
        peaks_parameters[cell,peaks,3] = peak_props['width_heights']
        peaks_parameters[cell,peaks,4] = isi
        if verbose:
            plt.plot((cell*offset)+signal[cell,:], color = "green", alpha=0.2)
            plt.plot(peaks, (cell*offset)+signal[cell,peaks], "o", color="black", markersize=1.5)
    if verbose:
        plt.show()
    return peaks_parameters


def get_well_id(file):
    """
    Returns the well id from a file.
    """
    fname = file.split('/')[-1]
    fname_split = fname.split('_')
    well_id = fname_split[4]
    return well_id


def get_channel_id(file):
    """
    Returns the channel name of a file.
    """
    fname = file.split('/')[-1]
    fname_split = fname.split('_')
    channel_name = fname_split[5]
    return channel_name


def find_matching_files(list1, list2, match_lambda):
    """
    Find and return matching and unmatched files between two lists based on a matching criterion defined by a lambda function.

    Parameters
    ----------
    list1 : list
        The first list of files to match.
    list2 : list
        The second list of files to match.
    match_lambda : function
        A lambda or function that takes a file from either list and returns a key used for matching files. 
        The lambda should handle file transformation or extraction of characteristics used for comparison.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - matched_files: A list of tuples where each tuple contains a pair of matched files from list1 and list2.
        - unmatched_files: A list of files that did not match from both lists.

    """
    # Transform file lists based on the lambda function
    transformed_list1 = {match_lambda(file): file for file in list1}
    transformed_list2 = {match_lambda(file): file for file in list2}

    # Find matched keys
    matched_keys = set(transformed_list1.keys()) & set(transformed_list2.keys())

    # Create matched and unmatched lists
    matched_files = [(transformed_list1[key], transformed_list2[key]) for key in matched_keys]
    unmatched_files = [file for file in list1 if match_lambda(file) not in matched_keys] + \
                      [file for file in list2 if match_lambda(file) not in matched_keys]

    return matched_files, unmatched_files


def get_matching_channel_files(file_list: List[str]) -> List[Tuple[str,str]]:
    """
    Matches files based on well IDs for two specific channels, FITC and DAPI, from a list of file names.

    Parameters
    ----------
    file_list : List[str]
        A list of file names which are expected to include channel identifiers and well IDs.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples, where each tuple contains a pair of matched files between the FITC and DAPI channels.
    """
    fitc_list = [x for x in file_list if 'DAPI' not in get_channel_id(x)]
    dapi_list = [x for x in file_list if 'DAPI' in get_channel_id(x)]
    matched_files, unmatched_files = find_matching_files(fitc_list,
                                                        dapi_list,
                                                        get_well_id)
    if len(unmatched_files) > 0:
        print(f'No matches found for the following files: {unmatched_files}')
    return matched_files
    

def check_zarr_group_existance(zarr_root, path_to_check):
    """
    Checks for the existence of a zarr group and generates a 
    unique path if the specified path already exists.

    Parameters
    ----------
    zarr_root : Zarr hierarchy
        The root of the Zarr file hierarchy.
    path_to_check : str
        The path to check for existence within the Zarr hierarchy.

    Returns
    -------
    str
        A unique path derived from `path_to_check` if the 
        original path exists; otherwise, returns the original path.
    """
    split_check = path_to_check.split('/')
    end_path = split_check[-1]
    while end_path in list(zarr_root['/'.join(split_check[:-1])].keys()):
        end_path_split = end_path.split('_')
        unique_id = int(end_path_split[-1]) + 1
        end_path = end_path_split[0] + "_" + str(unique_id)
    return end_path


def calculate_flatfield_image(video_files):
    """
    Calculates a flatfield image from the first five frames across multiple video files.

    Parameters
    ----------
    video_files : list
        A list of paths to video files to process.

    Returns
    -------
    ndarray
        A flatfield-corrected image, which is a smoothed average of the 20th percentile of pixel values from the first 
        five frames across all videos.

    Notes
    -----
    This function computes a quantile image for each of the first five frames across all provided video files,
    averages these quantile images, and then applies a Gaussian filter for smoothing.
    """
    ff_image = []
    print('Calculating Flatfield Image...')
    for frame_num in np.arange(5):
        for_quantile = []
        for file in tqdm(video_files):
            vid_obj = ND2Video(file) #vup.ND2Video(video_file)
            vid = vid_obj.get_frames(frame_num) # shape: [num_fov, y, x]
            for_quantile.extend(vid)
        for_quantile = np.array(for_quantile)
        ff_image.append(np.quantile(for_quantile, 0.2, axis=0))
    ff_image = np.array(ff_image)
    return skimage.filters.gaussian(np.mean(ff_image, axis=0), sigma=10)


def exponential_decay(x, a, b, c):
    """
    Exponential decay function.
    
    Parameters
    ----------
    x : ndarray
        Independent variable.
    a : float
        Amplitude.
    b : float
        Decay constant.
    c : float
        Offset.

    Returns
    -------
    ndarray
        Values of the exponential decay function.
    """
    return a * np.exp(-b * x) + c


def fit_exponential_decay(data):
    """
    Fit an exponential decay function to the given data and return the line of best fit.

    Parameters
    ----------
    data : ndarray
        1D array of data points to fit.

    Returns
    -------
    best_fit : ndarray
        Line of best fit based on the exponential decay function.
    """
    x = np.arange(len(data))
    popt, _ = curve_fit(exponential_decay, x, data, maxfev=5000)
    best_fit = exponential_decay(x, *popt)
    return best_fit


def get_average_background(zarr_root: zarr.hierarchy):
    """
    Calculate the average background signal across all wells and fields of view.

    Parameters
    ----------
    zarr_root : zarr.hierarchy
        The root of the Zarr hierarchy.

    Returns
    -------
    ndarray
        The average background signal across all wells and fields of view.
    """
    all_backgrounds = []
    for well in zarr_root.keys():
        for fov in zarr_root[well].keys():
            bg = np.array(zarr_root[f'{well}/{fov}/background'])
            # normalizing the background signal between 0 and 1
            if bg[30] - bg[0] < 0:
                bg = bg - np.amin(bg)
                bg = bg / np.amax(bg[:30])
            else:
                bg = bg - np.amin(bg)
                bg = bg / np.amin(bg[:30])
            all_backgrounds.append(bg)
    all_backgrounds = np.vstack(all_backgrounds)
    return np.mean(all_backgrounds, axis=0)


def asymmetric_least_squares(y, lam=1e4, p=0.001, niter=10):
    """
    Fits an asymmetric least squares model to fluorescence signal to correct for photobleaching decay.

    Parameters
    ----------
    y : array_like
        The input fluorescence signal (1D array).
    lam : float, optional
        The smoothing parameter lambda. Higher values make the baseline smoother.
    p : float, optional
        The asymmetry parameter (between 0 and 1). Lower values give more weight to negative deviations.
    niter : int, optional
        The number of iterations to perform.

    Returns
    -------
    z : ndarray
        The fitted baseline representing the photobleaching decay.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))

    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def fit_baselines_to_traces(raw_signals: np.ndarray):
    """
    This function fits baselines to the traces using asymetric least squares regression.

    Parameters
    ----------
    raw_signals : np.ndarray
        The raw signals to fit baselines to.
    
    Returns
    -------
    deltaf_f : np.ndarray
        The deltaf_f signals.
    baselines : np.ndarray
        The fitted baselines.
    inferred_calcium : np.ndarray
        The inferred calcium signals.
    inferred_spiking : np.ndarray
        The inferred spiking signals.
    lagrange_multiplier : np.ndarray
        The lagrange multiplier.
    ar2_coeffs : np.ndarray
        The AR(2) model coefficient.
    """
    baselines = []
    deltaf_f = []
    inferred_calcium = []
    inferred_spiking = []
    lagrange_multiplier = []
    ar2_coeffs = []
    for signal in raw_signals:
        baseline = asymmetric_least_squares(signal, lam=1e5, p=0.001, niter=10)
        df_f = (signal - baseline) / baseline
        c, s, _, g, lam = deconvolve(df_f, penalty=2, optimize_g=0)
        inferred_calcium.append(c)
        inferred_spiking.append(s)
        baselines.append(baseline)
        lagrange_multiplier.append(lam)
        ar2_coeffs.append(g)
        deltaf_f.append(df_f)
    baselines = np.vstack(baselines)
    deltaf_f = np.vstack(deltaf_f)
    inferred_calcium = np.vstack(inferred_calcium)
    inferred_spiking = np.vstack(inferred_spiking)
    lagrange_multiplier = np.hstack(lagrange_multiplier)
    ar2_coeffs = np.hstack(ar2_coeffs)
    return deltaf_f, baselines, inferred_calcium, inferred_spiking, lagrange_multiplier, ar2_coeffs


def normalize_raw_signals(zarr_root: zarr.hierarchy):
    """
    Normalize the raw signals in the Zarr hierarchy.

    Parameters
    ----------
    zarr_root : zarr.hierarchy
        The root of the Zarr hierarchy.

    Returns
    -------
    None
    """
    print("Normalizing Raw Signals...")
    for well in tqdm(zarr_root.keys()):
        for fov in zarr_root[well].keys():
            raw_signal = np.array(zarr_root[f'{well}/{fov}/raw_signal']) # shape: [num_cells, time]
            deltaf_f, baselines, inferred_calcium, inferred_spiking, lagrange_multiplier, ar2_coeffs = fit_baselines_to_traces(raw_signal)
            grp = zarr_root[f'{well}/{fov}']
            grp.create_dataset('baselines', data=baselines)
            grp.create_dataset('signal', data=deltaf_f)
            grp.create_dataset('inferred_calcium', data=inferred_calcium)
            grp.create_dataset('inferred_spiking', data=inferred_spiking)
            grp.create_dataset('lagrange_multiplier', data=lagrange_multiplier)
            grp.create_dataset('ar2_coeffs', data=ar2_coeffs)


def extract_all_signals(zarr_root: zarr.hierarchy,
                        file_list: List[str],
                        ff_image: np.ndarray,
                        find_nuclei: bool,
                        verbose: bool,
                        background_noise_threshold: float,
                        stimulation_window: Union[int, None]) -> None:
    """
    This function extracts all the signals.

    Parameters
    ----------
    zarr_root : zarr.hierarchy
        The root of the Zarr hierarchy.
    file_list : list
        A list of file names to extract signals from.
    ff_image : ndarray
        A flatfield-corrected image.
    find_nuclei : bool
        If True, will find nuclei for downstream analysis.
    verbose : bool
        If True, will plot out additional information.
    background_noise_threshold : float
        The threshold for filtering out background noise.
    stimulation_window : int
        The window of time to consider for stimulation.

    
    Returns
    -------
    None
    """
    # Pre populating the zarr file
    well_set = set()
    for file in file_list:
        well_id = get_well_id(file)
        well_set.add(well_id)
    for well_id in well_set:
        zarr_root.create_group(f"well_{well_id}")
    
    print("Extracting Signals from fluorescent videos...")
    # Adding data to zarr file
    pretrained_model = os.path.join(os.path.dirname(__file__), '..', 'segmentation/model/CP_20241011_geci_learn')
    pretrained_model_nuclei = os.path.join(os.path.dirname(__file__), '..', 'segmentation/model/CP_20241011_geci_learn_nuclei_1')
    if torch.cuda.is_available():
        cp_model = cpm.CellposeModel(gpu=True, pretrained_model=pretrained_model)
        cp_model_nuclei = cpm.CellposeModel(gpu=True, pretrained_model=pretrained_model_nuclei)
    else:
        cp_model = cpm.CellposeModel(gpu=False, pretrained_model=pretrained_model)
        cp_model_nuclei = cpm.CellposeModel(gpu=True, pretrained_model=pretrained_model_nuclei)

    if find_nuclei:
        # Extracting data for DAPI and FITC channels that are matched
        zipped_files = get_matching_channel_files(file_list)
        for vid_file, dapi_file in tqdm(zipped_files): #tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable
            processed_dict = extract_nd2_signal(vid_file,
                                                ff_image,
                                                cp_model=cp_model,
                                                cp_model_nuclei=cp_model_nuclei,
                                                nuclei_file=dapi_file,
                                                verbose=verbose,
                                                background_noise_threshold=background_noise_threshold,
                                                stimulation_window=stimulation_window)
            to_zip = [processed_dict['signals'],
                      processed_dict['background'],
                      processed_dict['masks'],
                      processed_dict['images'],
                      processed_dict['nuclei_positive']]
            well_id = get_well_id(vid_file)
            fov_num = 0
            for all_sig, bg_sig, all_m, images_to_save, nb in zip(*to_zip):
                # nb: Nuclei Boolean array
                if all_sig is None:
                    print(f'No signal for {well_id} {fov_num}')
                    fov_num += 1
                else:
                    fov_group_name = check_zarr_group_existance(zarr_root, f'/well_{well_id}/fov_{fov_num}')
                    fov_group = zarr_root[f'well_{well_id}'].create_group(fov_group_name)
                    fov_group.create_dataset('raw_signal', data=np.array(all_sig))  # shape: [num_cells, time]
                    fov_group.create_dataset('background', data=np.array(bg_sig))  # shape: [time]
                    fov_group.create_dataset('mask', data=np.array(all_m))  # shape: [num_cells+1, y, x]
                    fov_group.create_dataset('gcamp_image', data=images_to_save[0])  # shape: [y, x]
                    fov_group.create_dataset('nuclei_image', data=images_to_save[1])  # shape: [y, x]
                    fov_group.create_dataset('contains_nuclei', data=np.array(nb))  # shape: [num_cells]
                    fov_num += 1
    else:
        FITC_files = np.array(sorted([file for file in file_list if 'DAPI' not in file]))
        # Adding data to zarr file
        for vid_file in tqdm(FITC_files): #tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable
            processed_dict = extract_nd2_signal(vid_file,
                                                ff_image,
                                                cp_model=cp_model,
                                                cp_model_nuclei=cp_model_nuclei,
                                                nuclei_file=None,
                                                verbose=verbose,
                                                background_noise_threshold=background_noise_threshold)
            well_id = get_well_id(vid_file)
            fov_num = 0
            to_zip = [processed_dict['signals'],
                        processed_dict['background'],
                        processed_dict['masks'],
                        processed_dict['images']]
            for all_sig, bg_sig, all_m, images_to_save in zip(*to_zip):
                if all_sig is None:
                    print(f'No signal for {well_id} {fov_num}')
                    fov_num+=1
                else:
                    fov_group_name = check_zarr_group_existance(zarr_root, f'/well_{well_id}/fov_{fov_num}')
                    fov_group = zarr_root[f'well_{well_id}'].create_group(fov_group_name)
                    fov_group.create_dataset('raw_signal', data=np.array(all_sig))
                    fov_group.create_dataset('background', data=np.array(bg_sig))
                    fov_group.create_dataset('mask', data=np.array(all_m))
                    fov_group.create_dataset('gcamp_image', data=images_to_save[0])
                    fov_num+=1
    # Going through the zarr file to calculate photobleaching curve to extract
    normalize_raw_signals(zarr_root)


###########################################
###########################################
## Defining Main
###########################################
###########################################


def main():
    # Setting up CLI
    parser = ArgumentParser()
    parser.add_argument("--file_directory",
                        type=str,
                        help="Path to the location with the video files. Note: Currently only supports ND2 files.")
    parser.add_argument("--zarr_file",
                        type=str,
                        help="The location where the zarr file will be saved.")
    parser.add_argument("--find_nuclei",
                        action="store_true",
                        help="If set, will find nuclei for downstream analysis.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="If set, will print out additional information.")
    parser.add_argument("--background_noise_threshold",
                        type=float,
                        default=0.12,
                        help="The minimum mean difference required for \
                            a signal to be considered different from the background.")
    parser.add_argument("--stimulation",
                        action="store_true",
                        help="If set, this will only take the first and last windows of the timeseries to calculate the signal.")
    parser.add_argument("--stimulation_window",
                        type=int,
                        default=1200,
                        help="The number of frames to take for the stimulation window. Only used if --stimulation is set.")
                
    args = parser.parse_args()
    # Reading wishlist genes into list
    file_location = args.file_directory
    nd2_files = collate_files(file_location, '.nd2')
    FITC_files = np.array([file for file in nd2_files if 'DAPI' not in file])
    ff_image = calculate_flatfield_image(FITC_files)
    zarr_root = zarr.open(f'{args.zarr_file}', 'w')
    if args.stimulation:
        stimulation_window = args.stimulation_window
    else:
        stimulation_window = None
    extract_all_signals(zarr_root,
                        nd2_files,
                        ff_image,
                        find_nuclei=args.find_nuclei,
                        verbose=args.verbose,
                        background_noise_threshold=args.background_noise_threshold,
                        stimulation_window=stimulation_window)

if __name__ == "__main__":
    main()
