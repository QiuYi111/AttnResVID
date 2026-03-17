# This script shows how to load this in Python using PyNWB and LINDI
# It assumes you have installed PyNWB and LINDI (pip install pynwb lindi)

import pynwb
import lindi

local_cache = lindi.LocalCache()

# Load https://api.dandiarchive.org/api/assets/bed42c25-4cd4-4eb0-a93a-dda87cc68b17/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001413/assets/bed42c25-4cd4-4eb0-a93a-dda87cc68b17/nwb.lindi.json", local_cache=local_cache)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

nwb.session_description # (str) Calcium imaging in SMA during the arm reaching condition
nwb.identifier # (str) 97b8b793-0cd9-4d05-a836-80a2dd6ee1ec
nwb.session_start_time # (datetime) 2023-03-28T11:18:37.534000+00:00
nwb.file_create_date # (datetime) 2025-04-14T19:40:22.327883+00:00
nwb.timestamps_reference_time # (datetime) 2023-03-28T11:18:37.534000+00:00
nwb.experimenter # (List[str]) []
nwb.experiment_description # (str) 
nwb.institution # (str) 
nwb.keywords # (List[str]) []
nwb.protocol # (str) 
nwb.lab # (str) 
nwb.subject # (Subject)
nwb.subject.age # (str) P4Y
nwb.subject.age__reference # (str) birth
nwb.subject.description # (str) 
nwb.subject.genotype # (str) 
nwb.subject.sex # (str) M
nwb.subject.species # (str) Macaca mulatta
nwb.subject.subject_id # (str) U
nwb.subject.weight # (str) 
nwb.subject.date_of_birth # (datetime) 

OnePhotonSeries = nwb.acquisition["OnePhotonSeries"] # (OnePhotonSeries) Miniscope imaging data
OnePhotonSeries.imaging_plane # (ImagingPlane)
OnePhotonSeries.data # (h5py.Dataset) shape [7497, 1280, 800] [ num_frames, num_rows, num_columns ]; dtype <u2
OnePhotonSeries.starting_time # 0 sec
OnePhotonSeries.rate # 10 Hz

ophys = nwb.processing["ophys"] # (ProcessingModule) Optical physiology data obtained by processing raw calcium imaging data

EventAmplitude = nwb.processing["ophys"]["EventAmplitude"] # (RoiResponseSeries) Amplitude of neural events associated with spatial footprints
EventAmplitude.data # (h5py.Dataset) shape [7497, 14]; dtype <f8
EventAmplitude.starting_time # 0 sec
EventAmplitude.rate # 10.00376786792083 Hz

Fluorescence = nwb.processing["ophys"]["Fluorescence"] # (Fluorescence) 

RoiResponseSeries = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"] # (RoiResponseSeries) Fluorescence data associated with spatial footprints
RoiResponseSeries.data # (h5py.Dataset) shape [7497, 14]; dtype <f4
RoiResponseSeries.starting_time # 0 sec
RoiResponseSeries.rate # 10 Hz

ImageSegmentation = nwb.processing["ophys"]["ImageSegmentation"] # (ImageSegmentation) 

PlaneSegmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"] # (PlaneSegmentation) Footprints of individual cells obtained by segmenting the field of view
PlaneSegmentation["image_mask"].data # (h5py.Dataset) shape [14, 318, 198] [ num_masks, num_rows, num_columns ]; dtype <f4

