# plexus-extract
**Overview**: Tools for extracting neuronal activity timeseries from microscopy videos. Generates Zarr files from .nd2 file containing videos.
If interested in expanding the usage to other file types other than Nikon nd2 files you will need to update the ND2Video and ND2Image classes.

# Installation
```bash
git clone https://github.com/pgrosjean/plexus-extract.git
cd plexus-extract
bash setup_mamba.sh
conda activate plexus_extract
pip install -e .
```

# Hardware Specifications
Requires at least one NVIDIA GPU with at least 24 Gb memory for deep learning segmentation with CellPose.

# Usage (CLI)
### Viewing the flags for plexus-extract
```bash
# view the flags for plexus-extract
plexus-extract --help
```

### Example Usage
```bash
# Extraction where there are video and nuclei image nd2 files (used for telling if CRISPRi guides were delivered)
gl-extract --file_directory /home/nd2_files_plate_1/ --zarr_file /home/zarr_files/plate1.zarr --find_nuclei --background_noise_threshold 0.03
```

```bash
# Extraction where there are only nd2 files with videos
gl-extract --file_directory /home/nd2_files_plate_1/ --zarr_file /home/zarr_files/plate1.zarr --background_noise_threshold 0.03
```

