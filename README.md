# BVSR
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Batch Video Size Reduction (BVSR) using FFMPEG

A python script to reduce the size of all videos in a folder while keeping its exact structure.

## Requirements
+ Python 3
+ [FFMPEG](https://ffmpeg.org/download.html) 

## Installation
```bash
pip install bvsr
```
or clone from source
```bash
git clone https://github.com/abdeladim-s/bvsr && cd bvsr
```
##

## Usage
```bash
usage: bvsr.py [-h] [--destination-folder DESTINATION_FOLDER] [--version] [--ffmpeg-exec FFMPEG_EXEC] [--encoder ENCODER] [-i]
               [--crf CRF | --video-quality VIDEO_QUALITY | --target-size TARGET_SIZE] [--audio-quality AUDIO_QUALITY]
               source_folder

positional arguments:
  source_folder         The Source folder of the videos

options:
  -h, --help            show this help message and exit
  --destination-folder DESTINATION_FOLDER
                        The directory where the output videos will be stored, default to the same folder name with `bvsr` suffix in the
                        parent directory
  --version             show program's version number and exit
  --ffmpeg-exec FFMPEG_EXEC
                        The ffmpeg executable file, default to `ffmpeg`
  --encoder ENCODER     The video encoder name
  -i, --ignore-other-files
                        Ignore the other non-video files, the default operation is to copy the other files to the target folder to keep the
                        same source folder structure
  --crf CRF             Target Constant Rate Factor (CRF) value (RECOMMENDED)[More info at: https://trac.ffmpeg.org/wiki/Encode/H.264]
  --video-quality VIDEO_QUALITY
                        Target video quality. Available qualities: ['videophone', 'videoconferencing', '240p', '360p', '480p', 'VCD', '720p',
                        '720p60', '1080p', '1080p60']
  --target-size TARGET_SIZE
                        Target upper bound video size (in MB)
  --audio-quality AUDIO_QUALITY
                        Target audio quality. Default to the audio quality of the source video. Available qualities: ['low', 'mid-range',
                        'medium', 'high', 'highest']

```

## Examples

+ The **recommended** way to reduce the size of a video file is to use the [Constant Rate Factor (CRF)](https://trac.ffmpeg.org/wiki/Encode/H.264):
```bash
python bvsr --crf 34 /path/to/the/source_folder
```
This will output the results in a folder in the parent directory with the same name of your `source_folder` suffixed with `_bvsr`. 
The output folder will have the same structure as the `source_folder` (i.e. processing the video files and just copying any other files. Use `--ignore-other-files` to ignore them instead).

## 

+ If you want to specify a video quality rather than using the CRF:
```bash
python bvsr --video-quality 480p /path/to/the/source_folder
```
[Available qualities](https://en.wikipedia.org/wiki/Bit_rate): `['videophone', 'videoconferencing', '240p', '360p', '480p', 'VCD', '720p',
                        '720p60', '1080p', '1080p60']`

+ If you just care about the size and not the quality, you can specify a  target size in `MB` directly:
```bash
python bvsr --target-size 100 --destination-folder /path/to/destination_folder
```
_Although it is not guaranteed._ 

## License

GPLv3 © [bvsr](https://github.com/abdeladim-s/bvsr)
