# BVSR
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Batch Video Size Reduction (BVSR) using [ffmpeg](https://ffmpeg.org).

A python script to reduce the size of all videos in a folder while keeping its exact structure.

## Requirements
+ Python 3
+ [ffmpeg](https://ffmpeg.org/download.html) 

## Installation
```bash
pip install bvsr
```
or clone from source
```bash
git clone https://github.com/abdeladim-s/bvsr && cd bvsr
pip install -r requirements.txt
```
##

## Usage
```bash
usage: bvsr.py [-h] [--version] [--destination-folder DESTINATION_FOLDER]
               [--crf CRF | --video-quality VIDEO_QUALITY | --target-size TARGET_SIZE] [--audio-quality AUDIO_QUALITY]
               [--ffmpeg-exec FFMPEG_EXEC] [--encoder ENCODER] [-i]
               source_folder

positional arguments:
  source_folder         The Source folder of the videos

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --destination-folder DESTINATION_FOLDER
                        The directory where the output videos will be stored, default to the same folder name with `bvsr` suffix in the
                        parent directory
  --crf CRF             Target Constant Rate Factor (CRF) value (RECOMMENDED)[More info at: https://trac.ffmpeg.org/wiki/Encode/H.264]
  --video-quality VIDEO_QUALITY
                        Target video quality. Available qualities: ['videophone', 'videoconferencing', '240p', '360p', '480p', 'VCD', '720p',
                        '720p60', '1080p', '1080p60']
  --target-size TARGET_SIZE
                        Target upper bound video size (in MB)
  --audio-quality AUDIO_QUALITY
                        Target audio quality. Default to the audio quality of the source video. Available qualities: ['low', 'mid-range',
                        'medium', 'high', 'highest']
  --ffmpeg-exec FFMPEG_EXEC
                        The ffmpeg executable file, default to `ffmpeg`
  --encoder ENCODER     The video encoder name
  -i, --ignore-other-files
                        Ignore the other non-video files, the default operation is to copy the other files to the target folder to keep the
                        same source folder structure


```

## Examples

+ The **recommended** way to reduce the size of a video file is to use the [Constant Rate Factor (CRF)](https://trac.ffmpeg.org/wiki/Encode/H.264):
> The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible. A lower value generally leads to higher quality, and a subjectively sane range is 17–28. Consider 17 or 18 to be visually lossless or nearly so; it should look the same or nearly the same as the input but it isn't technically lossless.
The range is exponential, so increasing the CRF value +6 results in roughly half the bitrate / file size, while -6 leads to roughly twice the bitrate.

>Choose the highest CRF value that still provides an acceptable quality. If the output looks good, then try a higher value. If it looks bad, choose a lower value.

Run the following command to use a CRF value of 34 for example:
```bash
python bvsr --crf 34 /path/to/the/source_folder
```

This will output the results in a folder in the parent directory with the same name of your `source_folder` suffixed with `_bvsr`. 
The output folder will have the same structure as the `source_folder` (i.e. processing the video files and just copying any other file. Use `--ignore-other-files` to ignore them instead).

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

GPLv3 © [Batch Video Size Reduction (BVSR)](https://github.com/abdeladim-s/bvsr)
