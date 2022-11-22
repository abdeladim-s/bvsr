#!/usr/bin/env python
# -*- coding: utf-8 -*-

VIDEO_BITRATES = {
    # https://en.wikipedia.org/wiki/Bit_rate
    'videophone': 16_000,  # videophone quality (minimum necessary for a consumer-acceptable "talking head" picture)
    'videoconferencing': 256_000,  # business-oriented videoconferencing quality using video compression
    '240p': 400_000,  # YouTube 240p videos (using H.264)[21]
    '360p': 750_000,  # YouTube 360p videos (using H.264)[21]
    '480p': 1_000_000,  # YouTube 480p videos (using H.264)[21]
    'VCD': 1_150_000,  # 1.15 Mbit/s max – VCD quality (using MPEG1 compression)[22]
    '720p': 2_500_000,  # 2.5 Mbit/s YouTube 720p videos (using H.264)[21]
    '720p60': 3_800_000,  # 3.8 Mbit/s YouTube 720p60 (60 FPS) videos (using H.264)[21]
    '1080p': 4_500_000,  # 4.5 Mbit/s YouTube 1080p videos (using H.264)[21]
    '1080p60': 6_800_000,  # 6.8 Mbit/s YouTube 1080p60 (60 FPS) videos (using H.264)[21]
}

AUDIO_BITRATES = {
    # https://en.wikipedia.org/wiki/Bit_rate
    'low': 96_000,  # generally used for speech or low-quality streaming
    'mid-range': 128_000,  # mid-range bitrate quality
    'medium': 192_000,  # medium quality bitrate
    'high': 256_000,  # a commonly used high-quality bitrate
    'highest': 320_000,  # highest level supported by the MP3 standard
}


if __name__ == '__main__':
    pass
