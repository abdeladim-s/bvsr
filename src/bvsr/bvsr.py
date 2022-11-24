#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Batch Video Size Reduction (BVSR) using ffmpeg.

A python script to reduce the size of all videos in a source folder while keeping the same folder structure.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

Resources and credits:
https://unix.stackexchange.com/questions/28803/how-can-i-reduce-a-videos-size-with-ffmpeg/289322#289322
https://slhck.info/video/2017/02/24/crf-guide.html
https://trac.ffmpeg.org/wiki/Encode/H.264
https://en.wikipedia.org/wiki/Bit_rate
"""

from pathlib import Path
from pprint import pprint

from tqdm import tqdm
import argparse
import contextlib
import ffmpeg
import gevent
import gevent.monkey;
import socket
import sys
import tempfile
import shutil
import os
import mimetypes
from os.path import exists
import importlib.metadata

gevent.monkey.patch_all(thread=False)

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2022, "
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = __author__
__github__ = "https://github.com/abdeladim-s/bvsr"
__version__ = importlib.metadata.version('bvsr')


__header__ = f"""

██████╗ ██╗   ██╗███████╗██████╗ 
██╔══██╗██║   ██║██╔════╝██╔══██╗
██████╔╝██║   ██║███████╗██████╔╝
██╔══██╗╚██╗ ██╔╝╚════██║██╔══██╗
██████╔╝ ╚████╔╝ ███████║██║  ██║
╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝

Batch Video Size Reduction (BVSR) using ffmpeg
Version: {__version__}               
===================================
"""

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


class BVSR:
    """
    Batch Video Size Reduction (BVSR)
    """

    def __init__(self, source_folder,
                 destination_folder=None,
                 destination_folder_suffix="_bvsr",
                 ignore_other_files=False,
                 ffmpeg_exec='ffmpeg',
                 encoder='libx264',
                 crf=23,
                 video_quality=None,
                 target_size=None,
                 audio_quality=None,
                 ):
        self.source_folder = Path(source_folder).absolute()
        self.destination_folder = destination_folder
        self.destination_folder_suffix = destination_folder_suffix
        self.ignore_other_files = ignore_other_files

        self.ffmpeg_exec = ffmpeg_exec
        self.encoder = encoder
        # Size reduction techniques
        self.crf = crf
        self.video_quality = video_quality
        self.target_size = target_size
        self.audio_quality = audio_quality

    def process_video(self, video_input, video_output, video_format):
        probe = ffmpeg.probe(video_input)
        # pprint(probe)
        format = probe['format']['format_long_name']
        error_flag = False
        try:
            duration = float(probe['format']['duration'])
            audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
        except KeyError as e:
            # https://stackoverflow.com/questions/34118013/how-to-determine-webm-duration-using-ffprobe
            error_flag = True
            print(f'!!! Unable to get duration and bitrate, Unsupported video type `{format}`')
            return

        if self.audio_quality is not None:
            audio_bitrate = AUDIO_BITRATES[self.audio_quality]

        # pprint(probe)
        video_input = ffmpeg.input(video_input)

        if self.video_quality is not None:
            if error_flag:
                print(f"!!! Unsupported operation with videos of type {format}")
                return
            cmd = {'b:v': VIDEO_BITRATES[self.video_quality], 'b:a': audio_bitrate}
        elif self.target_size is not None:
            if error_flag:
                print(f"!!! Unsupported operation with videos of type {format}")
                return
            # Credits to: https://stackoverflow.com/questions/64430805/how-to-compress-video-to-target-size-by-python
            target_total_bitrate = (self.target_size * 1000 * 1024 * 8) / (1.073741824 * duration)
            # Target video bitrate, in bps.
            video_bitrate = target_total_bitrate - audio_bitrate
            cmd = {'b:v': video_bitrate, 'b:a': audio_bitrate}
        else:
            cmd = {'crf': self.crf}

        self.__ffmpeg(video_input, duration, video_output, video_format, cmd)

    def __ffmpeg(self, video_input, duration, video_output, video_format, cmd):
        with show_progress(duration) as socket_filename:
            # See https://ffmpeg.org/ffmpeg-filters.html#Examples-44
            try:
                out, err = (ffmpeg.output(video_input, video_output, loglevel="quiet", **{'c:v': self.encoder,
                                                                                          **cmd, 'f': video_format})
                            .global_args('-progress', 'unix://{}'.format(socket_filename)).overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True, cmd=self.ffmpeg_exec)
                            )
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(e.stderr, file=sys.stderr)
                # raise e
                raise KeyboardInterrupt

    def run(self):
        try:
            # create the output folder first
            if self.destination_folder is None:
                parts = self.source_folder.parts
                base_folder = self.source_folder.parent
                destination_folder_name = parts[-1] + self.destination_folder_suffix
                destination_folder = (base_folder / destination_folder_name).absolute()
            else:
                destination_folder = Path(self.destination_folder).absolute()

            print("creating folder `{}` ...".format(destination_folder))
            os.makedirs(destination_folder, exist_ok=True)
            # walk through the folder
            for root, dirs, files in os.walk(self.source_folder, topdown=True):
                for name in files:
                    # create the same directory structure in the output folder
                    target_dir = Path(root.replace(str(self.source_folder), str(destination_folder)))
                    os.makedirs(target_dir, exist_ok=True)
                    if exists(os.path.join(target_dir, name)):  # if file already exist in target dir, skip it
                        print(f"Skip file `{target_dir.name}/{name}` (already exists)")
                        continue
                    file_type = mimetypes.guess_type(os.path.join(root, name))[0]
                    if file_type is not None and file_type.startswith('video'):
                        print(f"Video file Found: `{target_dir.name}/{name}`")
                        print("Processing ...")
                        video_path = os.path.join(root, name)
                        video_output = os.path.join(target_dir, name)
                        file_type, file_ext = file_type.split('/')
                        self.process_video(video_path, video_output, file_ext)
                    else:
                        # if ignore other files is true, just skip them
                        if self.ignore_other_files:
                            continue
                        else:
                            # copy the file to the target directory
                            print("Copying file {} ...".format(name))
                            Copy.copy_with_progress(os.path.join(root, name), os.path.join(target_dir, name))
                            print()

            print("Done :D")

        except KeyboardInterrupt:
            # remove the file being process, because it is not finished
            print('Process aborted!')
            print(f"Removing file {os.path.join(target_dir, name)}")
            os.remove(os.path.join(target_dir, name))
            sys.exit(1)


class Copy:

    @staticmethod
    def progress_percentage(perc, width=None):
        # This will only work for python 3.3+ due to use of
        # os.get_terminal_size the print function etc.

        FULL_BLOCK = '█'
        # this is a gradient of incompleteness
        INCOMPLETE_BLOCK_GRAD = ['░', '▒', '▓']

        assert (isinstance(perc, float))
        assert (0. <= perc <= 100.)
        # if width unset use full terminal
        if width is None:
            width = os.get_terminal_size().columns
        # progress bar is block_widget separator perc_widget : ####### 30%
        max_perc_widget = '[100.00%]'  # 100% is max
        separator = ' '
        blocks_widget_width = width - len(separator) - len(max_perc_widget)
        assert (blocks_widget_width >= 10)  # not very meaningful if not
        perc_per_block = 100.0 / blocks_widget_width
        # epsilon is the sensitivity of rendering a gradient block
        epsilon = 1e-6
        # number of blocks that should be represented as complete
        full_blocks = int((perc + epsilon) / perc_per_block)
        # the rest are "incomplete"
        empty_blocks = blocks_widget_width - full_blocks

        # build blocks widget
        blocks_widget = ([FULL_BLOCK] * full_blocks)
        blocks_widget.extend([INCOMPLETE_BLOCK_GRAD[0]] * empty_blocks)
        # marginal case - remainder due to how granular our blocks are
        remainder = perc - full_blocks * perc_per_block
        # epsilon needed for rounding errors (check would be != 0.)
        # based on reminder modify first empty block shading
        # depending on remainder
        if remainder > epsilon:
            grad_index = int((len(INCOMPLETE_BLOCK_GRAD) * remainder) / perc_per_block)
            blocks_widget[full_blocks] = INCOMPLETE_BLOCK_GRAD[grad_index]

        # build perc widget
        str_perc = '%.2f' % perc
        # -1 because the percentage sign is not included
        perc_widget = '[%s%%]' % str_perc.ljust(len(max_perc_widget) - 3)

        # form progressbar
        progress_bar = '%s%s%s' % (''.join(blocks_widget), separator, perc_widget)
        # return progressbar as string
        return ''.join(progress_bar)

    @staticmethod
    def copy_progress(copied, total):
        print('\r' + Copy.progress_percentage(100 * copied / total, width=30), end='')

    @staticmethod
    def copyfile(src, dst, *, follow_symlinks=True):
        """Copy data from src to dst.

        If follow_symlinks is not set and src is a symbolic link, a new
        symlink will be created instead of copying the file it points to.

        """
        if shutil._samefile(src, dst):
            raise shutil.SameFileError("{!r} and {!r} are the same file".format(src, dst))

        for fn in [src, dst]:
            try:
                st = os.stat(fn)
            except OSError:
                # File most likely does not exist
                pass
            else:
                # What about other special files? (sockets, devices...)
                if shutil.stat.S_ISFIFO(st.st_mode):
                    raise shutil.SpecialFileError("`%s` is a named pipe" % fn)

        if not follow_symlinks and os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            size = os.stat(src).st_size
            with open(src, 'rb') as fsrc:
                with open(dst, 'wb') as fdst:
                    Copy.copyfileobj(fsrc, fdst, callback=Copy.copy_progress, total=size)
        return dst

    @staticmethod
    def copyfileobj(fsrc, fdst, callback, total, length=16 * 1024):
        copied = 0
        while True:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            callback(copied, total=total)

    @staticmethod
    def copy_with_progress(src, dst, *, follow_symlinks=True):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        Copy.copyfile(src, dst, follow_symlinks=follow_symlinks)
        shutil.copymode(src, dst)
        return dst


@contextlib.contextmanager
def _tmpdir_scope():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def _do_watch_progress(filename, sock, handler):
    """Function to run in a separate gevent greenlet to read progress
    events from a unix-domain socket."""
    connection, client_address = sock.accept()
    data = b''
    try:
        while True:
            more_data = connection.recv(16)
            if not more_data:
                break
            data += more_data
            lines = data.split(b'\n')
            for line in lines[:-1]:
                line = line.decode()
                parts = line.split('=')
                key = parts[0] if len(parts) > 0 else None
                value = parts[1] if len(parts) > 1 else None
                handler(key, value)
            data = lines[-1]
    except Exception as e:
        raise e
    finally:
        connection.close()


@contextlib.contextmanager
def _watch_progress(handler):
    """Context manager for creating a unix-domain socket and listen for
    ffmpeg progress events.
    The socket filename is yielded from the context manager and the
    socket is closed when the context manager is exited.
    Args:
        handler: a function to be called when progress events are
            received; receives a ``key`` argument and ``value``
            argument. (The example ``show_progress`` below uses tqdm)
    Yields:
        socket_filename: the name of the socket file.
    """
    with _tmpdir_scope() as tmpdir:
        socket_filename = os.path.join(tmpdir, 'sock')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(socket_filename)
            sock.listen(1)
            child = gevent.spawn(_do_watch_progress, socket_filename, sock, handler)
            try:
                yield socket_filename
            except:
                gevent.kill(child)
                raise


@contextlib.contextmanager
def show_progress(total_duration):
    """Create a unix-domain socket to watch progress and render tqdm
    progress bar."""
    with tqdm(total=round(total_duration, 2)) as bar:
        def handler(key, value):
            if key == 'out_time_ms':
                time = round(float(value) / 1000000., 2)
                bar.update(time - bar.n)
            elif key == 'progress' and value == 'end':
                bar.update(bar.total - bar.n)

        with _watch_progress(handler) as socket_filename:
            yield socket_filename


def main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('source_folder', type=str, help="The Source folder of the videos")

    # Optional args
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    parser.add_argument('--destination-folder', default=None,
                        help='The directory where the output videos will be stored, default to the same folder name '
                             'with `bvsr` suffix in the parent directory')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--crf', default=23, type=int,
                       help='Target Constant Rate Factor (CRF) value (RECOMMENDED)'
                            '[More info at: https://trac.ffmpeg.org/wiki/Encode/H.264]')
    group.add_argument('--video-quality', default=None,
                       help=f'Target video quality. Available qualities: {list(VIDEO_BITRATES.keys())}')
    group.add_argument('--target-size', default=None, type=float,
                       help='Target upper bound video size (in MB)')
    parser.add_argument('--audio-quality', default=None,
                        help=f'Target audio quality. Default to the audio quality of the source video. '
                             f'Available qualities: {list(AUDIO_BITRATES.keys())}')

    parser.add_argument('--ffmpeg-exec', default='ffmpeg',
                        help='The ffmpeg executable file, default to `ffmpeg`')
    parser.add_argument('--encoder', default='libx264',
                        help='The video encoder name')
    parser.add_argument('-i', '--ignore-other-files', action='store_true',
                        help='Ignore the other non-video files, the default operation is to copy the other files to the'
                             ' target folder to keep the same source folder structure')
    args = parser.parse_args()

    bvsr = BVSR(source_folder=args.source_folder,
                destination_folder=args.destination_folder,
                ffmpeg_exec=args.ffmpeg_exec,
                encoder=args.encoder,
                crf=args.crf,
                video_quality=args.video_quality,
                target_size=args.target_size,
                ignore_other_files=args.ignore_other_files)
    bvsr.run()


if __name__ == '__main__':
    main()
