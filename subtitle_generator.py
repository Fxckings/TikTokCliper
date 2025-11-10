import os
import logging
from typing import Any, List, Dict, Tuple
import whisper
import torch
from tqdm import tqdm
from moviepy import VideoFileClip
import tempfile

import config

logger = logging.getLogger("AutoTikTok")

class SubtitleGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.WHISPER_MODEL
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading Whisper model: {self.model_name}")

        try:
            device = "cuda" if config.USE_GPU and torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.model = whisper.load_model(self.model_name, device=device)
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def transcribe(
        self,
        video_path: str,
        start_time: float = None,
        end_time: float = None,
        language: str = None
    ) -> Dict:
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio from video...")

        try:
            options = {
                "language": language or config.WHISPER_LANGUAGE,
                "task": "transcribe",
                "verbose": False,
                "fp16": torch.cuda.is_available(),
                "word_timestamps": True,
            }

            result = self.model.transcribe(
                video_path,
                **options
            )

            if start_time is not None or end_time is not None:
                filtered_segments = []
                for segment in result['segments']:
                    seg_start = segment['start']
                    seg_end = segment['end']

                    if start_time is not None and seg_end < start_time:
                        continue
                    if end_time is not None and seg_start > end_time:
                        continue

                    if start_time is not None:
                        segment['start'] = max(0, seg_start - start_time)
                        segment['end'] = seg_end - start_time

                    filtered_segments.append(segment)

                result['segments'] = filtered_segments

            logger.info(f"Transcription complete: {len(result['segments'])} segments")
            return result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {'text': '', 'segments': []}

    def format_word_level_subtitles(
        self,
        segments: List[Dict],
        start_offset: float = 0.0
    ) -> List[Dict]:
        subtitles = []
        words_per_group = config.SUBTITLE_WORDS_PER_GROUP
        max_chars = config.SUBTITLE_MAX_CHARS_PER_LINE

        for segment in segments:
            if 'words' not in segment or not segment['words']:
                continue

            words = segment['words']

            i = 0
            while i < len(words):
                word_group = []
                group_start = words[i].get('start', segment['start'])

                while i < len(words) and len(word_group) < words_per_group:
                    word = words[i]['word'].strip()
                    if word:
                        word_group.append(word)
                    i += 1

                if not word_group:
                    continue

                group_end = words[i-1].get('end', segment['end'])

                text = ' '.join(word_group)

                if len(text) > max_chars:
                    lines = self._smart_line_break(word_group, max_chars)
                    text = '\n'.join(lines)

                subtitles.append({
                    'text': text,
                    'start': group_start - start_offset,
                    'end': group_end - start_offset,
                    'duration': group_end - group_start
                })

        return subtitles

    def _smart_line_break(self, words: List[str], max_chars: int) -> List[str]:
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word)

            if current_length + word_len + (1 if current_line else 0) > max_chars:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_len
                else:
                    lines.append(word)
            else:
                current_line.append(word)
                current_length += word_len + (1 if len(current_line) > 1 else 0)

        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else ['']

    def format_subtitles(
        self,
        segments: List[Dict],
        max_chars_per_line: int = None
    ) -> List[Dict]:
        if segments and len(segments) > 0 and 'words' in segments[0]:
            return self.format_word_level_subtitles(segments)

        max_chars = max_chars_per_line or config.SUBTITLE_MAX_CHARS_PER_LINE
        subtitles = []

        for segment in segments:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']

            if len(text) > max_chars:
                words = text.split()
                lines = self._smart_line_break(words, max_chars)
                text = '\n'.join(lines)

            subtitles.append({
                'text': text,
                'start': start,
                'end': end,
                'duration': end - start
            })

        return subtitles

    def generate_subtitles(
        self,
        video_path: str,
        start_time: float = None,
        end_time: float = None,
        language: str = None
    ) -> List[Dict]:
        temp_file = None
        actual_video_path = video_path
        original_start = start_time
        original_end = end_time

        if start_time is not None and end_time is not None:
            try:
                logger.info(f"Extracting audio segment from {start_time:.2f}s to {end_time:.2f}s")

                video = VideoFileClip(video_path)
                segment = video.subclipped(start_time, end_time)

                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                temp_audio_path = temp_file.name
                temp_file.close()

                segment.audio.write_audiofile(
                    temp_audio_path,
                    codec='libmp3lame',
                    bitrate='192k',
                    fps=44100,
                    logger=None
                )

                segment.close()
                video.close()

                actual_video_path = temp_audio_path
                start_time = None
                end_time = None

            except Exception as e:
                logger.error(f"Error extracting audio segment: {e}")
                if temp_file:
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                actual_video_path = video_path

        try:
            result = self.transcribe(actual_video_path, start_time, end_time, language)
            subtitles = self.format_subtitles(result['segments'])

            if subtitles and temp_file and original_start is not None and original_end is not None:
                clip_duration = original_end - original_start
                subtitles = [s for s in subtitles if s['start'] < clip_duration]
                logger.info(f"Filtered subtitles to {len(subtitles)} segments within {clip_duration:.2f}s duration")

            return subtitles

        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def save_srt(self, subtitles: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate[Dict](subtitles, 1):
                f.write(f"{i}\n")

                start = self._format_timestamp(subtitle['start'])
                end = self._format_timestamp(subtitle['end'])
                f.write(f"{start} --> {end}\n")

                f.write(f"{subtitle['text']}\n\n")

        logger.info(f"Saved SRT subtitles to: {output_path}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_subtitles(
    video_path: str,
    start_time: float = None,
    end_time: float = None,
    language: str = None
) -> List[Dict]:
    generator = SubtitleGenerator()
    return generator.generate_subtitles(video_path, start_time, end_time, language)
