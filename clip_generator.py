import os
import logging
from typing import Any, List, Dict, Optional
from pathlib import Path
import numpy as np
import cv2
from moviepy import VideoFileClip, CompositeVideoClip, TextClip, vfx
from tqdm import tqdm

import config
from subtitle_generator import SubtitleGenerator
from utils import ensure_clip_duration, sanitize_filename, format_time

logger = logging.getLogger("AutoTikTok")

def find_system_font() -> Optional[str]:
    font_paths = []

    if os.name == 'posix':
        font_paths.extend([
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/SFCompact.ttf',
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
            '/Library/Fonts/Arial Bold.ttf',
            '/Library/Fonts/Arial.ttf',
        ])

    elif os.name == 'nt':
        font_paths.extend([
            'C:/Windows/Fonts/arialbd.ttf',
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/verdanab.ttf',
        ])

    else:
        font_paths.extend([
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        ])

    for font_path in font_paths:
        if os.path.exists(font_path):
            logger.info(f"Using font: {font_path}")
            return font_path

    logger.warning("No system font found, using moviepy default")
    return None

class ClipGenerator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.subtitle_generator = SubtitleGenerator()
        self.font_path = config.SUBTITLE_FONT or find_system_font()

    def create_vertical_clip(
        self,
        clip: VideoFileClip,
        target_width: int = None,
        target_height: int = None
    ) -> VideoFileClip:
        target_width = target_width or config.TARGET_WIDTH
        target_height = target_height or config.TARGET_HEIGHT

        current_width, current_height = clip.size
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        logger.debug(
            f"Converting {current_width}x{current_height} "
            f"(aspect: {current_aspect:.2f}) to "
            f"{target_width}x{target_height} (aspect: {target_aspect:.2f})"
        )

        if current_aspect > target_aspect:
            if config.USE_BLURRED_BACKGROUND:
                return self._create_blurred_background_clip(
                    clip, target_width, target_height
                )
            else:
                new_width = int(current_height * target_aspect)
                x_center = current_width / 2
                x1 = int(x_center - new_width / 2)

                clipped = clip.cropped(x1=x1, width=new_width)
                resized = clipped.with_effects([vfx.Resize(height=target_height)])
                return resized

        else:
            new_height = int(current_width / target_aspect)
            y_center = current_height / 2
            y1 = int(y_center - new_height / 2)

            clipped = clip.cropped(y1=y1, height=new_height)
            resized = clipped.with_effects([vfx.Resize(height=target_height)])
            return resized

    def _create_blurred_background_clip(
        self,
        clip: VideoFileClip,
        target_width: int,
        target_height: int
    ) -> CompositeVideoClip:
        background = clip.with_effects([vfx.Resize(height=target_height)])

        def blur_and_darken(image):
            blurred = cv2.GaussianBlur(
                image,
                (config.BLUR_STRENGTH, config.BLUR_STRENGTH),
                0
            )
            darkened = (blurred * config.BACKGROUND_OPACITY).astype('uint8')
            return darkened

        background = background.image_transform(blur_and_darken)

        main_clip = clip.with_effects([vfx.Resize(height=target_height)])
        main_clip = main_clip.with_position("center")

        final_clip = CompositeVideoClip(
            [background, main_clip],
            size=(target_width, target_height)
        )

        return final_clip

    def _create_animated_subtitle(
        self,
        text: str,
        start: float,
        duration: float,
        clip_width: int,
        clip_height: int
    ) -> Optional[TextClip]:
        try:
            txt_clip = TextClip(
                text=text,
                font=self.font_path,
                font_size=config.SUBTITLE_FONT_SIZE,
                color=config.SUBTITLE_FONT_COLOR,
                stroke_color=config.SUBTITLE_STROKE_COLOR,
                stroke_width=config.SUBTITLE_STROKE_WIDTH,
                method='caption',
                size=(int(clip_width * 0.9), None),
                text_align='center',
                interline=-5
            )

            y_position = clip_height - config.SUBTITLE_BOTTOM_MARGIN - txt_clip.h

            if config.SUBTITLE_ANIMATION_ENABLED:
                animation_type = config.SUBTITLE_ANIMATION_TYPE

                if animation_type == "slide_up":
                    def position_func(t):
                        if t < 0.3:
                            progress = t / 0.3
                            offset = 50 * (1 - progress)
                            return ('center', y_position + offset)
                        return ('center', y_position)

                    txt_clip = txt_clip.with_position(position_func)

                elif animation_type == "fade_in":
                    def opacity_func(t):
                        if t < 0.3:
                            return t / 0.3
                        return 1.0

                    txt_clip = txt_clip.with_position(('center', y_position))
                    txt_clip = txt_clip.with_effects([
                        lambda clip: clip.with_opacity(lambda t: opacity_func(t))
                    ])

                elif animation_type == "zoom_in":
                    def size_func(t):
                        if t < 0.3:
                            scale = 0.8 + (0.2 * t / 0.3)
                            return scale
                        return 1.0

                    original_size = txt_clip.size

                    def resize_func(get_frame, t):
                        scale = size_func(t)
                        if scale == 1.0:
                            return get_frame(t)
                        new_w = int(original_size[0] * scale)
                        new_h = int(original_size[1] * scale)
                        frame = get_frame(t)
                        return cv2.resize(frame, (new_w, new_h))

                    txt_clip = txt_clip.with_position(('center', y_position))

            else:
                txt_clip = txt_clip.with_position(('center', y_position))

            txt_clip = txt_clip.with_start(start).with_duration(duration)

            return txt_clip

        except Exception as e:
            logger.warning(f"Error creating animated subtitle: {e}")
            return None

    def add_subtitles(
        self,
        clip: VideoFileClip,
        subtitles: List[Dict]
    ) -> CompositeVideoClip:
        if not subtitles:
            return clip

        subtitle_clips = []
        clip_duration = clip.duration

        logger.info(f"Creating {len(subtitles)} animated subtitle clips...")

        for i, subtitle in enumerate[Dict](subtitles):
            text = subtitle['text']
            start = subtitle['start']
            duration = subtitle['duration']

            if start >= clip_duration:
                logger.debug(f"Skipping subtitle at {start:.2f}s (clip duration: {clip_duration:.2f}s)")
                continue

            if start + duration > clip_duration:
                old_duration = duration
                duration = clip_duration - start
                logger.debug(f"Trimming subtitle duration from {old_duration:.2f}s to {duration:.2f}s")

            txt_clip = self._create_animated_subtitle(
                text, start, duration, clip.w, clip.h
            )

            if txt_clip:
                subtitle_clips.append(txt_clip)

        if subtitle_clips:
            logger.info(f"Compositing video with {len(subtitle_clips)} subtitles...")
            final = CompositeVideoClip([clip] + subtitle_clips)
            return final

        return clip

    def generate_clip(
        self,
        segment: Dict,
        output_filename: str,
        clip_index: int = 0
    ) -> str:
        start_time = segment['start']
        end_time = segment['end']

        video_clip = VideoFileClip(self.video_path)
        video_duration = video_clip.duration
        video_clip.close()

        start_time, end_time = ensure_clip_duration(
            start_time, end_time, video_duration
        )

        logger.info(
            f"Generating clip #{clip_index + 1}: "
            f"{format_time(start_time)} - {format_time(end_time)} "
            f"(duration: {end_time - start_time:.1f}s, score: {segment['score']:.3f})"
        )

        try:
            os.environ['FFREPORT'] = 'level=error'
            logger.info(f"Loading video segment...")
            clip = VideoFileClip(self.video_path, audio=True, fps_source='fps').subclipped(start_time, end_time)

            logger.info("Converting to vertical format...")
            clip = self.create_vertical_clip(clip)

            logger.info("Generating subtitles...")
            subtitles = self.subtitle_generator.generate_subtitles(
                self.video_path,
                start_time=start_time,
                end_time=end_time
            )

            if subtitles:
                logger.info(f"Adding {len(subtitles)} subtitle segments...")
                clip = self.add_subtitles(clip, subtitles)

            timestamp = format_time(start_time).replace(':', '-')
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"{output_filename}_clip_{clip_index + 1}_{timestamp}.mp4"
            )

            logger.info(f"Exporting clip to: {output_path}")
            clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate=config.OUTPUT_BITRATE,
                fps=config.OUTPUT_FPS,
                preset='ultrafast',
                threads=2,
                logger='bar',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                write_logfile=False,
                ffmpeg_params=[
                    '-strict', '-2',
                    '-err_detect', 'ignore_err',
                    '-flags', '+global_header',
                    '-fflags', '+genpts+igndts',
                    '-max_error_rate', '1.0'
                ]
            )

            clip.close()

            logger.info(f"✓ Clip saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating clip: {e}")
            raise

    def generate_clips(
        self,
        segments: List[Dict],
        num_clips: int = None
    ) -> List[str]:
        num_clips = num_clips or config.NUM_CLIPS

        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_filename = sanitize_filename(video_basename)

        top_segments = segments[:min(len(segments), num_clips)]

        logger.info(f"Generating {len(top_segments)} clips...")

        self.subtitle_generator.load_model()

        generated_clips = []

        for i, segment in enumerate(top_segments):
            try:
                logger.info(f"Processing clip {i+1}/{len(top_segments)}...")
                clip_path = self.generate_clip(segment, output_filename, i)
                generated_clips.append(clip_path)
                logger.info(f"Progress: {len(generated_clips)}/{len(top_segments)} clips completed")
            except Exception as e:
                logger.error(f"Failed to generate clip #{i + 1}: {e}")
                continue

        logger.info(f"✓ Generated {len(generated_clips)} clips successfully")
        return generated_clips

def generate_clips(video_path: str, segments: List[Dict], num_clips: int = None) -> List[str]:
    generator = ClipGenerator(video_path)
    return generator.generate_clips(segments, num_clips)
