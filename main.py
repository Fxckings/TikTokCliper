#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime

import config
from utils import (
    setup_logging,
    print_banner,
    get_video_info,
    format_time,
    sanitize_filename
)
from analyzer import analyze_video
from clip_generator import generate_clips

def validate_video_path(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    if not os.path.isfile(video_path):
        raise ValueError(f"Path is not a file: {video_path}")

    valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
    ext = os.path.splitext(video_path)[1].lower()

    if ext not in valid_extensions:
        raise ValueError(
            f"Unsupported video format: {ext}\n"
            f"Supported formats: {', '.join(valid_extensions)}"
        )

    return os.path.abspath(video_path)

def print_video_info(video_path: str, logger):
    try:
        info = get_video_info(video_path)

        logger.info("=" * 60)
        logger.info("VIDEO INFORMATION")
        logger.info("=" * 60)
        logger.info(f"File: {os.path.basename(video_path)}")
        logger.info(f"Resolution: {info['width']}x{info['height']}")
        logger.info(f"Aspect Ratio: {info['aspect_ratio']:.2f}")
        logger.info(f"FPS: {info['fps']:.2f}")
        logger.info(f"Duration: {format_time(info['duration'])}")
        logger.info(f"Total Frames: {info['frame_count']:,}")
        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"Could not retrieve video info: {e}")

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description='Automatically generate TikTok-style clips from long videos'
    )

    parser.add_argument(
        'video',
        help='Path to input video file'
    )

    parser.add_argument(
        '-n', '--clips',
        type=int,
        default=config.NUM_CLIPS,
        help=f'Number of clips to generate (default: {config.NUM_CLIPS})'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=config.OUTPUT_DIR,
        help=f'Output directory for clips (default: {config.OUTPUT_DIR})'
    )

    parser.add_argument(
        '-l', '--language',
        type=str,
        default=None,
        help='Language code for speech recognition (e.g., en, ru) or auto-detect if not specified'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default=config.WHISPER_MODEL,
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help=f'Whisper model size (default: {config.WHISPER_MODEL})'
    )

    parser.add_argument(
        '--min-duration',
        type=int,
        default=config.MIN_CLIP_DURATION,
        help=f'Minimum clip duration in seconds (default: {config.MIN_CLIP_DURATION})'
    )

    parser.add_argument(
        '--max-duration',
        type=int,
        default=config.MAX_CLIP_DURATION,
        help=f'Maximum clip duration in seconds (default: {config.MAX_CLIP_DURATION})'
    )

    parser.add_argument(
        '--no-blur',
        action='store_true',
        help='Disable blurred background (use center crop instead)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    config.NUM_CLIPS = args.clips
    config.OUTPUT_DIR = args.output
    config.WHISPER_MODEL = args.model
    config.MIN_CLIP_DURATION = args.min_duration
    config.MAX_CLIP_DURATION = args.max_duration
    config.USE_BLURRED_BACKGROUND = not args.no_blur

    if args.language:
        config.WHISPER_LANGUAGE = args.language

    if args.verbose:
        config.LOG_LEVEL = 'DEBUG'

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    try:
        video_path = validate_video_path(args.video)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    logger = setup_logging(sanitize_filename(video_basename))

    logger.info("AutoTikTok Clip Generator Started")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Number of clips to generate: {config.NUM_CLIPS}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")

    print_video_info(video_path, logger)

    start_time = time.time()

    try:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 1: Analyzing video")
        logger.info("=" * 60)

        segments = analyze_video(video_path)

        if not segments:
            logger.error("No interesting segments found in video")
            sys.exit(1)

        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Generating clips")
        logger.info("=" * 60)

        clips = generate_clips(video_path, segments, config.NUM_CLIPS)

        if not clips:
            logger.error("No clips were generated")
            sys.exit(1)

        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generated {len(clips)} clips in {elapsed_time:.1f} seconds")
        logger.info(f"Output directory: {os.path.abspath(config.OUTPUT_DIR)}")
        logger.info("")
        logger.info("Generated clips:")

        for i, clip_path in enumerate(clips, 1):
            logger.info(f"  {i}. {os.path.basename(clip_path)}")

        logger.info("")
        logger.info("✓ All done! Your clips are ready to upload to TikTok!")

    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nError during processing: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
