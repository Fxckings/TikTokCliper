import os
import logging
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import cv2
from colorama import Fore, Style, init

import config

init(autoreset=True)

def setup_logging(video_name: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"{video_name}_{timestamp}.log")

    logger = logging.getLogger("AutoTikTok")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_formatter = ColoredFormatter(config.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def calculate_aspect_ratio(width: int, height: int) -> float:
    return width / height if height > 0 else 0

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    info['aspect_ratio'] = calculate_aspect_ratio(info['width'], info['height'])

    cap.release()
    return info

def calculate_motion_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return np.mean(magnitude)

def normalize_score(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores

    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score - min_score == 0:
        return np.ones_like(scores) * 0.5

    return (scores - min_score) / (max_score - min_score)

def merge_overlapping_segments(segments: list, max_gap: float = 2.0) -> list:
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]

        if current_start <= last_end + max_gap:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged

def ensure_clip_duration(
    start: float,
    end: float,
    video_duration: float,
    min_duration: float = config.MIN_CLIP_DURATION,
    max_duration: float = config.MAX_CLIP_DURATION,
    preferred_duration: float = config.PREFERRED_CLIP_DURATION
) -> Tuple[float, float]:
    duration = end - start

    if duration < preferred_duration:
        target_duration = min(preferred_duration, video_duration)
        deficit = target_duration - duration

        extend_each = deficit / 2
        new_start = max(0, start - extend_each)
        new_end = min(video_duration, end + extend_each)

        actual_deficit = target_duration - (new_end - new_start)
        if actual_deficit > 0:
            if new_start == 0:
                new_end = min(video_duration, new_end + actual_deficit)
            elif new_end == video_duration:
                new_start = max(0, new_start - actual_deficit)
            else:
                new_start = max(0, new_start - actual_deficit / 2)
                new_end = min(video_duration, new_end + actual_deficit / 2)

        start, end = new_start, new_end

    elif duration > max_duration:
        new_duration = preferred_duration
        excess = duration - new_duration

        start = start + excess / 2
        end = end - excess / 2

    start = max(0, start)
    end = min(video_duration, end)

    return start, end

def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    filename = filename.strip(' .')

    return filename

def print_banner():
    banner = f"""{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              🎬  AutoTikTok Clip Generator  🎬           ║
║                                                          ║
║         Automatically generate TikTok-style clips       ║
║              from long-form video content                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""
    print(banner)

def print_progress(stage: str, current: int, total: int):
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r{Fore.CYAN}{stage}: {Fore.GREEN}[{bar}] {percentage:.1f}% ({current}/{total}){Style.RESET_ALL}", end='', flush=True)

    if current == total:
        print()
