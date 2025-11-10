import os
import logging
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
from tqdm import tqdm

import config
from utils import calculate_motion_score, normalize_score, format_time

logger = logging.getLogger("AutoTikTok")

class VideoAnalyzer:
    def __init__(self, video_path: str, use_cache: bool = True):
        self.video_path = video_path
        self.scenes = []
        self.audio_data = None
        self.sample_rate = None
        self.use_cache = use_cache
        self.cache_dir = os.path.join(config.TEMP_DIR, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, prefix: str) -> str:
        stat = os.stat(self.video_path)
        cache_key = f"{self.video_path}_{stat.st_mtime}_{stat.st_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prefix}_{cache_hash}.pkl")

    def _load_from_cache(self, cache_path: str) -> Optional[any]:
        if not self.use_cache or not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_path: str, data: any):
        if not self.use_cache:
            return

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def detect_intro_end(self, max_intro_duration: float = 60.0) -> float:
        cache_path = self._get_cache_path('intro_end')
        cached_intro = self._load_from_cache(cache_path)

        if cached_intro is not None:
            logger.info(f"Loaded intro end from cache: {format_time(cached_intro)}")
            return cached_intro

        logger.info("Detecting intro/opening credits...")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        max_frames = min(int(max_intro_duration * fps), total_frames)

        frame_brightness = []
        frame_changes = []
        prev_gray = None

        sample_rate = max(1, int(fps / 5))

        for frame_idx in range(0, max_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90))

            brightness = np.mean(gray_small)
            frame_brightness.append(brightness)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray_small)
                change_score = np.mean(diff)
                frame_changes.append(change_score)
            else:
                frame_changes.append(0)

            prev_gray = gray_small

        cap.release()

        if len(frame_changes) < 10:
            logger.info("Video too short for intro detection, starting from beginning")
            return 0.0

        frame_changes = np.array(frame_changes)
        frame_brightness = np.array(frame_brightness)

        window_size = 20
        smoothed_changes = np.convolve(frame_changes, np.ones(window_size)/window_size, mode='same')

        threshold = np.percentile(smoothed_changes, 70)

        intro_end_frame = 0
        for i in range(len(smoothed_changes)):
            if smoothed_changes[i] > threshold and i > 10:
                intro_end_frame = i
                break

        if intro_end_frame == 0:
            black_threshold = 30
            is_black = frame_brightness < black_threshold

            for i in range(len(is_black)):
                if not is_black[i] and i > 5:
                    intro_end_frame = i
                    break

        intro_end_time = (intro_end_frame * sample_rate) / fps

        intro_end_time = min(intro_end_time, max_intro_duration)

        logger.info(f"Detected intro end at: {format_time(intro_end_time)}")

        self._save_to_cache(cache_path, intro_end_time)

        return intro_end_time

    def find_peak_moment(self, start_time: float, end_time: float) -> float:
        duration = end_time - start_time
        if duration < 5:
            return start_time

        window_size = 2.0
        windows = []
        current = start_time

        while current + window_size <= end_time:
            windows.append((current, current + window_size))
            current += 1.0

        if not windows:
            return start_time

        window_scores = []

        for win_start, win_end in windows:
            motion = self.analyze_motion(win_start, win_end)
            audio = self.analyze_audio_energy(win_start, win_end)

            score = motion + audio
            window_scores.append(score)

        if not window_scores:
            return start_time

        peak_idx = np.argmax(window_scores)
        peak_start, peak_end = windows[peak_idx]

        adjusted_start = max(start_time, peak_start - config.CONTEXT_BEFORE_PEAK)

        return adjusted_start

    def detect_scenes(self) -> List[Tuple[float, float]]:
        cache_path = self._get_cache_path('scenes')
        cached_scenes = self._load_from_cache(cache_path)

        if cached_scenes is not None:
            logger.info(f"Loaded {len(cached_scenes)} scenes from cache")
            self.scenes = cached_scenes
            return cached_scenes

        logger.info("Detecting scenes in video...")

        try:
            video = open_video(self.video_path)

            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=config.SCENE_DETECTION_THRESHOLD)
            )

            scene_manager.detect_scenes(video, show_progress=True)

            scene_list = scene_manager.get_scene_list()

            scenes = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scenes.append((start_time, end_time))

            logger.info(f"Detected {len(scenes)} scenes")
            self.scenes = scenes

            self._save_to_cache(cache_path, scenes)

            return scenes

        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            cap = cv2.VideoCapture(self.video_path)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.scenes = [(0, duration)]
            return self.scenes

    def analyze_motion(self, start_time: float, end_time: float) -> float:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        duration = end_time - start_time
        num_frames = int(duration * fps)

        sample_rate = max(1, int(fps / 5))
        motion_scores = []

        prev_gray = None

        for i in range(0, num_frames, sample_rate):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))

            if prev_gray is not None:
                motion = calculate_motion_score(prev_gray, gray)
                motion_scores.append(motion)

            prev_gray = gray

            for _ in range(sample_rate - 1):
                cap.read()

        cap.release()

        if not motion_scores:
            return 0.0

        return np.mean(motion_scores)

    def load_audio(self) -> Tuple[np.ndarray, int]:
        if self.audio_data is not None:
            return self.audio_data, self.sample_rate

        cache_path = self._get_cache_path('audio')
        cached_audio = self._load_from_cache(cache_path)

        if cached_audio is not None:
            self.audio_data, self.sample_rate = cached_audio
            logger.info(f"Loaded audio from cache: {len(self.audio_data) / self.sample_rate:.2f} seconds")
            return self.audio_data, self.sample_rate

        logger.info("Loading audio from video...")

        try:
            self.audio_data, self.sample_rate = librosa.load(
                self.video_path,
                sr=22050,
                mono=True
            )

            logger.info(f"Audio loaded: {len(self.audio_data) / self.sample_rate:.2f} seconds")

            self._save_to_cache(cache_path, (self.audio_data, self.sample_rate))

            return self.audio_data, self.sample_rate

        except Exception as e:
            logger.warning(f"Error loading audio with librosa: {e}, trying alternative method")
            try:
                import subprocess
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(['ffmpeg', '-i', self.video_path, '-ar', '22050', '-ac', '1', '-y', tmp_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                self.audio_data, self.sample_rate = librosa.load(tmp_path, sr=22050, mono=True)
                os.unlink(tmp_path)
                return self.audio_data, self.sample_rate
            except:
                logger.error(f"Could not load audio, using empty array")
                self.audio_data = np.array([])
                self.sample_rate = 22050
                return self.audio_data, self.sample_rate

    def analyze_audio_energy(self, start_time: float, end_time: float) -> float:
        if self.audio_data is None:
            self.load_audio()

        if len(self.audio_data) == 0:
            return 0.0

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        segment = self.audio_data[start_sample:end_sample]

        if len(segment) == 0:
            return 0.0

        energy = librosa.feature.rms(y=segment)[0]

        return np.mean(energy)

    def detect_speech_segments(self) -> List[Tuple[float, float]]:
        cache_path = self._get_cache_path('speech')
        cached_speech = self._load_from_cache(cache_path)

        if cached_speech is not None:
            logger.info(f"Loaded {len(cached_speech)} speech segments from cache")
            return cached_speech

        logger.info("Detecting speech segments...")

        try:
            audio = AudioSegment.from_file(self.video_path)

            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=int(config.MIN_SILENCE_DURATION * 1000),
                silence_thresh=audio.dBFS - 16
            )

            speech_segments = [
                (start / 1000.0, end / 1000.0)
                for start, end in nonsilent_ranges
            ]

            logger.info(f"Detected {len(speech_segments)} speech segments")

            self._save_to_cache(cache_path, speech_segments)

            return speech_segments

        except Exception as e:
            logger.error(f"Error detecting speech segments: {e}")
            return []

    def calculate_speech_density(self, start_time: float, end_time: float, speech_segments: List[Tuple[float, float]]) -> float:
        duration = end_time - start_time
        if duration <= 0:
            return 0.0

        speech_time = 0.0
        for speech_start, speech_end in speech_segments:
            overlap_start = max(start_time, speech_start)
            overlap_end = min(end_time, speech_end)

            if overlap_start < overlap_end:
                speech_time += overlap_end - overlap_start

        return speech_time / duration

    def score_segment(
        self,
        start_time: float,
        end_time: float,
        speech_segments: List[Tuple[float, float]],
        all_motion_scores: List[float],
        all_audio_scores: List[float]
    ) -> float:
        motion = self.analyze_motion(start_time, end_time)
        audio_energy = self.analyze_audio_energy(start_time, end_time)
        speech_density = self.calculate_speech_density(start_time, end_time, speech_segments)

        all_motion_scores.append(motion)
        all_audio_scores.append(audio_energy)

        motion_normalized = normalize_score(np.array(all_motion_scores))[-1]
        audio_normalized = normalize_score(np.array(all_audio_scores))[-1]

        score = (
            config.WEIGHT_MOTION * motion_normalized +
            config.WEIGHT_AUDIO_ENERGY * audio_normalized +
            config.WEIGHT_SPEECH_DENSITY * speech_density
        )

        return score

    def _analyze_scene_batch(
        self,
        scenes: List[Tuple[float, float]],
        speech_segments: List[Tuple[float, float]]
    ) -> List[Dict]:
        segments = []
        all_motion_scores = []
        all_audio_scores = []

        for i, (start_time, end_time) in enumerate(scenes):
            if end_time - start_time < config.MIN_SCENE_DURATION:
                continue

            score = self.score_segment(
                start_time,
                end_time,
                speech_segments,
                all_motion_scores,
                all_audio_scores
            )

            if score >= config.MIN_SEGMENT_SCORE:
                adjusted_start = self.find_peak_moment(start_time, end_time)

                segments.append({
                    'start': adjusted_start,
                    'end': end_time,
                    'duration': end_time - adjusted_start,
                    'score': score,
                    'scene_index': i
                })

        return segments

    def analyze_video(self) -> List[Dict]:
        logger.info(f"Analyzing video: {self.video_path}")

        intro_end = self.detect_intro_end()

        scenes = self.detect_scenes()

        self.load_audio()

        speech_segments = self.detect_speech_segments()

        logger.info("Scoring video segments...")
        segments = []
        all_motion_scores = []
        all_audio_scores = []

        for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing scenes")):
            if start_time < intro_end:
                continue

            if end_time - start_time < config.MIN_SCENE_DURATION:
                continue

            score = self.score_segment(
                start_time,
                end_time,
                speech_segments,
                all_motion_scores,
                all_audio_scores
            )

            if score >= config.MIN_SEGMENT_SCORE:
                adjusted_start = self.find_peak_moment(start_time, end_time)

                segments.append({
                    'start': adjusted_start,
                    'end': end_time,
                    'duration': end_time - adjusted_start,
                    'score': score,
                    'scene_index': i
                })

        segments.sort(key=lambda x: x['score'], reverse=True)

        segments = segments[:min(len(segments), config.MAX_SEGMENTS_TO_ANALYZE)]

        logger.info(f"Found {len(segments)} interesting segments")

        for i, seg in enumerate(segments[:5]):
            logger.info(
                f"  #{i+1}: {format_time(seg['start'])} - {format_time(seg['end'])} "
                f"(duration: {seg['duration']:.1f}s, score: {seg['score']:.3f})"
            )

        return segments

def analyze_video(video_path: str) -> List[Dict]:
    analyzer = VideoAnalyzer(video_path)
    return analyzer.analyze_video()
