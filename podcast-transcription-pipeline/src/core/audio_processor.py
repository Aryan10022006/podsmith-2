from pydub import AudioSegment
import os
import logging

class AudioProcessor:
    def __init__(self, session_id, output_dir):
        self.session_id = session_id
        self.output_dir = output_dir
        self.audio_file_path = None
        self.audio_segment = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('AudioProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.output_dir, 'processing_log.txt'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_audio(self, audio_file_path):
        self.audio_file_path = audio_file_path
        try:
            self.audio_segment = AudioSegment.from_file(audio_file_path)
            self.logger.info(f'Loaded audio file: {audio_file_path}')
        except Exception as e:
            self.logger.error(f'Error loading audio file: {e}')
            raise

    def convert_audio_format(self, target_format='wav'):
        if self.audio_segment is None:
            self.logger.error('No audio segment loaded. Please load an audio file first.')
            return

        base_name = os.path.splitext(os.path.basename(self.audio_file_path))[0]
        output_file_path = os.path.join(self.output_dir, f'{base_name}.{target_format}')
        try:
            self.audio_segment.export(output_file_path, format=target_format)
            self.logger.info(f'Converted audio to {target_format}: {output_file_path}')
            return output_file_path
        except Exception as e:
            self.logger.error(f'Error converting audio format: {e}')
            raise

    def get_duration(self):
        if self.audio_segment is None:
            self.logger.error('No audio segment loaded. Please load an audio file first.')
            return None
        duration = len(self.audio_segment) / 1000  # duration in seconds
        self.logger.info(f'Audio duration: {duration} seconds')
        return duration

    def split_audio(self, segment_length_ms):
        if self.audio_segment is None:
            self.logger.error('No audio segment loaded. Please load an audio file first.')
            return []

        segments = []
        for start in range(0, len(self.audio_segment), segment_length_ms):
            end = min(start + segment_length_ms, len(self.audio_segment))
            segment = self.audio_segment[start:end]
            segments.append(segment)
            self.logger.info(f'Created audio segment from {start} to {end} ms')
        return segments

    def save_segment(self, segment, segment_id, target_format='wav'):
        segment_file_path = os.path.join(self.output_dir, f'segment_{segment_id}.{target_format}')
        try:
            segment.export(segment_file_path, format=target_format)
            self.logger.info(f'Saved audio segment: {segment_file_path}')
        except Exception as e:
            self.logger.error(f'Error saving audio segment: {e}')
            raise