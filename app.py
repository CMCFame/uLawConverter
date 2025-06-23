#!/usr/bin/env python3
"""
Enhanced μ-law to WAV Audio Converter and Transcriber - Streamlit Web Interface
Handles large-scale batch processing with directory structure preservation and consolidated reporting.
"""

import streamlit as st
import wave
import struct
import io
import zipfile
from pathlib import Path
import tempfile
import os
import pandas as pd
from deepgram import DeepgramClient, PrerecordedOptions
import threading
import queue
import time
import logging
from datetime import datetime
import json
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass, asdict
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ESSENTIAL: μ-law to linear PCM conversion table ---
ULAW_TO_LINEAR = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0
]

@dataclass
class ProcessingJob:
    """Represents a single file processing job"""
    file_path: Path
    relative_path: Path
    parent_folder: str
    file_name: str
    status: str = "pending"
    wav_path: Optional[Path] = None
    transcript: str = ""
    error_message: str = ""
    processing_time: float = 0.0
    file_size: int = 0

@dataclass
class ProcessingStats:
    """Statistics for the current processing session"""
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    current_file: str = ""
    current_step: str = ""
    step_start_time: Optional[datetime] = None

@dataclass
class ProcessingSession:
    """Tracks a processing session with persistent state"""
    session_id: str
    start_time: datetime
    total_files: int
    processed: int = 0
    successful: int = 0
    failed: int = 0
    current_file: str = ""
    is_active: bool = True
    log_file: str = ""

class PersistentLogger:
    """Handles persistent logging that survives app restarts"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_dir = Path("ulaw_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"session_{session_id}.log"
        self.status_file = self.log_dir / f"session_{session_id}_status.json"
        
        # Setup file logger
        self.logger = logging.getLogger(f"session_{session_id}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log(self, level: str, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "SUCCESS":
            self.logger.info(f"✅ {message}")
    
    def get_recent_logs(self, lines: int = 50) -> List[str]:
        """Get recent log lines"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return [line.strip() for line in all_lines[-lines:]]
        except Exception as e:
            return [f"Error reading log: {e}"]
    
    def save_session_status(self, session: ProcessingSession):
        """Save current session status"""
        try:
            with open(self.status_file, 'w') as f:
                # Convert datetime to string for JSON serialization
                session_dict = asdict(session)
                session_dict['start_time'] = session.start_time.isoformat()
                json.dump(session_dict, f, indent=2)
        except Exception as e:
            self.log("ERROR", f"Failed to save session status: {e}")
    
    def load_session_status(self) -> Optional[ProcessingSession]:
        """Load session status if exists"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    data['start_time'] = datetime.fromisoformat(data['start_time'])
                    return ProcessingSession(**data)
        except Exception as e:
            self.log("ERROR", f"Failed to load session status: {e}")
        return None

class RateLimitedTranscriber:
    """Handles Deepgram API calls with rate limiting"""
    
    def __init__(self, api_key: str, max_calls_per_minute: int = 40):
        self.api_key = api_key
        self.client = DeepgramClient(api_key) if api_key else None
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()
    
    def transcribe(self, wav_data: bytes) -> str:
        """Transcribe audio with rate limiting"""
        if not self.client:
            return "Transcription disabled (no API key)"
        
        # Rate limiting
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.max_calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.call_times.append(now)
        
        try:
            payload = {"buffer": wav_data}
            options = PrerecordedOptions(model="nova-2", smart_format=True)
            response = self.client.listen.prerecorded.v("1").transcribe_file(payload, options)
            return response["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription Failed: {str(e)}"

class BatchProcessor:
    """Handles large-scale batch processing of μ-law files with persistent logging"""
    
    def __init__(self, transcriber: RateLimitedTranscriber, sample_rate: int = 8000, channels: int = 1):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.channels = channels
        self.job_queue = queue.Queue()
        self.results = []
        self.stats = ProcessingStats()
        self.is_running = False
        self.should_stop = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = PersistentLogger(self.session_id)
        self.current_session = None
        
        # Try to recover previous session
        self.recover_session()
        
    def recover_session(self):
        """Try to recover a previous session"""
        log_dir = Path("ulaw_logs")
        if log_dir.exists():
            # Find the most recent session
            status_files = list(log_dir.glob("*_status.json"))
            if status_files:
                latest_file = max(status_files, key=lambda x: x.stat().st_mtime)
                session_id = latest_file.stem.replace("_status", "").replace("session_", "")
                
                temp_logger = PersistentLogger(session_id)
                recovered_session = temp_logger.load_session_status()
                
                if recovered_session and recovered_session.is_active:
                    self.logger.log("INFO", f"Found previous session: {session_id}")
                    self.session_id = session_id
                    self.logger = temp_logger
                    self.current_session = recovered_session
        
    def ulaw_to_linear(self, ulaw_byte):
        """Convert a single μ-law byte to 16-bit linear PCM."""
        return ULAW_TO_LINEAR[ulaw_byte & 0xFF]

    def convert_ulaw_to_wav_bytes(self, ulaw_data, sample_rate=8000, channels=1):
        """Convert μ-law data to WAV format and return as bytes."""
        pcm_data = bytearray()
        for byte in ulaw_data:
            linear_value = self.ulaw_to_linear(byte)
            pcm_data.extend(struct.pack('<h', linear_value))
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(pcm_data))
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def scan_directory(self, root_path: Path) -> List[ProcessingJob]:
        """Scan directory for μ-law files and create processing jobs"""
        jobs = []
        ulaw_extensions = {'.ulaw', '.ul', '.au', '.raw'}
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ulaw_extensions:
                relative_path = file_path.relative_to(root_path)
                parent_folder = str(relative_path.parent) if relative_path.parent != Path('.') else root_path.name
                
                job = ProcessingJob(
                    file_path=file_path,
                    relative_path=relative_path,
                    parent_folder=parent_folder,
                    file_name=file_path.name,
                    file_size=file_path.stat().st_size
                )
                jobs.append(job)
        
        return sorted(jobs, key=lambda x: x.file_path)
    
    def process_single_file(self, job: ProcessingJob, progress_callback=None) -> ProcessingJob:
        """Process a single μ-law file with detailed step tracking"""
        start_time = time.time()
        step_times = {}
        
        try:
            job.status = "processing"
            
            # Step 1: Reading μ-law file
            step_start = time.time()
            # Thread-safe progress callback
            if progress_callback:
                try:
                    progress_callback("reading", job.file_name)
                except:
                    pass  # Ignore Streamlit context errors in background thread
            
            self.logger.log("INFO", f"📖 Reading μ-law file: {job.parent_folder}/{job.file_name} ({job.file_size:,} bytes)")
            
            with open(job.file_path, 'rb') as f:
                ulaw_data = f.read()
            
            step_times['read'] = time.time() - step_start
            self.logger.log("INFO", f"✅ File read complete ({step_times['read']:.2f}s)")
            
            # Step 2: Converting to WAV (this is the bottleneck!)
            step_start = time.time()
            if progress_callback:
                try:
                    progress_callback("converting", job.file_name)
                except:
                    pass
            
            self.logger.log("INFO", f"🔄 Converting μ-law to WAV: {len(ulaw_data):,} bytes to process...")
            
            wav_data = self.convert_ulaw_to_wav_bytes(ulaw_data, self.sample_rate, self.channels)
            
            step_times['convert'] = time.time() - step_start
            self.logger.log("INFO", f"✅ WAV conversion complete ({step_times['convert']:.2f}s) - Generated {len(wav_data):,} bytes")
            
            # Step 3: Creating output directory and saving WAV
            step_start = time.time()
            if progress_callback:
                try:
                    progress_callback("saving_wav", job.file_name)
                except:
                    pass
            
            wav_dir = job.file_path.parent / "wav"
            wav_dir.mkdir(exist_ok=True)
            
            wav_filename = job.file_path.stem + ".wav"
            wav_path = wav_dir / wav_filename
            with open(wav_path, 'wb') as f:
                f.write(wav_data)
            
            job.wav_path = wav_path
            step_times['save_wav'] = time.time() - step_start
            self.logger.log("INFO", f"💾 WAV file saved: {wav_path} ({step_times['save_wav']:.2f}s)")
            
            # Step 4: Transcribing with Deepgram
            step_start = time.time()
            if progress_callback:
                try:
                    progress_callback("transcribing", job.file_name)
                except:
                    pass
            
            self.logger.log("INFO", f"🎤 Starting transcription with Deepgram...")
            
            job.transcript = self.transcriber.transcribe(wav_data)
            
            step_times['transcribe'] = time.time() - step_start
            transcript_preview = job.transcript[:50] + "..." if len(job.transcript) > 50 else job.transcript
            self.logger.log("INFO", f"📝 Transcription complete ({step_times['transcribe']:.2f}s): '{transcript_preview}'")
            
            job.status = "completed"
            job.processing_time = time.time() - start_time
            
            # Log detailed timing breakdown
            total_time = job.processing_time
            self.logger.log("SUCCESS", f"🎉 COMPLETED: {job.file_name}")
            self.logger.log("INFO", f"⏱️  TIMING BREAKDOWN:")
            self.logger.log("INFO", f"   📖 Read file: {step_times['read']:.2f}s ({(step_times['read']/total_time)*100:.1f}%)")
            self.logger.log("INFO", f"   🔄 Convert μ-law→WAV: {step_times['convert']:.2f}s ({(step_times['convert']/total_time)*100:.1f}%)")
            self.logger.log("INFO", f"   💾 Save WAV: {step_times['save_wav']:.2f}s ({(step_times['save_wav']/total_time)*100:.1f}%)")
            self.logger.log("INFO", f"   🎤 Transcribe: {step_times['transcribe']:.2f}s ({(step_times['transcribe']/total_time)*100:.1f}%)")
            self.logger.log("INFO", f"   🏁 TOTAL: {total_time:.2f}s")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.processing_time = time.time() - start_time
            self.logger.log("ERROR", f"❌ FAILED {job.file_name}: {str(e)}")
            if progress_callback:
                try:
                    progress_callback("failed", job.file_name)
                except:
                    pass
        
        return job
    
    def process_batch(self, jobs: List[ProcessingJob], max_workers: int = 2, 
                     progress_callback=None, status_callback=None, step_callback=None):
        """Process a batch of jobs with thread-safe callbacks to avoid Streamlit context warnings"""
        self.is_running = True
        self.should_stop = False
        self.results = []
        
        # Thread-safe wrapper for step progress that won't cause Streamlit warnings
        def safe_step_progress_wrapper(step, filename):
            try:
                self.stats.current_step = step
                self.stats.step_start_time = datetime.now()
                # Only call step_callback if it won't cause context issues
                if step_callback:
                    step_callback(step, filename)
            except Exception:
                # Silently ignore Streamlit context errors from background thread
                pass
        
        # Create new session
        self.current_session = ProcessingSession(
            session_id=self.session_id,
            start_time=datetime.now(),
            total_files=len(jobs),
            log_file=str(self.logger.log_file)
        )
        
        self.logger.log("INFO", f"=== STARTING BATCH PROCESSING ===")
        self.logger.log("INFO", f"Session ID: {self.session_id}")
        self.logger.log("INFO", f"Total files: {len(jobs)}")
        self.logger.log("INFO", f"Max workers: {max_workers}")
        self.logger.log("INFO", f"Sample rate: {self.sample_rate} Hz")
        self.logger.log("INFO", f"Channels: {self.channels}")
        self.logger.log("INFO", f"Average file size: {sum(job.file_size for job in jobs) / len(jobs) / 1024:.1f} KB")
        
        # Initialize stats
        self.stats = ProcessingStats(
            total_files=len(jobs),
            start_time=datetime.now()
        )
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs with thread-safe step callback
                future_to_job = {
                    executor.submit(self.process_single_file, job, safe_step_progress_wrapper): job 
                    for job in jobs
                }
                
                for future in concurrent.futures.as_completed(future_to_job):
                    if self.should_stop:
                        self.logger.log("WARNING", "Processing stopped by user")
                        break
                    
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Update stats (thread-safe operations)
                        self.stats.processed += 1
                        self.current_session.processed += 1
                        
                        if result.status == "completed":
                            self.stats.successful += 1
                            self.current_session.successful += 1
                        else:
                            self.stats.failed += 1
                            self.current_session.failed += 1
                        
                        self.stats.current_file = result.file_name
                        self.current_session.current_file = result.file_name
                        self.stats.current_step = "completed"
                        
                        # Estimate completion time
                        if self.stats.processed > 0:
                            elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                            avg_time_per_file = elapsed / self.stats.processed
                            remaining_files = self.stats.total_files - self.stats.processed
                            eta_seconds = remaining_files * avg_time_per_file
                            self.stats.estimated_completion = datetime.now() + pd.Timedelta(seconds=eta_seconds)
                        
                        # Save session status periodically
                        if self.stats.processed % 10 == 0:  # Every 10 files
                            self.logger.save_session_status(self.current_session)
                            remaining = self.stats.total_files - self.stats.processed
                            self.logger.log("INFO", f"📊 PROGRESS UPDATE: {self.stats.processed}/{self.stats.total_files} ({(self.stats.processed/self.stats.total_files)*100:.1f}%) - {remaining} files remaining")
                        
                        # Thread-safe callback calls
                        try:
                            if progress_callback:
                                progress_callback(self.stats)
                            if status_callback:
                                status_callback(result)
                        except Exception:
                            # Ignore Streamlit context errors from background thread
                            pass
                            
                    except Exception as e:
                        self.logger.log("ERROR", f"Error in future result: {e}")
                        
        except Exception as e:
            self.logger.log("ERROR", f"Critical error in batch processing: {e}")
                        
        finally:
            self.is_running = False
            self.current_session.is_active = False
            self.logger.save_session_status(self.current_session)
            
            self.logger.log("INFO", f"=== BATCH PROCESSING COMPLETED ===")
            self.logger.log("INFO", f"Total processed: {self.stats.processed}")
            self.logger.log("INFO", f"Successful: {self.stats.successful}")
            self.logger.log("INFO", f"Failed: {self.stats.failed}")
            
            # Auto-save results when processing completes
            if self.results:
                try:
                    auto_save_path = self.export_results_to_csv(Path("temp.csv"), auto_save=True)
                    if auto_save_path:
                        self.logger.log("SUCCESS", f"Results auto-saved to: {auto_save_path}")
                except Exception as e:
                    self.logger.log("ERROR", f"Error auto-saving results: {e}")
    
    def stop_processing(self):
        """Stop the current processing"""
        self.should_stop = True
    
    def export_results_to_csv(self, output_path: Path, auto_save: bool = True):
        """Export all results to a CSV file"""
        if not self.results:
            return None
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'Parent Folder': result.parent_folder,
                'File Name': result.file_name,
                'Relative Path': str(result.relative_path),
                'Status': result.status,
                'Transcript': result.transcript,
                'Processing Time (s)': round(result.processing_time, 2),
                'File Size (bytes)': result.file_size,
                'WAV Path': str(result.wav_path) if result.wav_path else "",
                'Error Message': result.error_message,
                'Processed At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(data)
        
        # Save CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        if auto_save:
            # Also create an auto-save copy with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_dir = Path("ulaw_results")
            auto_save_dir.mkdir(exist_ok=True)
            auto_save_path = auto_save_dir / f"transcription_results_{timestamp}.csv"
            df.to_csv(auto_save_path, index=False, encoding='utf-8-sig')
            return auto_save_path
        
        return output_path

def main():
    st.set_page_config(
        page_title="Enhanced μ-law Batch Processor",
        page_icon="🎵",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: #e0e0e0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .status-success { color: #28a745; }
    .status-failed { color: #dc3545; }
    .status-processing { color: #007bff; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎵 Enhanced μ-law Batch Processor")
    st.markdown("Large-scale μ-law to WAV conversion with transcription and consolidated reporting.")
    
    # Get API key
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.warning("DEEPGRAM_API_KEY not found in secrets.toml. Transcription will be disabled.", icon="⚠️")
        api_key = None

    # Initialize session state
    if 'processor' not in st.session_state:
        transcriber = RateLimitedTranscriber(api_key)
        st.session_state.processor = BatchProcessor(transcriber)
    
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = ProcessingStats()
    
    if 'current_jobs' not in st.session_state:
        st.session_state.current_jobs = []
    
    if 'show_logs' not in st.session_state:
        st.session_state.show_logs = False

    tab1, tab2, tab3, tab4 = st.tabs(["Directory Processing & Logs", "Job Monitoring", "Results", "Settings"])
    
    with tab1:
        st.header("Large-Scale Directory Processing")
        
        # Session recovery notice
        if st.session_state.processor.current_session and st.session_state.processor.current_session.is_active:
            st.info(f"🔄 Recovered session: {st.session_state.processor.session_id}")
        
        # Directory selection
        st.markdown("### 📁 Select Root Directory")
        directory_path = st.text_input(
            "Root Directory Path",
            placeholder="Enter the path to your directory containing μ-law files",
            help="Path to the directory containing subdirectories with .ulaw files"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Scan Directory", type="primary"):
                if directory_path and Path(directory_path).exists():
                    with st.spinner("Scanning directory for μ-law files..."):
                        jobs = st.session_state.processor.scan_directory(Path(directory_path))
                        st.session_state.current_jobs = jobs
                    
                    if jobs:
                        st.success(f"Found {len(jobs)} μ-law files across {len(set(job.parent_folder for job in jobs))} directories")
                        
                        # Show directory structure preview
                        folder_counts = {}
                        for job in jobs:
                            folder_counts[job.parent_folder] = folder_counts.get(job.parent_folder, 0) + 1
                        
                        st.markdown("### 📊 Directory Structure")
                        structure_df = pd.DataFrame([
                            {"Folder": folder, "File Count": count}
                            for folder, count in sorted(folder_counts.items())
                        ])
                        st.dataframe(structure_df, use_container_width=True)
                    else:
                        st.warning("No μ-law files found in the specified directory.")
                else:
                    st.error("Please enter a valid directory path.")
        
        with col2:
            # Reduced max_workers for stability with large batches
            if st.button("▶️ Start Processing", disabled=not st.session_state.current_jobs):
                if st.session_state.current_jobs and not st.session_state.processor.is_running:
                    st.session_state.processing_stats = ProcessingStats()
                    st.session_state.show_logs = True
                    
                    # Start processing in background with fewer workers for large batches
                    max_workers = 2 if len(st.session_state.current_jobs) > 100 else 4
                    
                    def progress_callback(stats):
                        # Thread-safe update of session state
                        try:
                            st.session_state.processing_stats = stats
                        except Exception:
                            # Ignore context errors - will be picked up on next refresh
                            pass
                    
                    def status_callback(result):
                        # Minimal callback to avoid context issues
                        pass
                    
                    def step_callback(step, filename):
                        # Thread-safe step tracking
                        try:
                            if not hasattr(st.session_state, 'current_step_info'):
                                st.session_state.current_step_info = {}
                            st.session_state.current_step_info = {
                                'step': step, 
                                'filename': filename,
                                'timestamp': datetime.now()
                            }
                        except Exception:
                            # Ignore context errors
                            pass
                    
                    threading.Thread(
                        target=st.session_state.processor.process_batch,
                        args=(st.session_state.current_jobs, max_workers, progress_callback, status_callback, step_callback),
                        daemon=True
                    ).start()
                    
                    st.success("Processing started! Monitor detailed progress below. Results will be auto-saved when complete.")
                    st.info("ℹ️ Note: Any 'ScriptRunContext' warnings in the terminal are normal and can be ignored.")
                    st.rerun()
        
        with col3:
            if st.button("⏹️ Stop Processing", disabled=not st.session_state.processor.is_running):
                st.session_state.processor.stop_processing()
                st.warning("Processing stopped.")
                st.rerun()
        
        # Real-time processing status and logs
        if st.session_state.show_logs or st.session_state.processor.is_running:
            st.markdown("---")
            st.markdown("### 📊 Real-time Processing Status")
            
            # Auto-refresh during processing
            if st.session_state.processor.is_running:
                time.sleep(2)
                st.rerun()
            
            # Current stats
            stats = st.session_state.processing_stats
            if stats.total_files > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Files", stats.total_files)
                with col2:
                    st.metric("Processed", f"{stats.processed}/{stats.total_files}")
                with col3:
                    st.metric("Successful", stats.successful)
                with col4:
                    st.metric("Failed", stats.failed)
                
                # Progress bar
                if stats.total_files > 0:
                    progress = stats.processed / stats.total_files
                    st.progress(progress, text=f"Progress: {progress:.1%}")
                
                # Current file and timing
                if stats.current_file:
                    # Show current step with detailed info
                    step_display = {
                        "reading": "📖 Reading μ-law file",
                        "converting": "🔄 Converting μ-law to WAV (bottleneck step)",
                        "saving_wav": "💾 Saving WAV file",
                        "transcribing": "🎤 Transcribing with Deepgram",
                        "completed": "✅ File completed",
                        "failed": "❌ Processing failed"
                    }
                    
                    current_step_text = step_display.get(stats.current_step, stats.current_step)
                    
                    # Calculate step duration if available
                    step_duration = ""
                    if stats.step_start_time and stats.current_step in ["reading", "converting", "saving_wav", "transcribing"]:
                        step_elapsed = (datetime.now() - stats.step_start_time).total_seconds()
                        step_duration = f" ({step_elapsed:.1f}s elapsed)"
                    
                    if stats.current_step == "converting":
                        st.warning(f"🔄 **Currently processing:** {stats.current_file}")
                        st.info(f"⚙️ **Step:** {current_step_text}{step_duration}")
                        st.markdown("💡 **Note:** μ-law to WAV conversion is typically the slowest step")
                    else:
                        st.info(f"🔄 **Currently processing:** {stats.current_file}")
                        st.info(f"⚙️ **Step:** {current_step_text}{step_duration}")
                
                if stats.start_time:
                    elapsed = datetime.now() - stats.start_time
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**⏱️ Total Elapsed:** {str(elapsed).split('.')[0]}")
                    with col2:
                        if stats.estimated_completion and stats.processed > 0:
                            eta = stats.estimated_completion - datetime.now()
                            if eta.total_seconds() > 0:
                                st.markdown(f"**⏳ ETA:** {str(eta).split('.')[0]}")
                    
                    # Show processing rate
                    if stats.processed > 0:
                        files_per_minute = (stats.processed / elapsed.total_seconds()) * 60
                        st.markdown(f"**📈 Processing Rate:** {files_per_minute:.1f} files/minute")
            
            # Real-time logs
            st.markdown("### 📝 Processing Logs")
            
            # Performance tips
            if st.session_state.processor.is_running:
                st.markdown("### 💡 Performance Information")
                with st.expander("Understanding Processing Steps", expanded=False):
                    st.markdown("""
                    **Processing Pipeline for each file:**
                    
                    1. **📖 Read File** (Fast) - Read μ-law data from disk
                    2. **🔄 Convert μ-law→WAV** (Slowest) - Mathematical conversion of audio data
                    3. **💾 Save WAV** (Fast) - Write converted audio to disk  
                    4. **🎤 Transcribe** (Medium) - Upload to Deepgram API and get transcript
                    
                    **Why conversion is slow:**
                    - Each μ-law byte must be converted using a lookup table
                    - Large files require processing millions of audio samples
                    - This is CPU-intensive work done on your local machine
                    - Deepgram transcription is much faster as it's optimized for this task
                    
                    **Tips for better performance:**
                    - Close other CPU-intensive applications
                    - Processing speed depends on your CPU and file sizes
                    - Multiple workers help, but too many can overwhelm your system
                    """)
            
            # Log controls
            col1, col2 = st.columns([3, 1])
            with col1:
                log_lines = st.slider("Number of log lines to show", 10, 200, 50)
            with col2:
                if st.button("🔄 Refresh Logs"):
                    st.rerun()
            
            # Get and display logs
            if hasattr(st.session_state.processor, 'logger'):
                recent_logs = st.session_state.processor.logger.get_recent_logs(log_lines)
                
                if recent_logs:
                    # Create a container for logs with styling
                    log_container = st.container()
                    with log_container:
                        # Display logs in a code block for better formatting
                        log_text = "\n".join(recent_logs)
                        
                        # Color code different log levels
                        log_display = ""
                        for line in recent_logs:
                            if "ERROR" in line:
                                log_display += f"🔴 {line}\n"
                            elif "✅" in line or "SUCCESS" in line:
                                log_display += f"🟢 {line}\n"
                            elif "WARNING" in line:
                                log_display += f"🟡 {line}\n"
                            else:
                                log_display += f"ℹ️ {line}\n"
                        
                        st.text_area(
                            "Recent Activity",
                            log_display,
                            height=300,
                            help="Live log feed - refreshes automatically during processing"
                        )
                        
                        # Log file info
                        log_file_path = st.session_state.processor.logger.log_file
                        st.markdown(f"📄 **Full log file:** `{log_file_path}`")
                else:
                    st.info("No logs available yet. Start processing to see activity.")
            
            # Processing summary when not running
            if not st.session_state.processor.is_running and st.session_state.processor.results:
                st.markdown("### ✅ Processing Complete")
                results = st.session_state.processor.results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    successful = len([r for r in results if r.status == "completed"])
                    st.metric("Successful", successful)
                with col3:
                    success_rate = (successful / len(results)) * 100 if results else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                if st.button("📋 View Full Results", type="secondary"):
                    st.session_state.show_logs = False
                    st.rerun()
    
    with tab2:
        st.header("📊 Job Monitoring")
        
        # Auto-refresh every 2 seconds during processing
        if st.session_state.processor.is_running:
            time.sleep(2)
            st.rerun()
        
        stats = st.session_state.processing_stats
        
        if stats.total_files > 0:
            # Progress metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Files", stats.total_files)
            with col2:
                st.metric("Processed", f"{stats.processed}/{stats.total_files}")
            with col3:
                st.metric("Successful", stats.successful, delta=stats.successful - stats.failed)
            with col4:
                st.metric("Failed", stats.failed)
            
            # Progress bar
            if stats.total_files > 0:
                progress = stats.processed / stats.total_files
                st.progress(progress, text=f"Progress: {progress:.1%}")
            
            # Current status
            if stats.current_file:
                st.info(f"Currently processing: {stats.current_file}")
            
            # Time estimates
            if stats.start_time:
                elapsed = datetime.now() - stats.start_time
                st.markdown(f"**Elapsed Time:** {str(elapsed).split('.')[0]}")
                
                if stats.estimated_completion and stats.processed > 0:
                    eta = stats.estimated_completion - datetime.now()
                    if eta.total_seconds() > 0:
                        st.markdown(f"**Estimated Time Remaining:** {str(eta).split('.')[0]}")
        
        # Real-time results table (last 10 processed files)
        if st.session_state.processor.results:
            st.markdown("### 📋 Recent Results")
            recent_results = st.session_state.processor.results[-10:]
            
            display_data = []
            for result in reversed(recent_results):
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "processing": "🔄"
                }.get(result.status, "⏳")
                
                display_data.append({
                    "Status": f"{status_emoji} {result.status.title()}",
                    "Folder": result.parent_folder,
                    "File": result.file_name,
                    "Size": f"{result.file_size:,} bytes",
                    "Time": f"{result.processing_time:.1f}s",
                    "Transcript Preview": result.transcript[:50] + "..." if len(result.transcript) > 50 else result.transcript
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    with tab3:
        st.header("📄 Results & Export")
        
        if st.session_state.processor.results:
            st.markdown(f"### 📊 Processing Summary")
            
            results = st.session_state.processor.results
            total_processed = len(results)
            successful = len([r for r in results if r.status == "completed"])
            failed = len([r for r in results if r.status == "failed"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", total_processed)
            with col2:
                st.metric("Successful", successful, delta=successful - failed)
            with col3:
                st.metric("Success Rate", f"{(successful/total_processed)*100:.1f}%")
            
            # Export options
            st.markdown("### 💾 Export Results")
            st.info("📁 Results are automatically saved to the 'ulaw_results' directory")
            
            # Manual CSV export
            if st.button("📄 Download CSV Report", type="primary"):
                try:
                    results_df = pd.DataFrame([
                        {
                            'Parent Folder': r.parent_folder,
                            'File Name': r.file_name,
                            'Relative Path': str(r.relative_path),
                            'Status': r.status,
                            'Transcript': r.transcript,
                            'Processing Time (s)': round(r.processing_time, 2),
                            'File Size (bytes)': r.file_size,
                            'WAV Path': str(r.wav_path) if r.wav_path else "",
                            'Error Message': r.error_message,
                            'Processed At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        for r in results
                    ])
                    
                    csv_data = results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ulaw_transcription_results_{timestamp}.csv"
                    
                    st.download_button(
                        "📥 Download CSV File",
                        csv_data,
                        filename,
                        "text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error creating CSV file: {e}")
            
            # Auto-save status
            auto_save_dir = Path("ulaw_results")
            if auto_save_dir.exists():
                csv_files = list(auto_save_dir.glob("*.csv"))
                if csv_files:
                    st.markdown("### 📂 Auto-saved Files")
                    st.markdown(f"**Location:** `{auto_save_dir.absolute()}`")
                    
                    # Show recent auto-saved files
                    recent_files = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                    for file_path in recent_files:
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        st.markdown(f"- `{file_path.name}` (saved: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
                    
                    if len(csv_files) > 5:
                        st.markdown(f"... and {len(csv_files) - 5} more files")
            
            # Quick auto-save current results
            if st.button("💾 Save Current Results", type="secondary"):
                try:
                    auto_save_path = st.session_state.processor.export_results_to_csv(
                        Path("temp.csv"), auto_save=True
                    )
                    if auto_save_path:
                        st.success(f"✅ Results auto-saved to: `{auto_save_path}`")
                    else:
                        st.warning("No results to save.")
                except Exception as e:
                    st.error(f"Error auto-saving: {e}")
            
            # Results table with filtering
            st.markdown("### 🔍 Detailed Results")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox("Filter by Status", ["All", "Completed", "Failed"])
            with col2:
                folder_filter = st.selectbox(
                    "Filter by Folder",
                    ["All"] + sorted(set(r.parent_folder for r in results))
                )
            
            # Apply filters
            filtered_results = results
            if status_filter != "All":
                filtered_results = [r for r in filtered_results if r.status == status_filter.lower()]
            if folder_filter != "All":
                filtered_results = [r for r in filtered_results if r.parent_folder == folder_filter]
            
            # Display filtered results
            if filtered_results:
                display_df = pd.DataFrame([
                    {
                        'Folder': r.parent_folder,
                        'File': r.file_name,
                        'Status': r.status.title(),
                        'Size (KB)': f"{r.file_size/1024:.1f}",
                        'Time (s)': f"{r.processing_time:.1f}",
                        'Transcript': r.transcript[:100] + "..." if len(r.transcript) > 100 else r.transcript,
                        'Error': r.error_message[:50] + "..." if len(r.error_message) > 50 else r.error_message
                    }
                    for r in filtered_results
                ])
                
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.info("No results match the current filters.")
        else:
            st.info("No results available. Start processing some files first!")
    
    with tab4:
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Audio Settings")
            new_sample_rate = st.selectbox(
                "Sample Rate (Hz)",
                [8000, 16000, 22050, 44100, 48000],
                index=0,
                help="Most μ-law files use 8000 Hz"
            )
            
            new_channels = st.radio(
                "Channels",
                [1, 2],
                format_func=lambda x: "Mono" if x == 1 else "Stereo",
                help="Most μ-law files are mono"
            )
            
            if st.button("Apply Audio Settings"):
                st.session_state.processor.sample_rate = new_sample_rate
                st.session_state.processor.channels = new_channels
                st.success("Audio settings updated!")
        
        with col2:
            st.markdown("### 🔄 Processing Settings")
            
            max_workers = st.slider(
                "Max Concurrent Workers",
                min_value=1,
                max_value=6,
                value=2,
                help="Number of files processed simultaneously (reduced for stability with large batches)"
            )
            
            rate_limit = st.slider(
                "API Rate Limit (calls/minute)",
                min_value=10,
                max_value=100,
                value=40,
                help="Deepgram API calls per minute"
            )
            
            if st.button("Apply Processing Settings"):
                if hasattr(st.session_state.processor.transcriber, 'max_calls_per_minute'):
                    st.session_state.processor.transcriber.max_calls_per_minute = rate_limit
                st.success("Processing settings updated!")
        
        # System information
        st.markdown("### 📊 System Information")
        
        if api_key:
            st.success("✅ Deepgram API key configured")
        else:
            st.error("❌ Deepgram API key not found")
        
        st.info(f"📁 Current working directory: {os.getcwd()}")
        st.info(f"💾 Auto-save directory: {Path('ulaw_results').absolute()}")
        st.info(f"📝 Log directory: {Path('ulaw_logs').absolute()}")
        
        # Show auto-save directory status
        auto_save_dir = Path("ulaw_results")
        if auto_save_dir.exists():
            csv_count = len(list(auto_save_dir.glob("*.csv")))
            st.success(f"📂 Auto-save directory exists with {csv_count} saved CSV files")
        else:
            st.warning("📂 Auto-save directory will be created when first results are saved")
        
        # Show log directory status
        log_dir = Path("ulaw_logs")
        if log_dir.exists():
            log_count = len(list(log_dir.glob("*.log")))
            session_count = len(list(log_dir.glob("*_status.json")))
            st.success(f"📝 Log directory exists with {log_count} log files and {session_count} session files")
        else:
            st.warning("📝 Log directory will be created when processing starts")
        
        # Current session info
        if hasattr(st.session_state.processor, 'session_id'):
            st.markdown(f"**Current Session:** `{st.session_state.processor.session_id}`")
        
        # Thread safety info
        st.markdown("### 🔧 Technical Information")
        with st.expander("About Thread Safety & Warnings", expanded=False):
            st.markdown("""
            **Background Processing:**
            - File processing runs in background threads to avoid blocking the UI
            - Thread-safe logging prevents Streamlit context warnings
            - All errors are handled gracefully without affecting functionality
            
            **Terminal Warnings:**
            - Any "missing ScriptRunContext" warnings can be safely ignored
            - These are handled internally and don't affect processing
            - The app is designed to work correctly even with these warnings
            
            **Performance Notes:**
            - Processing continues even if the web interface refreshes
            - Session state is preserved and can be recovered
            - Logs are written to files for persistence
            """)
            
        # Clear all warnings from terminal
        if st.button("🧹 Clear Terminal Warnings", help="Restart processor to clear any accumulated warnings"):
            # Reinitialize processor to clear any thread context issues
            transcriber = RateLimitedTranscriber(api_key)
            st.session_state.processor = BatchProcessor(transcriber)
            st.success("✅ Processor reinitialized - terminal warnings should be cleared")
        
        # Clear session data
        st.markdown("### 🗑️ Reset Data")
        if st.button("Clear All Data", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All data cleared! Please refresh the page.")

if __name__ == '__main__':
    main()