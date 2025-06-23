#!/usr/bin/env python3
"""
Enhanced μ-law to WAV Audio Converter and Transcriber - Streamlit Cloud Version
Handles file uploads with batch processing and transcription capabilities.
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

# Configure logging for in-memory logging
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
class UploadedFileJob:
    """Represents a single uploaded file processing job"""
    file_name: str
    file_data: bytes
    file_size: int
    status: str = "pending"
    wav_data: Optional[bytes] = None
    transcript: str = ""
    error_message: str = ""
    processing_time: float = 0.0

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

class InMemoryLogger:
    """Handles in-memory logging for Streamlit Cloud"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_entries = []
        self.max_entries = 1000  # Keep last 1000 log entries
        
    def log(self, level: str, message: str):
        """Log a message to memory"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} | {level} | {message}"
        self.log_entries.append(entry)
        
        # Keep only recent entries
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]
    
    def get_recent_logs(self, lines: int = 50) -> List[str]:
        """Get recent log lines"""
        return self.log_entries[-lines:] if self.log_entries else []

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

class CloudBatchProcessor:
    """Handles batch processing of uploaded μ-law files in Streamlit Cloud"""
    
    def __init__(self, transcriber: RateLimitedTranscriber, sample_rate: int = 8000, channels: int = 1):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.channels = channels
        self.results = []
        self.stats = ProcessingStats()
        self.is_running = False
        self.should_stop = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = InMemoryLogger(self.session_id)
        
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
    
    def process_single_file(self, job: UploadedFileJob, progress_callback=None) -> UploadedFileJob:
        """Process a single uploaded μ-law file with detailed step tracking"""
        start_time = time.time()
        step_times = {}
        
        try:
            job.status = "processing"
            
            # Step 1: Reading μ-law file (already in memory)
            step_start = time.time()
            if progress_callback:
                try:
                    progress_callback("reading", job.file_name)
                except:
                    pass
            
            self.logger.log("INFO", f"📖 Processing uploaded file: {job.file_name} ({job.file_size:,} bytes)")
            ulaw_data = job.file_data
            
            step_times['read'] = time.time() - step_start
            self.logger.log("INFO", f"✅ File data loaded ({step_times['read']:.2f}s)")
            
            # Step 2: Converting to WAV (this is the bottleneck!)
            step_start = time.time()
            if progress_callback:
                try:
                    progress_callback("converting", job.file_name)
                except:
                    pass
            
            self.logger.log("INFO", f"🔄 Converting μ-law to WAV: {len(ulaw_data):,} bytes to process...")
            
            wav_data = self.convert_ulaw_to_wav_bytes(ulaw_data, self.sample_rate, self.channels)
            job.wav_data = wav_data
            
            step_times['convert'] = time.time() - step_start
            self.logger.log("INFO", f"✅ WAV conversion complete ({step_times['convert']:.2f}s) - Generated {len(wav_data):,} bytes")
            
            # Step 3: Transcribing with Deepgram
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
            self.logger.log("INFO", f"   📖 Load file: {step_times['read']:.2f}s ({(step_times['read']/total_time)*100:.1f}%)")
            self.logger.log("INFO", f"   🔄 Convert μ-law→WAV: {step_times['convert']:.2f}s ({(step_times['convert']/total_time)*100:.1f}%)")
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
    
    def process_batch(self, jobs: List[UploadedFileJob], max_workers: int = 2, 
                     progress_callback=None, status_callback=None, step_callback=None):
        """Process a batch of uploaded files with thread-safe callbacks"""
        self.is_running = True
        self.should_stop = False
        self.results = []
        
        # Thread-safe wrapper for step progress
        def safe_step_progress_wrapper(step, filename):
            try:
                self.stats.current_step = step
                self.stats.step_start_time = datetime.now()
                if step_callback:
                    step_callback(step, filename)
            except Exception:
                pass
        
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
                        
                        if result.status == "completed":
                            self.stats.successful += 1
                        else:
                            self.stats.failed += 1
                        
                        self.stats.current_file = result.file_name
                        self.stats.current_step = "completed"
                        
                        # Estimate completion time
                        if self.stats.processed > 0:
                            elapsed = (datetime.now() - self.stats.start_time).total_seconds()
                            avg_time_per_file = elapsed / self.stats.processed
                            remaining_files = self.stats.total_files - self.stats.processed
                            eta_seconds = remaining_files * avg_time_per_file
                            self.stats.estimated_completion = datetime.now() + pd.Timedelta(seconds=eta_seconds)
                        
                        # Log progress periodically
                        if self.stats.processed % 5 == 0:  # Every 5 files
                            remaining = self.stats.total_files - self.stats.processed
                            self.logger.log("INFO", f"📊 PROGRESS: {self.stats.processed}/{self.stats.total_files} ({(self.stats.processed/self.stats.total_files)*100:.1f}%) - {remaining} files remaining")
                        
                        # Thread-safe callback calls
                        try:
                            if progress_callback:
                                progress_callback(self.stats)
                            if status_callback:
                                status_callback(result)
                        except Exception:
                            pass
                            
                    except Exception as e:
                        self.logger.log("ERROR", f"Error in future result: {e}")
                        
        except Exception as e:
            self.logger.log("ERROR", f"Critical error in batch processing: {e}")
                        
        finally:
            self.is_running = False
            self.logger.log("INFO", f"=== BATCH PROCESSING COMPLETED ===")
            self.logger.log("INFO", f"Total processed: {self.stats.processed}")
            self.logger.log("INFO", f"Successful: {self.stats.successful}")
            self.logger.log("INFO", f"Failed: {self.stats.failed}")
    
    def stop_processing(self):
        """Stop the current processing"""
        self.should_stop = True
    
    def export_results_to_csv(self) -> bytes:
        """Export all results to CSV and return as bytes"""
        if not self.results:
            return b""
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'File Name': result.file_name,
                'Status': result.status,
                'Transcript': result.transcript,
                'Processing Time (s)': round(result.processing_time, 2),
                'File Size (bytes)': result.file_size,
                'Error Message': result.error_message,
                'Processed At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    
    def create_wav_zip(self) -> bytes:
        """Create a ZIP file containing all converted WAV files"""
        if not self.results:
            return b""
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for result in self.results:
                if result.wav_data and result.status == "completed":
                    wav_filename = Path(result.file_name).stem + ".wav"
                    zip_file.writestr(wav_filename, result.wav_data)
        
        zip_buffer.seek(0)
        return zip_buffer.read()

def main():
    st.set_page_config(
        page_title="μ-law Converter & Transcriber - Cloud Edition",
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
    
    st.title("🎵 μ-law Converter & Transcriber - Cloud Edition")
    st.markdown("Upload μ-law files for conversion to WAV and AI transcription. Works with single files or batches.")
    
    # Get API key
    try:
        api_key = st.secrets["DEEPGRAM_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.warning("DEEPGRAM_API_KEY not found in secrets. Transcription will be disabled.", icon="⚠️")
        api_key = None

    # Initialize session state
    if 'processor' not in st.session_state:
        transcriber = RateLimitedTranscriber(api_key)
        st.session_state.processor = CloudBatchProcessor(transcriber)
    
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = ProcessingStats()

    tab1, tab2, tab3, tab4 = st.tabs(["File Upload & Processing", "Job Monitoring", "Results", "Settings"])
    
    with tab1:
        st.header("Upload μ-law Files")
        
        # File upload options
        upload_option = st.radio(
            "Choose upload method:",
            ["Single File", "Multiple Files (Batch)"],
            horizontal=True
        )
        
        uploaded_files = []
        
        if upload_option == "Single File":
            st.markdown("### 📁 Upload Single File")
            uploaded_file = st.file_uploader(
                "Choose a μ-law file",
                type=['ulaw', 'ul', 'au', 'raw'],
                help="Select a μ-law encoded audio file"
            )
            if uploaded_file:
                uploaded_files = [uploaded_file]
        
        else:  # Batch upload
            st.markdown("### 📁 Upload Multiple Files")
            uploaded_files = st.file_uploader(
                "Choose multiple μ-law files",
                type=['ulaw', 'ul', 'au', 'raw'],
                accept_multiple_files=True,
                help="Select multiple μ-law files for batch processing"
            )
        
        # Audio settings
        if uploaded_files:
            st.markdown("### ⚙️ Audio Settings")
            col1, col2 = st.columns(2)
            with col1:
                sample_rate = st.selectbox("Sample Rate (Hz)", [8000, 16000, 22050, 44100, 48000], 0, help="Most μ-law files use 8000 Hz")
            with col2:
                channels = st.radio("Channels", [1, 2], 0, format_func=lambda x: "Mono" if x == 1 else "Stereo", help="Most μ-law files are mono")
            
            # Update processor settings
            st.session_state.processor.sample_rate = sample_rate
            st.session_state.processor.channels = channels
            
            # Show file info
            st.markdown("### 📊 Uploaded Files")
            total_size = sum(f.size for f in uploaded_files)
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded • Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            
            # File list
            if len(uploaded_files) > 1:
                file_data = []
                for f in uploaded_files:
                    file_data.append({
                        "File Name": f.name,
                        "Size (KB)": f"{f.size/1024:.1f}",
                        "Type": f.type or "μ-law audio"
                    })
                st.dataframe(pd.DataFrame(file_data), use_container_width=True)
            
            # Processing controls
            st.markdown("### 🎛️ Processing Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Processing", type="primary", disabled=st.session_state.processor.is_running):
                    # Convert uploaded files to processing jobs
                    jobs = []
                    for uploaded_file in uploaded_files:
                        file_data = uploaded_file.read()
                        job = UploadedFileJob(
                            file_name=uploaded_file.name,
                            file_data=file_data,
                            file_size=uploaded_file.size
                        )
                        jobs.append(job)
                        # Reset file pointer for potential reuse
                        uploaded_file.seek(0)
                    
                    # Start processing
                    st.session_state.processing_stats = ProcessingStats()
                    
                    # Determine optimal workers based on file count
                    max_workers = min(2, len(jobs)) if len(jobs) > 20 else min(4, len(jobs))
                    
                    def progress_callback(stats):
                        try:
                            st.session_state.processing_stats = stats
                        except Exception:
                            pass
                    
                    def status_callback(result):
                        pass
                    
                    def step_callback(step, filename):
                        try:
                            st.session_state.current_step_info = {
                                'step': step, 
                                'filename': filename,
                                'timestamp': datetime.now()
                            }
                        except Exception:
                            pass
                    
                    threading.Thread(
                        target=st.session_state.processor.process_batch,
                        args=(jobs, max_workers, progress_callback, status_callback, step_callback),
                        daemon=True
                    ).start()
                    
                    st.success("🚀 Processing started! Monitor progress below.")
                    st.info("ℹ️ Note: Processing happens in the cloud. All files are processed in memory.")
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Processing", disabled=not st.session_state.processor.is_running):
                    st.session_state.processor.stop_processing()
                    st.warning("⏹️ Processing stopped.")
                    st.rerun()
        
        # Real-time processing status
        if st.session_state.processor.is_running or st.session_state.processor.results:
            st.markdown("---")
            st.markdown("### 📊 Processing Status")
            
            # Auto-refresh during processing
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
                    st.metric("Successful", stats.successful)
                with col4:
                    st.metric("Failed", stats.failed)
                
                # Progress bar
                if stats.total_files > 0:
                    progress = stats.processed / stats.total_files
                    st.progress(progress, text=f"Progress: {progress:.1%}")
                
                # Current step information
                if stats.current_file:
                    step_display = {
                        "reading": "📖 Loading file data",
                        "converting": "🔄 Converting μ-law to WAV (main processing step)",
                        "transcribing": "🎤 Transcribing with Deepgram AI",
                        "completed": "✅ File completed",
                        "failed": "❌ Processing failed"
                    }
                    
                    current_step_text = step_display.get(stats.current_step, stats.current_step)
                    
                    if stats.current_step == "converting":
                        st.warning(f"🔄 **Currently processing:** {stats.current_file}")
                        st.info(f"⚙️ **Step:** {current_step_text}")
                        st.markdown("💡 **Note:** μ-law to WAV conversion is the most time-consuming step")
                    else:
                        st.info(f"🔄 **Currently processing:** {stats.current_file}")
                        st.info(f"⚙️ **Step:** {current_step_text}")
                
                # Timing information
                if stats.start_time:
                    elapsed = datetime.now() - stats.start_time
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**⏱️ Elapsed:** {str(elapsed).split('.')[0]}")
                    with col2:
                        if stats.estimated_completion and stats.processed > 0:
                            eta = stats.estimated_completion - datetime.now()
                            if eta.total_seconds() > 0:
                                st.markdown(f"**⏳ ETA:** {str(eta).split('.')[0]}")
                    
                    # Processing rate
                    if stats.processed > 0:
                        files_per_minute = (stats.processed / elapsed.total_seconds()) * 60
                        st.markdown(f"**📈 Rate:** {files_per_minute:.1f} files/minute")
            
            # Live logs
            st.markdown("### 📝 Processing Logs")
            
            # Performance tips
            if st.session_state.processor.is_running:
                with st.expander("💡 Understanding Cloud Processing", expanded=False):
                    st.markdown("""
                    **Cloud Processing Pipeline:**
                    
                    1. **📖 Load File** - File data loaded from upload
                    2. **🔄 Convert μ-law→WAV** - CPU-intensive conversion step
                    3. **🎤 Transcribe** - AI transcription via Deepgram API
                    
                    **Cloud Benefits:**
                    - No local file system required
                    - Scalable processing power
                    - Secure - files processed in memory only
                    - Results available for immediate download
                    """)
            
            # Log display
            if hasattr(st.session_state.processor, 'logger'):
                recent_logs = st.session_state.processor.logger.get_recent_logs(30)
                
                if recent_logs:
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
                        height=200,
                        help="Live processing log - updates automatically"
                    )
                else:
                    st.info("No processing logs yet. Start processing to see activity.")
    
    with tab2:
        st.header("📊 Job Monitoring")
        
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
                    "File": result.file_name,
                    "Size (KB)": f"{result.file_size/1024:.1f}",
                    "Time (s)": f"{result.processing_time:.1f}",
                    "Transcript Preview": result.transcript[:50] + "..." if len(result.transcript) > 50 else result.transcript
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        else:
            st.info("No processing results yet. Upload and process some files first!")
        
        # System status
        st.markdown("### 🖥️ System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_key:
                st.success("✅ Deepgram API")
            else:
                st.error("❌ Deepgram API")
        
        with col2:
            if st.session_state.processor.is_running:
                st.warning("🔄 Processing Active")
            else:
                st.success("✅ Ready")
        
        with col3:
            session_id = st.session_state.processor.session_id
            st.info(f"🆔 Session: {session_id}")
    
    with tab3:
        st.header("📄 Results & Downloads")
        
        if st.session_state.processor.results:
            results = st.session_state.processor.results
            total_processed = len(results)
            successful = len([r for r in results if r.status == "completed"])
            failed = len([r for r in results if r.status == "failed"])
            
            # Summary metrics
            st.markdown("### 📊 Processing Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", total_processed)
            with col2:
                st.metric("Successful", successful, delta=successful - failed)
            with col3:
                success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Download options
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_data = st.session_state.processor.export_results_to_csv()
                if csv_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ulaw_transcription_results_{timestamp}.csv"
                    
                    st.download_button(
                        "📊 Download Transcription Results (CSV)",
                        csv_data,
                        filename,
                        "text/csv",
                        use_container_width=True,
                        type="primary"
                    )
            
            with col2:
                # WAV files ZIP download
                zip_data = st.session_state.processor.create_wav_zip()
                if zip_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"converted_wav_files_{timestamp}.zip"
                    
                    st.download_button(
                        "🎵 Download WAV Files (ZIP)",
                        zip_data,
                        zip_filename,
                        "application/zip",
                        use_container_width=True,
                        type="secondary"
                    )
            
            # Individual file previews
            st.markdown("### 🎧 Individual File Results")
            
            for i, result in enumerate(results):
                with st.expander(f"📁 {result.file_name} ({result.status})", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Status:** {result.status}")
                        st.markdown(f"**Processing Time:** {result.processing_time:.2f}s")
                        st.markdown(f"**File Size:** {result.file_size:,} bytes")
                        
                        if result.transcript:
                            st.markdown("**Transcript:**")
                            st.text_area("", result.transcript, height=100, key=f"transcript_{i}")
                        
                        if result.error_message:
                            st.error(f"Error: {result.error_message}")
                    
                    with col2:
                        if result.wav_data and result.status == "completed":
                            st.markdown("**Audio Preview:**")
                            st.audio(result.wav_data, format='audio/wav')
                            
                            # Individual WAV download
                            wav_filename = Path(result.file_name).stem + ".wav"
                            st.download_button(
                                "📥 Download WAV",
                                result.wav_data,
                                wav_filename,
                                "audio/wav",
                                key=f"wav_download_{i}",
                                use_container_width=True
                            )
        else:
            st.info("No results available. Upload and process some files first!")
    
    with tab4:
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Audio Settings")
            st.info("Audio settings are configured per processing session in the main tab.")
            
            st.markdown("### 🔄 Processing Settings")
            
            rate_limit = st.slider(
                "API Rate Limit (calls/minute)",
                min_value=10,
                max_value=100,
                value=40,
                help="Deepgram API calls per minute"
            )
            
            if st.button("Apply Rate Limit"):
                if hasattr(st.session_state.processor.transcriber, 'max_calls_per_minute'):
                    st.session_state.processor.transcriber.max_calls_per_minute = rate_limit
                st.success("Rate limit updated!")
        
        with col2:
            st.markdown("### 📊 System Information")
            
            if api_key:
                st.success("✅ Deepgram API key configured")
            else:
                st.error("❌ Deepgram API key not found")
            
            st.info("🌐 Running in Streamlit Cloud")
            st.info("💾 Files processed in memory (secure)")
            st.info("🔒 No files stored on server")
            
            # Current session info
            session_id = st.session_state.processor.session_id
            st.markdown(f"**Current Session:** `{session_id}`")
        
        # Reset options
        st.markdown("### 🔄 Reset Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Results", type="secondary"):
                st.session_state.processor.results = []
                st.session_state.processing_stats = ProcessingStats()
                st.success("Results cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Session", type="secondary"):
                # Reinitialize processor
                transcriber = RateLimitedTranscriber(api_key)
                st.session_state.processor = CloudBatchProcessor(transcriber)
                st.session_state.processing_stats = ProcessingStats()
                st.success("Session reset!")
                st.rerun()
        
        # Cloud-specific information
        st.markdown("### ☁️ Cloud Processing Information")
        with st.expander("How it works in the cloud", expanded=False):
            st.markdown("""
            **Streamlit Cloud Advantages:**
            - 🔒 **Secure**: Files are processed in memory and never stored on disk
            - 🚀 **Scalable**: Uses cloud computing resources for processing
            - 🌍 **Accessible**: Works from any device with a web browser
            - 💾 **No Storage**: All results are provided as downloads
            
            **File Processing:**
            - Upload files directly through the web interface
            - Files are processed entirely in memory
            - Results are available for immediate download
            - No local file system access required
            
            **Limitations:**
            - File size limited by Streamlit Cloud memory constraints
            - Processing power depends on cloud resources
            - Session data is temporary (clears on restart)
            """)

if __name__ == '__main__':
    main()