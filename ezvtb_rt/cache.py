from typing import Set
import numpy as np
from collections import OrderedDict
import os
from queue import Queue
import turbojpeg
from typing import List,Dict
import threading

"""
Thread-safe cache system with JPEG compression and LRU eviction.
Handles high-throughput image data with background compression/eviction.
Uses TurboJPEG for efficient GPU-accelerated compression/decompression.
"""

def threadCompressSave(cache:OrderedDict, lock:threading.Lock, queue:Queue, max_volume:int, cache_quality:int):
    """Background worker thread for cache compression and management.
    
    Args:
        cache: OrderedDict storing compressed entries (LRU)
        lock: Thread lock for cache access
        queue: Input queue of (hash, raw_data) tuples to process  
        max_volume: Maximum cache size in gigabytes
        cache_quality: JPEG quality (1-100) for compression
    """
    max_kbytes = max_volume * 1024 * 1024  # Convert GB to KB
    cached_kbytes = 0  # Tracks total cached data size
    
    while True:
        # Get next item from processing queue
        hs, data = queue.get(block=True)
        
        # Check if already cached
        lock.acquire(blocking=True)
        cached = cache.get(hs)
        lock.release()
        if cached is not None:
            continue  # Skip if already exists
            
        # Pack RGBA data into 1024x512 buffer:
        # - Top 512 rows: RGB channels 
        # - Bottom 512 rows: Alpha channel in R
        new_data = np.zeros((1024, 512, 4), data.dtype)
        new_data[:512, :, :] = data  # Store RGB
        new_data[512:, :, 0] = data[:,:,3]  # Store alpha

        # Compress with TurboJPEG - different settings for lossless vs lossy
        if cache_quality == 100:
            # Lossless compression settings
            compressed = turbojpeg.compress(
                new_data, cache_quality, 
                turbojpeg.SAMP.Y420, 
                lossless=True, 
                fastdct=False,  # Higher quality DCT
                optimize=True,  # Optimize Huffman tables
                pixelformat=turbojpeg.BGRA
            )
        else:
            # Lossy compression settings
            compressed = turbojpeg.compress(
                new_data, cache_quality,
                turbojpeg.SAMP.Y444,  # Full chroma sampling
                fastdct=False,
                optimize=True,
                pixelformat=turbojpeg.BGRA  
            )
        # Add to cache and update size tracking
        lock.acquire(blocking=True)
        cache[hs] = compressed
        lock.release()
        cached_kbytes += len(compressed) / 1024  # Track size in KB
        
        # LRU eviction when over capacity
        while cached_kbytes > max_kbytes:
            lock.acquire(blocking=True)
            poped = cache.popitem(last=False)  # Remove oldest entry
            lock.release()
            cached_kbytes -= len(poped[1]) / 1024
            poped = None  # Allow GC


class Cacher:
    """Main cache interface with background compression thread.
    
    Attributes:
        cache: OrderedDict storing compressed entries
        lock: Thread synchronization lock
        queue: Compression task queue
        hits: Total cache hits
        miss: Total cache misses
        cache_quality: JPEG compression quality (1-100)
        continues_hits: Counter for sequential hits (anti-thrashing)
        last_hs: Last accessed hash key
    """
    
    def __init__(self, max_volume:float = 2.0, cache_quality:int = 90):
        """Initialize cache with specified size and quality.
        
        Args:
            max_volume: Maximum cache size in gigabytes (default 2.0)
            cache_quality: JPEG compression quality (default 90)
        """
        self.cache = OrderedDict()  # LRU cache storage
        self.lock = threading.Lock()  # Cache access synchronization
        self.queue = Queue()  # Background processing queue

        # Performance tracking
        self.hits = 0  # Total successful cache retrievals
        self.miss = 0  # Total cache misses
        self.cache_quality = cache_quality  # Compression quality setting

        # Start background compression thread
        self.thread = threading.Thread(
            target=threadCompressSave,
            args=(self.cache, self.lock, self.queue, max_volume, cache_quality),
            daemon=True
        )
        self.thread.start()
        
        # Cache state tracking
        self.temp_data = None  # Temporary buffer (unused in current code)
        self.continues_hits = 0  # Sequential hit counter
        self.last_hs = -1  # Last accessed hash key
    def read(self, hs:int) -> np.ndarray:
        """Retrieve cached data by hash key.
        
        Args:
            hs: Hash key of requested data
            
        Returns:
            np.ndarray: Decompressed image data or None if miss
        """
        # Check cache existence
        self.lock.acquire(blocking=True)
        cached = self.cache.get(hs)
        self.lock.release()
        
        # Anti-thrashing: Force miss after 5 sequential hits
        if self.continues_hits > 5:
            cached = None
            
        if cached is not None:
            # Update hit tracking
            if self.last_hs != hs:
                self.continues_hits += 1  # Increment sequential counter
            else:
                self.continues_hits = 0  # Reset if same key
            self.last_hs = hs
            self.hits += 1
            
            # Promote to MRU position
            self.lock.acquire(blocking=True)
            self.cache.move_to_end(hs)
            self.lock.release()
            
            # Decompress and unpack RGBA data
            res = turbojpeg.decompress(
                cached, 
                fastdct=False,  # High quality DCT
                fastupsample=False,  # Quality upsampling
                pixelformat=turbojpeg.BGRA
            )
            decompressed_img = np.ndarray((1024,512,4), dtype=np.uint8, buffer=res)
            result_img = decompressed_img[:512,:,:]  # Extract RGB
            result_img[:,:,3] = decompressed_img[512:,:,0]  # Restore alpha
            return result_img
        else:
            # Update miss tracking
            self.miss += 1
            self.continues_hits = 0
            self.last_hs = hs
            return None
    def write(self, hs:int, data:np.ndarray):
        """Queue data for caching.
        
        Args:
            hs: Hash key for the data
            data: Raw image data to cache (will be copied)
        """
        self.queue.put_nowait((hs, data.copy()))  # Copy to avoid mutation
