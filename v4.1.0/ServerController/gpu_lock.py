# simple_gpu_lock.py - Simple GPU allocation test
import threading
import time
from contextlib import contextmanager

class gpu_lock:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_user = None
        self.queue_count = 0
    
    @contextmanager
    def acquire(self, client_id, operation_type, timeout=30):
        """Context manager for GPU access"""
        acquired = False
        start_time = time.time()
        
        try:
            self.queue_count += 1
            print(f"🔄 {client_id} waiting for GPU ({operation_type}) - Queue: {self.queue_count}")
            
            acquired = self.lock.acquire(timeout=timeout)
            if acquired:
                self.current_user = f"{client_id}_{operation_type}"
                self.queue_count -= 1
                print(f"🔒 GPU locked by {self.current_user}")
                yield True
            else:
                self.queue_count -= 1
                print(f"⏰ {client_id} GPU timeout after {timeout}s")
                yield False
                
        finally:
            if acquired:
                elapsed = time.time() - start_time
                print(f"🔓 GPU released by {self.current_user} after {elapsed:.2f}s")
                self.current_user = None
                self.lock.release()

# Global instance
gpu_lock = gpu_lock()