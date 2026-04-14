import cv2
import queue
import threading
import time

class VideoStream:
    def __init__(self, source, queue_size=5):
        """
        Initializes an async video stream using the Producer pattern.
        This prevents the slow cv2.read() IO calls from blocking the GPU.
        """
        self.stream = cv2.VideoCapture(source)
        if not self.stream.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # Metadata
        self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stopped = False
        
        # We use a LIFO queue or standard queue. Standard queue is fine, but we might want LIFO to always get the latest frame
        # For video files, we don't want to skip frames usually. For webcams, we do.
        self.is_live = isinstance(source, int)
        
        # A simple bounded queue
        self.Q = queue.Queue(maxsize=queue_size)
        
        # Start the background capture thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Producer Thread: Continuously grabs frames and places them in the queue."""
        while not self.stopped:
            # If the queue is bursting, wait a tiny bit to not overwhelm memory
            if self.Q.full():
                if self.is_live:
                    # Drop old frame to keep it real-time
                    try:
                        self.Q.get_nowait()
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.01)
                    continue
                
            ret, frame = self.stream.read()
            
            if not ret:
                self.stopped = True
                return
                
            self.Q.put(frame)

    def read(self):
        """Consumer Method: Grabs the next frame from the buffer."""
        try:
            return True, self.Q.get(timeout=10.0)
        except queue.Empty:
            return False, None

    def more(self):
        """Returns True if there are still frames in the Queue."""
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        """Stop the stream and release resources."""
        self.stopped = True
        self.thread.join()
        self.stream.release()
