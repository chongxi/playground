from pyaudio import PyAudio, paInt16
import numpy as np

'''
Adapted from pysine.
The original pysine version has a regular noise very few tens of seconds. 
'''


class PySine(object):
    BITRATE = 96000.

    def __init__(self):
        self.pyaudio = PyAudio()
        try:
            self.stream = self.pyaudio.open(
                # format=self.pyaudio.get_format_from_width(1),
                format=paInt16,
                channels=1,
                rate=int(self.BITRATE),
                output=True)
        except:
            logger.error("No audio output is available. Mocking audio stream to simulate one...")
            # output stream simulation with magicmock
            try:
                from mock import MagicMock
            except:  # python > 3.3
                from unittest.mock import MagicMock
            from time import sleep
            self.stream = MagicMock()
            def write(array):
                duration = len(array)/float(self.BITRATE)
                sleep(duration)
            self.stream.write = write

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def sine(self, frequency=440.0, duration=1.0):
        points = int(self.BITRATE * duration)
        try:
            times = np.linspace(0, duration, points, endpoint=False)
            data = np.array((np.sin(times*frequency*2*np.pi) + 1.0)*127.5, dtype=np.int8).tostring()
        except:  # do it without numpy
            data = ''
            omega = 2.0*np.pi*frequency/self.BITRATE
            for i in range(points):
                data += chr(int(127.5*(1.0+sin(float(i)*omega))))
        self.stream.write(data)

PYSINE = PySine()

def neuron_sound(frequency=5.0, duration=0.1):
    return PYSINE.sine(frequency=frequency, duration=duration)