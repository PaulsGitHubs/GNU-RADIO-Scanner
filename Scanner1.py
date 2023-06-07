from gnuradio import gr
import numpy as np
from scipy import signal as sig
import pmt

class ScanBlock(gr.sync_block):
    def __init__(self, sample_rate=1.0, gain=20, ppm=0, threshold=0.75, lo=-125000000, start=93000000, stop=95500000, step=100000):
        gr.sync_block.__init__(self, name="ScanBlock", in_sig=[np.complex64], out_sig=None)
        self.sample_rate = sample_rate
        self.gain = gain
        self.ppm = ppm
        self.threshold = threshold
        self.lo = lo
        self.start = start
        self.stop = stop
        self.step = step
        
        # Define an output message port
        self.message_port_register_out(pmt.intern('detected_signals'))

    def work(self, input_items, output_items):
        in0 = input_items[0]
        sample_rate = self.sample_rate

        iq_samples = in0

        # Decimating the samples
        iq_samples = sig.decimate(iq_samples, 48)

        # Performing Welch's method on the samples
        f, psd = sig.welch(iq_samples, fs=sample_rate / 48, nperseg=1024)

        peak_indices, frequencies = self.find_highest_magnitudes(psd, num_peaks=1, sample_rate=sample_rate / 48, fft_size=1024)

        if peak_indices:
            peak_index = peak_indices[0]
            peak_frequency = frequencies[0]
            peak_psd = psd[peak_index]
            
            # If a strong signal is detected
            if peak_psd > self.threshold:
                print(f"Strong signal detected at {peak_frequency} Hz, PSD: {peak_psd}")
                
                # Send this data as a dictionary to the next block
                msg_dict = pmt.make_dict()
                msg_dict = pmt.dict_add(msg_dict, pmt.intern('peak_frequency'), pmt.from_double(peak_frequency))
                msg_dict = pmt.dict_add(msg_dict, pmt.intern('peak_psd'), pmt.from_double(peak_psd))
                
                self.message_port_pub(pmt.intern('detected_signals'), msg_dict)

        # Consume all available samples from the input
        return len(in0)

    @staticmethod
    def find_highest_magnitudes(data, num_peaks=5, sample_rate=2.048e6, fft_size=1024):
        if len(data) < num_peaks:
            print("Not enough data points to find the desired number of peaks.")
            return [], []

        peak_indices = np.argpartition(data, -num_peaks)[-num_peaks:]
        peak_indices = peak_indices[np.argsort(-data[peak_indices])]
        bin_width = sample_rate / fft_size
        frequencies = peak_indices * bin_width
        return peak_indices, frequencies
