import unittest
import os
import importlib.util

class TestAnomalyDetection(unittest.TestCase):

    def test_library_opencv_installed(self):
        """ Test if OpenCV library is installed """
        opencv_installed = importlib.util.find_spec("cv2") is not None
        self.assertTrue(opencv_installed, "OpenCV library is not installed")

    def test_library_numpy_installed(self):
        """ Test if numpy library is installed """
        numpy_installed = importlib.util.find_spec("numpy") is not None
        self.assertTrue(numpy_installed, "numpy library is not installed")
        
    def test_library_anomalib_installed(self):
        """ Test if anomalib library is installed """
        anomalib_installed = importlib.util.find_spec("anomalib") is not None
        self.assertTrue(anomalib_installed, "anomalib library is not installed")
        
    def test_library_openvino_installed(self):
        """ Test if openvino library is installed """
        openvino_installed = importlib.util.find_spec("openvino") is not None
        self.assertTrue(openvino_installed, "openvino library is not installed")
        
    def test_library_matplotlib_installed(self):
        """ Test if matplotlib library is installed """
        matplotlib_installed = importlib.util.find_spec("matplotlib") is not None
        self.assertTrue(matplotlib_installed, "numpy library is not installed")

    def test_config_file_exists(self):
        """ Test if the config file exists """
        config_file_path = "/mnt/code/padim_config.yaml"
        self.assertTrue(os.path.isfile(config_file_path), "Config file does not exist")

if __name__ == '__main__':
    unittest.main()
