import re
import gxipy as gx
import numpy as np
import warnings
from gxipy.ImageFormatConvert import *

def match_device(device_info_list, model_name_query: str):
    """
    Match the provided name query against the device information list, 
    and return the indices of the matching devices.

    Args:
        device_info_list (list): A list of device information dictionaries.
        model_name_query (str): The model name query to search for.

    Returns:
        list: A list of indices of the matching devices.
    """
    matched_indexes = []
    for device in device_info_list:
        if re.search(model_name_query, device['model_name'], re.IGNORECASE):
            matched_indexes.append(device['index'])

    return matched_indexes

def get_best_valid_bits(pixel_format):
    """
    Determine the best valid bit depth for the given pixel format.
    Orignally from the Daheng Imaging Sample Codes for python.

    Args:
        pixel_format (int): The pixel format value.

    Returns:
        int: The valid bit depth for the given pixel format.
    """
    valid_bits = DxValidBit.BIT0_7
    if pixel_format in (GxPixelFormatEntry.MONO8, GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8, GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8
                        , GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8, GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        valid_bits = DxValidBit.BIT0_7
    elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.BAYER_GR10,
                          GxPixelFormatEntry.BAYER_RG10, GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10):
        valid_bits = DxValidBit.BIT2_9
    elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.BAYER_GR12,
                          GxPixelFormatEntry.BAYER_RG12, GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12):
        valid_bits = DxValidBit.BIT4_11
    elif pixel_format in (GxPixelFormatEntry.MONO14):
        valid_bits = DxValidBit.BIT6_13
    elif pixel_format in (GxPixelFormatEntry.MONO16):
        valid_bits = DxValidBit.BIT8_15
    return valid_bits

class GxCam:
    """
    A class for interacting with a Daheng USB Vision3 camera.
    """
    def __init__(self, model_query: str):
        """
        Initialize the GxCam object and open the first matching camera device.

        Args:
            model_query (str): The model name query to search for when opening the camera device.

        Raises:
            RuntimeError: If no devices are found, or the device cannot be initialized.
        """
        # create a device manager
        self._dev_manager = gx.DeviceManager()   # it means init API for whole Gx SDK, it should not be relased

        # create flags
        self._dev_open = False
        self._stream_open = False
        self._trigger_on = False

        dev_num, dev_info_list = self._dev_manager.update_all_device_list()

        idx = match_device(dev_info_list, model_query)

        if dev_num == 0:
            raise RuntimeError("No devices found. Please ensure the camera is connected properly.")
        elif len(idx) == 0:
            raise RuntimeError(f"No devices match the model query: {model_query}. Please check the model name.")
        elif len(idx) > 1:
            warnings.warn(f"Multiple devices match the model query: {model_query}. The first is opened.")

        try:
            # open the first matched device
            self._cam = self._open_device(idx[0])

            # get image convert object
            self._img_convert = self._dev_manager.create_image_format_convert()

            # get remote device feature control
            self._rem_dev_feat = self._cam.get_remote_device_feature_control()

            # get trigger mode and source control
            self._ctl_trigger_mode = self._rem_dev_feat.get_enum_feature("TriggerMode")
            self._ctl_trigger_source = self._rem_dev_feat.get_enum_feature("TriggerSource")

            # get exposure time control (unit: us)
            self._ctl_exposure_time = self._rem_dev_feat.get_float_feature("ExposureTime")

            # check mono or rgb
            pixel_format_value, _ = self._rem_dev_feat.get_enum_feature("PixelFormat").get()
            if not Utility.is_gray(pixel_format_value):
                raise RuntimeError("RGB camera is not supported. Please use a mono camera.")
            
        except RuntimeError as e:
             self._close_device()
             raise RuntimeError(f"Error during initializing the camera: {str(e)}")   
        
        # set pixel format
        try:
            # start continuous acquisition
            self.set_trigger_mode(False)
            self.open_stream()

            # get one frame
            raw_data = self._get_raw_data()

            # get pixel format
            pixel_format = raw_data.get_pixel_format()

            if pixel_format == GxPixelFormatEntry.MONO8:
                self._convert_flag = False

            else:
                self._convert_flag = True

                # get valid bits
                self._valid_bits = get_best_valid_bits(pixel_format)

                # set image convert object
                self._img_convert.set_dest_format(GxPixelFormatEntry.MONO8)
                self._img_convert.set_valid_bits(self._valid_bits)

                # try to convert raw data
                self._convert_raw_data(raw_data)

            # close stream in case software trigger is needed
            self.close_stream()

        except RuntimeError as e:
            self.close_stream()
            self.set_trigger_mode(False)
            self._close_device()
            raise RuntimeError(f"Error during pixel format retrieval or conversion: {str(e)}")

    def __del__(self):
        """
        Close the camera stream, disable trigger mode, and close the device, 
        when the GxCam object is destroyed.
        """
        if self._dev_open:
            self.close_stream()
            self.set_trigger_mode(False)
            self._close_device()

    def _open_device(self, idx: int):
        """
        Open the camera device by the given index.

        Returns:
            gx.Device: The opened camera device.
        """
        if not self._dev_open:
            cam = self._dev_manager.open_device_by_index(idx)
            self._dev_open = True
            return cam
        
    def _close_device(self):
        """
        Close the camera device if it is open.
        """
        if self._dev_open:
            self._cam.close_device()
            self._dev_open = False

    def get_trigger_mode(self):
        """
        Get the current trigger mode of the camera.

        Returns:
            bool: True if the trigger mode is enabled, False otherwise.
        """
        _, trigger_mode = self._ctl_trigger_mode.get()
        if trigger_mode == 'On':
            self._trigger_on = True
        else:
            self._trigger_on = False
        return self._trigger_on
    
    def set_trigger_mode(self, mode: bool = False):
        """
        Set the trigger mode of the camera.

        Raises:
            RuntimeError: If the stream is open and the trigger mode cannot be changed.
        """
        if self._stream_open:
            raise RuntimeError("Cannot change trigger mode while the stream is open. Please close the stream first.")

        if mode:
            self._ctl_trigger_mode.set("On")
            self._ctl_trigger_source.set("Software")
            self.__ctl_trigger = self._rem_dev_feat.get_command_feature("TriggerSoftware")
            self._trigger_on = True
        else:
            self._ctl_trigger_mode.set("Off")
            self.__ctl_trigger = None
            self._trigger_on = False
    
    def get_exposure_time_ms(self):
        """
        Get the current exposure time of the camera in milliseconds.

        Returns:
            float: The exposure time in milliseconds.
        """
        return self._ctl_exposure_time.get() / 1e3
    
    def set_exposure_time_ms(self, et: float):
        """
        Set the exposure time of the camera in milliseconds.
        """
        self._ctl_exposure_time.set(et * 1e3)

    def open_stream(self):
        """
        Open the camera stream if it is not already open.
        """
        if not self._stream_open:
            self._cam.stream_on()
            self._stream_open = True
    
    def close_stream(self):
        """
        Close the camera stream if it is open.
        """
        if self._stream_open:
            self._cam.stream_off()
            self._stream_open = False

    def _trigger(self):
        """
        Trigger the camera if the trigger mode is enabled.

        Raises:
            RuntimeError: If the trigger mode is not enabled.
        """
        if self._trigger_on:
            self.__ctl_trigger.send_command()
        else:
            raise RuntimeError("Trigger mode is not enabled.")
    
    def _get_raw_data(self):
        """
        Get raw data from the camera stream.

        Raises:
            RuntimeError: If the raw data cannot be retrieved.
        """
        # if not self._stream_open:
        #     raise RuntimeError("A stream must be opened before get one frame")
        
        raw_data = self._cam.data_stream[0].get_image()

        if raw_data is None:
            raise RuntimeError("Failed to get image")
        
        return raw_data
    
    def _convert_raw_data(self, raw_data):
        """
        Convert the raw data to a c_ubyte array.

        Returns:
            tuple: A tuple containing the c_ubyte array and the buffer size.
        """

        # create output image buffer
        buffer_out_size = self._img_convert.get_buffer_size_for_conversion(raw_data)
        output_img_array = (c_ubyte * buffer_out_size)()
        output_img = addressof(output_img_array)

        # convert to pixel format
        self._img_convert.convert(raw_data, output_img, buffer_out_size, False)

        if output_img is None:
            raise RuntimeError("Failed to convert pixel format")

        return output_img_array, buffer_out_size
    
    def get_snapshot(self):
        """
        Get a single frame from the camera and convert it to a 2D numpy array.
        Trigger is processed automatically.

        Returns:
            np.ndarray: The image data as a 2D numpy array.
        """
        if self._trigger_on:
            self._trigger()

        raw_data = self._get_raw_data()
        
        if self._convert_flag:
            img_array, buffer_size = self._convert_raw_data(raw_data)
            return np.frombuffer(img_array, dtype=np.ubyte, count=buffer_size). \
                    reshape(raw_data.frame_data.height, raw_data.frame_data.width)
        else:
            return raw_data.get_numpy_array()
    
    def get_images(self, frames: int = 1):
        """
        Get multiple frames from the camera and return them as a 3D numpy array.
        Trigger is processed automatically.

        Returns:
            np.ndarray: The image data as a 3D numpy array.

        Raises:
            ValueError: If the number of frames is not a positive integer.
        """
        img1 = self.get_snapshot()

        if frames > 1:
            img = np.zeros(img1.shape + (frames,), dtype=np.uint8)
            img[:,:,0] = img1
            for i in range(frames-1):
                img[:,:,i+1] = self.get_snapshot()
            return img
        
        elif frames == 1:
            return np.expand_dims(img1, axis=(2))
        
        else:
            raise ValueError("The number of frames must be 1 or greater.")
        