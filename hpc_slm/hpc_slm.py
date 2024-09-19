from ctypes import windll, c_int, c_uint8, POINTER
from typing import Optional, Tuple, Union
from pathlib import Path
import warnings
import numpy as np
import h5py


def expand_array(arr: np.ndarray, scale: int):
    """
    Expands the input array by a given scale factor using nearest neighbor interpolation.

    Args:
        arr (np.ndarray): Input 2D array to be expanded.
        scale (int): Factor by which to scale the array.

    Returns:
        np.ndarray: Scaled array with expanded dimensions.
    """
    arr = np.repeat(arr, scale, axis=0)
    arr = np.repeat(arr, scale, axis=1)
    return arr


def wrap_to_2pi(arr: np.ndarray):
    """
    Wraps angles in the array to the interval [0, 2*pi].
    Translated from MATLAB wrapTo2Pi.

    Args:
        arr (np.ndarray): Array of angles in radians.

    Returns:
        np.ndarray: Array with angles wrapped to [0, 2*pi].
    """
    positive_input = arr > 0
    # arr = np.mod(arr, 2 * np.pi)
    arr %= 2 * np.pi
    arr[(arr == 0) & positive_input] = 2 * np.pi
    return arr


# Base class for managing window and displaying gray value arrays (uint8) in an SLM
class HPC_SLM:

    def __init__(
        self, monitor_idx: int, win_idx: int, size: Optional[list[int]] = None
    ):
        """
        Initializes the HPC_SLM class, loading the necessary DLL and setting up the display window.

        Args:
            monitor_idx (int): Index of the monitor to be used.
            win_idx (int): Index of the window to be used.
            size (list[int], optional): Size of the display (width, height).
        """
        # Load the Image_Control DLL
        script_dir = Path(__file__).resolve().parent
        dll_path = script_dir / "Image_Control.dll"
        self._dll = windll.LoadLibrary(str(dll_path))

        # Initialize the window for display
        self.init_window(monitor_idx, win_idx)

        # Set up the function to display the array on the SLM
        self._array_to_display = self._dll.Window_Array_to_Display
        self._array_to_display.argtypes = [POINTER(c_uint8), c_int, c_int, c_int, c_int]
        self._array_to_display.restype = c_int

        # Initialize prefix size settings
        self.cancel_prefix_size()
        if size is not None:
            self.set_prefix_size(size)

    def __del__(self):
        """
        Destructor to ensure the display window is properly terminated.
        """
        self.terminate_window()

    def set_prefix_size(self, size: list[int]):
        """
        Sets the prefix size for the display to avoid unnessary calculation of sizes.

        Args:
            size (list[int]): Tuple containing the width and height of the display.
        """
        try:
            self._height, self._width = size
            self._total_size = self._width * self._height
            self._c_array = c_uint8 * self._total_size
            self._prefix_flag = True
        except (TypeError, IndexError) as e:
            self._prefix_flag = False
            warnings.warn(f"Invalid size format: {e}")

    def cancel_prefix_size(self):
        """
        Cancels any prefix size setting.
        """
        self._prefix_flag = False

    def init_window(self, monitor_idx: int, win_idx: int):
        """
        Initializes the display window on the specified monitor.

        Args:
            monitor_idx (int): Index of the monitor to be used.
            win_idx (int): Index of the window to be used.

        Raises:
            RuntimeError: If the window initialization fails.
        """
        self._win_idx = win_idx

        init_win = self._dll.Window_Settings
        init_win.argtypes = [c_int, c_int, c_int, c_int]
        init_win.restype = c_int
        result = init_win(monitor_idx, win_idx, 0, 0)

        if result == 0:
            self._have_win = True
        else:
            self._have_win = False
            raise RuntimeError(f"Window_Settings failed with error code {result}")

    def terminate_window(self):
        """
        Terminates the display window.

        Raises:
            RuntimeError: If the window termination fails.
        """
        term_win = self._dll.Window_Term
        term_win.argtypes = [c_int]
        term_win.restype = c_int
        result = term_win(self._win_idx)
        if result == 0:
            self._have_win = False
        else:
            raise RuntimeError(f"Window_Term failed with error code {result}")

    def display_img(self, img: np.ndarray):
        """
        Displays an image (gray value array) on the SLM.

        Args:
            img (np.ndarray): 2D array (dtype=uint8) representing the image to be displayed.

        Raises:
            RuntimeError: If the display window is not initialized or if displaying the array fails.
        """
        if not self._have_win:
            raise RuntimeError("Display window is not initialized.")

        if self._prefix_flag:
            img_array = self._c_array.from_buffer(img)
            result = self._array_to_display(
                img_array, self._width, self._height, self._win_idx, self._total_size
            )
        else:
            y, x = img.size()
            total_size = x * y
            c_array = c_uint8 * total_size
            img_array = c_array.from_buffer(img)
            result = self._array_to_display(img_array, x, y, self._win_idx, total_size)

        if result != 0:
            raise RuntimeError(
                f"Window_Array_to_Display failed with error code {result}"
            )


# Derived class with calibration capabilities, using local or global LUTs to calculate the phase into gray values (uint8)
class HPC_SLM_Calib(HPC_SLM):
    def __init__(
        self,
        monitor_idx: int,
        win_idx: int,
        size: Optional[list[int]] = None,
        calib_file: Optional[str] = None,
    ):
        """
        Initializes the SLM with calibration data, if provided.

        Args:
            monitor_idx (int): The monitor index where the SLM window should be displayed.
            win_idx (int): The window index to be used for display.
            size (list[int], optional): Size of the SLM display (width, height).
            calib_file (str, optional): Path to the calibration file.
        """
        super().__init__(monitor_idx, win_idx, size)

        # Initialize the calibration settings
        self.unload_calib_file()
        if calib_file is not None:
            self.load_calib_file(calib_file)

    def unload_calib_file(self):
        """
        Resets the calibration data to default values, disabling local and flat corrections.
        """
        self._avg_lut = 255.0
        self._local_flag = False
        self._flat_flag = False

    def load_calib_file(self, calib_file: str):
        """
        Loads calibration data from an HDF5 file.

        Args:
            calib_file (str): Path to the calibration file.

        Raises:
            KeyError: If required calibration data is not found in the file.
        """
        with h5py.File(calib_file, "r") as f:
            # get the flat phase
            try:
                flat_phs_rad = f["/calib/flat_phase_rad"][()]
                phs_bin = f["/calib/flat_phase_rad"].attrs["bin"]

                # if binned, interpolate
                if phs_bin > 1:
                    flat_phs_rad = expand_array(flat_phs_rad, int(phs_bin))

                self.flat_phs = flat_phs_rad
                self._flat_flag = True
            except KeyError:
                self._flat_flag = False
                warnings.warn(
                    "Flat phase calibration data not found in the file. Flat phase correction will be disabled.",
                    UserWarning,
                )

            # get global average LUT
            self._avg_lut = f["/calib/average_LUT_center"][()]

            # get local non-linear LUT
            try:
                if_coeff = f["/calib/IF_coefficients"][()]
                coeff_bin = f["/calib/IF_coefficients"].attrs["bin"]

                if_coeff = if_coeff.transpose(1, 2, 0)

                if coeff_bin > 1:
                    page_size = np.array(if_coeff.shape[0:2]) * coeff_bin
                    page_size = page_size.astype(int)
                    if_coeff_tmp = np.zeros(
                        (*page_size, if_coeff.shape[2]), dtype=np.float32
                    )

                    for i in range(if_coeff.shape[2]):
                        if_coeff_tmp[:, :, i] = expand_array(
                            if_coeff[:, :, i], int(coeff_bin)
                        )

                    if_coeff = if_coeff_tmp

                self.if_coeff = if_coeff
                self._local_flag = True

            except KeyError:
                self._local_flag = False
                warnings.warn(
                    "Local non-linear LUT data not found in the file. Local LUT correction will be disabled.",
                    UserWarning,
                )

    def phs_to_display(
        self,
        phs: np.ndarray,
        flat_flag: Optional[bool] = None,
        local_flag: Optional[bool] = None,
        global_lut: Optional[Union[int, float]] = None,
    ):
        """
        Converts phase values (in radians) to displayable gray values (uint8) on the SLM.

        Args:
            phs (np.ndarray): A 2D array of phase values (in radians) that need to be displayed.
            flat_flag (bool, optional): If True, applies flat phase correction using pre-loaded flat phase data.
            local_flag (bool, optional): If True, applies local non-linear LUT correction using pre-loaded coefficients.
            global_lut (int or float, optional): Global linear LUT scale factor to convert phase to gray values.
                                                 If not provided, using pre-loaded value.

        This method processes the phase data through several steps:
        - Applies flat phase correction if requested.
        - Wraps the phase values to the range [0, 2*pi].
        - Applies local non-linear LUT correction if requested, or uses global linear LUT for conversion.
        - Finally, converts the resulting values to uint8 format and displays the image on the SLM.
        """
        # Apply flat phase correction if requested and available
        if flat_flag is not None and flat_flag:
            if self._flat_flag:
                phs += (
                    self.flat_phs
                )  # In-place addition: add the flat phase to the input phase
            else:
                pass
                # warnings.warn(
                #     "Flat phase correction requested, but no flat phase data is loaded."
                # )

        # Ensure all phase values are non-negative by subtracting the minimum value
        if np.any(phs < 0):
            phs -= np.min(phs)

        # Wrap phase values to the range [0, 2*pi]
        phs = wrap_to_2pi(phs)

        # Apply local LUT correction if requested and available
        if local_flag is not None and local_flag and self._local_flag:
            # Fast polynomial evaluation of the local LUT
            img = np.zeros_like(phs)
            for i in range(self.if_coeff.shape[2]):
                # In-place operations
                img *= phs
                img += self.if_coeff[:, :, i]
        else:
            # if local_flag and not self._local_flag:
            #     warnings.warn(
            #         "Local LUT correction requested, but no local LUT data is loaded."
            #     )

            # Use global LUT if provided, otherwise fall back to the pre-loaded value
            if global_lut is not None:
                img = phs / 2 / np.pi * global_lut
            elif self._avg_lut is not None:
                img = phs / 2 / np.pi * self._avg_lut
            else:
                raise ValueError(
                    "No global LUT provided, and no average LUT is loaded."
                )

        img = img.astype(np.uint8)

        self.display_img(img)
