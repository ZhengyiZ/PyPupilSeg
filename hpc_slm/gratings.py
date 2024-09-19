import numpy as np
from typing import Optional, Tuple, Union

def apply_circle_mask(img: np.ndarray, diameter: int = 0):
    """
    This function applies a central circular mask to the input image.
    Pixels outside the circle will be set to 0.

    Args:
        img (np.ndarray): The input image to be masked, it can be 2D or 3D.
        diameter (int): The diameter of the circle. If 0 or not specified, 
                      the function will attempt to use the height of the image as the diameter. 

    Returns:
        np.ndarray: The image after applying the circular mask.
    """
    if diameter == 0:
        diameter = img.shape[0]
    
    center = (diameter - 1) / 2
    radius = diameter / 2
    Y, X = np.ogrid[:diameter, :diameter]
    dist = np.sqrt((X - center)**2 + (Y - center)**2)

    mask = dist > radius
    if len(img.shape) == 3:
        mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
        
    img[mask] = 0

    return img

class Gratings:
    """
    A class to generate and manipulate different types of gratings and patterns 
    on a 2D canvas, with options for masking and placing the gratings in specific locations.
    """
    def __init__(self, size: Tuple[int, int]):
        """
        Initializes the Gratings object with a canvas of specified size.

        Args:
            size (Tuple[int, int]): The coordinates of the canvas (height, width) where the gratings will be generated.
        """
        # Create a mesh grid based on the canvas size
        x = np.arange(1, size[1]+1, step=1)
        y = np.arange(1, size[0]+1, step=1)
        self.xx, self.yy = np.meshgrid(x, y, indexing='xy')

        # Placeholder for the mask to be applied to the canvas
        self.canvas_mask = None
            
    def gen_grating(self, type: str, x: int, y: Optional[int] = None, **kwargs):
        """
        Generates a 2D grating pattern of the specified type and size.

        Args:
            type (str): The type of grating to generate ('blazed' or 'binary').
            x (int): The width of the grating.
            y (Optional[int]): The height of the grating. Defaults to the value of x if not provided.
            grating_step (int): The step size for the grating pattern. Defaults vary depending on the grating type.
            circle_mask (bool): If True, applies a circular mask to the grating. Defaults is True.
            phase_inverse (bool): If True, inverts the phase. Defaults is False.
            rot90_k (int): The number of 90-degree rotations to apply to the grating. Optional.

        Returns:
            np.ndarray: The generated grating pattern as a 2D array.

        Raises:
            ValueError: If an unknown grating type is provided.
        """
        if y is None:
            y = x

        circle_mask = kwargs.get('circle_mask', True)
        phase_inverse = kwargs.get('phase_inverse', False)

        if type == 'blazed':
            grating_step = kwargs.get('grating_step', 10)
            phase = np.linspace(0, 2 * np.pi, grating_step, endpoint=False)
            grating = np.tile(phase, int(np.ceil(x / grating_step)))[:x]

        elif type == 'binary':
            grating_step = kwargs.get('grating_step', 1)
            phase = np.array([0, np.pi]).repeat(grating_step)
            grating = np.tile(phase, int(np.ceil(x / (2 * grating_step))))[:x]

        else:
            raise ValueError(f"Unknown grating type: {type}")

        if phase_inverse:
            grating = 2 * np.pi - grating

        grating_2d = np.tile(grating, (y, 1))

        rot90_k = kwargs.get('rot90_k', None)
        if rot90_k is not None:
            grating_2d = np.rot90(grating_2d, k=int(rot90_k))

        if circle_mask and x == y:
            grating_2d = apply_circle_mask(grating_2d, x)
    
        return grating_2d
    
    def gen_canvas_mask(self, diameter: np.ndarray, center_x: np.ndarray, 
                        center_y: np.ndarray, circle_mask: Optional[bool] = True):
        """
        Generates a mask for the canvas based on specified diameters and center points.

        Args:
            diameter (np.ndarray): An array of diameters for each mask.
            center_x (np.ndarray): An array of x-coordinates for the center of each mask.
            center_y (np.ndarray): An array of y-coordinates for the center of each mask.
            circle_mask (Optional[bool]): If True, generates circular masks. If False, generates square masks. Defaults to True.

        Raises:
            ValueError: If the shapes of 'diameter', 'center_x', and 'center_y' are not the same.
        """
        if not (diameter.shape == center_x.shape == center_y.shape):
            raise ValueError("'diameter', 'center_x', and 'center_y' must have the same shape")
        
        mask = np.zeros(self.xx.shape, dtype=bool)

        # Generate the mask for each specified diameter and center
        for d, cx, cy in zip(diameter, center_x, center_y):
            if circle_mask:
                dist_from_center = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2)
                mask |= dist_from_center <= d / 2
            else:
                mask |= (np.abs(self.xx - cx) <= d / 2) & (np.abs(self.yy - cy) <= d / 2)
        
        return mask # False represents the background

    def place_patterns(self, center_x: np.ndarray, center_y: np.ndarray, *gratings: np.ndarray):
        """
        Places the provided grating patterns on the canvas at the specified center coordinates.

        Args:
            center_x (np.ndarray): Array of x-coordinates for the centers of the gratings.
            center_y (np.ndarray): Array of y-coordinates for the centers of the gratings.
            *gratings (np.ndarray): Grating patterns to be placed on the canvas.

        Returns:
            np.ndarray: The canvas with the gratings placed at the specified positions.

        Raises:
            ValueError: If the number of gratings does not match the number of center coordinates.
        """
        canvas = np.zeros(self.xx.shape)

        if not (center_x.shape[0] == center_y.shape[0] == len(gratings)):
            raise ValueError("The number of gratings must match the length of center_x and center_y.")

        # Place each grating pattern on the canvas
        for i in range(len(gratings)):
            grating = gratings[i]
            g_h, g_w = grating.shape

            # Calculate the area where the grating will be placed
            top_left_y = round(center_y[i] - g_h / 2)
            top_left_x = round(center_x[i] - g_w / 2)
            bottom_right_y = top_left_y + g_h
            bottom_right_x = top_left_x + g_w

            # Ensure the grating does not exceed canvas boundaries
            canvas_top_left_y = max(0, top_left_y)
            canvas_top_left_x = max(0, top_left_x)
            canvas_bottom_right_y = min(canvas.shape[0], bottom_right_y)
            canvas_bottom_right_x = min(canvas.shape[1], bottom_right_x)

            # Handle cases where the grating extends beyond the canvas and needs to be cropped
            grating_top_left_y = max(0, -top_left_y)
            grating_top_left_x = max(0, -top_left_x)
            grating_bottom_right_y = g_h - max(0, bottom_right_y - canvas.shape[0])
            grating_bottom_right_x = g_w - max(0, bottom_right_x - canvas.shape[1])

            # Place the grating onto the canvas
            canvas[canvas_top_left_y:canvas_bottom_right_y, canvas_top_left_x:canvas_bottom_right_x] += \
                grating[grating_top_left_y:grating_bottom_right_y, grating_top_left_x:grating_bottom_right_x]

        return canvas
    
    def apply_canvas_mask(self, canvas: np.ndarray, mask: np.ndarray, background: Optional[np.ndarray] = None):
        """
        Applies the mask to the canvas. 
        Optionally fills the background with a specified pattern.

        Args:
            canvas (np.ndarray): The canvas to apply the mask on.
            mask (np.ndarray): The boolean mask.
            background (Optional[np.ndarray]): An optional background pattern to fill in the unmasked areas.

        Returns:
            np.ndarray: The masked canvas with optional background.

        Raises:
            ValueError: If the canvas mask has not been generated.
        """
        # if self.canvas_mask is None:
        #     raise ValueError("Canvas mask is not generated.")
        
        # Apply the mask to the canvas
        canvas = canvas * mask

        # Apply background pattern if provided
        if background is not None:
            canvas[~mask] = background[~mask]

        return canvas

    def gen_rotational_coors(self, diameter:int, d1: int, beam_diameter: int, 
                             center_x: int, center_y: int, 
                             turns: Optional[int] = 3, numbers: Optional[Union[int, np.ndarray]] = 9,
                             img_flag: Optional[bool] = False):
        """
        Generates rotational coordinates and an optional preview image for multiple positions.

        Args:
            diameter (int): The diameter of the displayed pattern. 
                            To maintain the plane wave approximation, the diameter cannot be too large.
            d1 (int): The distance between overall center and the center of the first turn.
            beam_diameter (int): The overall diameter of the beam (pupil).
            center_x (int): X-coor of the overall center.
            center_y (int): Y-coor of the overall center.
            turns (Optional[int]): The number of turns. Default is 3.
            numbers (Optional[int]): The number of points in each turn. Default is 9.
            img_flag (Optional[bool]): If True, generates an image showing the rotational pattern. Default is False.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): 
                - cx (np.ndarray): Array of x-coor for the centers of the circles.
                - cy (np.ndarray): Array of y-coor for the centers of the circles.
                - img (np.ndarray): The generated image showing the circles, if img_flag is True.
        """
        # Check if numbers is int or np.ndarray
        if isinstance(numbers, int):
             # If it's an integer, expand it to an array with increasing values
             numbers = np.array([numbers * (turn + 1) for turn in range(turns)])
        elif isinstance(numbers, np.ndarray):
            if len(numbers) == 1:
                # If array length is 1, expand it similarly to the int case
                numbers = np.array([numbers[0] * (turn + 1) for turn in range(turns)])
            elif len(numbers) < turns:
                # If the array length is less than the number of turns, raise an error
                raise ValueError(f"Length of `numbers` array should not less than the number of turns ({turns}).")
        
        img = np.zeros(self.xx.shape, dtype = np.uint8)

        if turns > 1:
            d2 = (beam_diameter / 2 - d1 - diameter / 2) / (turns - 1)
        else:
            d2 = 0

        if img_flag:
            central_mask = (self.xx - center_x) ** 2 + (self.yy - center_y) ** 2 <= (diameter / 2) ** 2
            img[central_mask] = 1

        cx_result = [center_x]
        cy_result = [center_y]

        for turn in range(turns):

            radius = d1 + turn * d2  # The radius increases with each turn
            angle_step = 2 * np.pi / numbers[turn]  # Angle step for each turn
            # init_angle = np.pi / numbers / (turn + 1)  # Initial angle for each turn

            for i in range(numbers[turn]):
                angle = angle_step * i
                cx_tmp = round(center_x + radius * np.cos(angle))
                cy_tmp = round(center_y + radius * np.sin(angle))
                
                cx_result.append(cx_tmp)
                cy_result.append(cy_tmp)

        cx = np.array(cx_result)
        cy = np.array(cy_result)

        if img_flag:
            
            idx = len(cx) - 1

            # Draw from the outermost to innermost circles to prevent overlap
            for turn in reversed(range(turns)):
                value = turn + 2
                for i in range(numbers[turn]):
                    mask = (self.xx - cx[idx]) ** 2 + (self.yy - cy[idx]) ** 2 <= (diameter / 2) ** 2
                    img[mask] = value
                    idx -= 1

        return cx, cy, img
