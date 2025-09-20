import cv2
import numpy as np
from typing import Tuple, Optional
import logging

class OMRProcessor:
    """
    Handles image preprocessing for OMR sheets including rotation correction,
    perspective correction, and binary conversion.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray, 
                        enable_perspective_correction: bool = True,
                        enable_rotation_correction: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for OMR sheet images.
        
        Args:
            image: Input image as numpy array
            enable_perspective_correction: Whether to apply perspective correction
            enable_rotation_correction: Whether to apply rotation correction
            
        Returns:
            Preprocessed binary image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Rotation correction
            if enable_rotation_correction:
                corrected = self._correct_rotation(blurred)
            else:
                corrected = blurred
            
            # Perspective correction
            if enable_perspective_correction:
                corrected = self._correct_perspective(corrected)
            
            # Convert to binary image
            binary = self._convert_to_binary(corrected)
            
            # Morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            # Fallback to simple thresholding
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return binary
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct rotation in the image using edge detection.
        """
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate the most common angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi - 90
                    if abs(angle) < 45:  # Only consider small angles
                        angles.append(angle)
                
                if angles:
                    # Use median angle to avoid outliers
                    rotation_angle = np.median(angles)
                    
                    # Only rotate if angle is significant (> 0.5 degrees)
                    if abs(rotation_angle) > 0.5:
                        return self._rotate_image(image, float(rotation_angle))
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Rotation correction failed: {str(e)}")
            return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle while maintaining dimensions.
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct perspective distortion by finding document contours.
        """
        try:
            # Create a copy for contour detection
            contour_image = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(contour_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find the largest contour (likely the document)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a quadrilateral
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we found a quadrilateral, apply perspective correction
            if len(approx) == 4:
                return self._apply_perspective_transform(image, approx)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Perspective correction failed: {str(e)}")
            return image
    
    def _apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation using detected corners.
        """
        # Order the corners: top-left, top-right, bottom-right, bottom-left
        corners = corners.reshape(4, 2)
        ordered_corners = self._order_corners(corners)
        
        # Calculate the dimensions of the corrected image
        width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        width = int(max(float(width_top), float(width_bottom)))
        
        height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
        height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
        height = int(max(float(height_left), float(height_right)))
        
        # Define destination points
        destination = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(
            ordered_corners.astype(np.float32), destination)
        
        # Apply perspective transformation
        corrected = cv2.warpPerspective(image, transform_matrix, (width, height))
        
        return corrected
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in clockwise order starting from top-left.
        """
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center to each corner
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        
        # Reorder to start from top-left (smallest angle in upper half)
        ordered = corners[sorted_indices]
        
        # Ensure we start with top-left corner
        top_corners = ordered[ordered[:, 1] < center[1]]
        if len(top_corners) >= 2:
            # Find leftmost of top corners
            leftmost_top_idx = np.argmin(top_corners[:, 0])
            start_idx = np.where(np.all(ordered == top_corners[leftmost_top_idx], axis=1))[0][0]
            ordered = np.roll(ordered, -start_idx, axis=0)
        
        return ordered
    
    def _convert_to_binary(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to binary using adaptive thresholding.
        """
        try:
            # Use Otsu's thresholding for automatic threshold selection
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # If Otsu doesn't work well, fall back to adaptive thresholding
            mean_val = np.mean(binary)
            if mean_val < 50 or mean_val > 200:  # If result seems poor
                binary = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Binary conversion failed: {str(e)}")
            # Simple thresholding as fallback
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            return binary
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better bubble detection.
        """
        try:
            # Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Slight sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image
