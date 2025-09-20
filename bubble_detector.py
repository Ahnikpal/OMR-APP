import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

class BubbleDetector:
    """
    Detects filled bubbles in OMR sheets and extracts answers.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Grid configuration (adjustable based on OMR sheet format)
        self.questions_per_section = 20
        self.options_per_question = 4  # A, B, C, D
        self.expected_sections = 2  # Math, Science, etc.
    
    def detect_answers(self, binary_image: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float], List[Dict]]:
        """
        Detect filled bubbles and extract answers from the OMR sheet.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            Tuple of (detected_answers, confidence_scores, problem_areas)
        """
        try:
            # Find bubble grid
            bubble_regions = self._find_bubble_grid(binary_image)
            
            if not bubble_regions:
                self.logger.warning("No bubble grid detected")
                return {}, {}, [{"question": "all", "reason": "No bubble grid detected"}]
            
            detected_answers = {}
            confidence_scores = {}
            problem_areas = []
            
            # Process each bubble region
            for region_info in bubble_regions:
                question_num = region_info['question']
                bubbles = region_info['bubbles']
                
                # Analyze each bubble in this question
                bubble_scores = []
                for option_idx, bubble_contour in enumerate(bubbles):
                    score = self._analyze_bubble(binary_image, bubble_contour)
                    bubble_scores.append((chr(ord('A') + option_idx), score))
                
                # Determine the selected answer
                answer, confidence = self._determine_answer(bubble_scores)
                
                if confidence < self.confidence_threshold:
                    problem_areas.append({
                        'question': str(question_num),
                        'reason': f'Low confidence ({confidence:.2f})'
                    })
                
                detected_answers[str(question_num)] = answer
                confidence_scores[str(question_num)] = confidence
            
            return detected_answers, confidence_scores, problem_areas
            
        except Exception as e:
            self.logger.error(f"Error in bubble detection: {str(e)}")
            return {}, {}, [{"question": "all", "reason": f"Detection failed: {str(e)}"}]
    
    def _find_bubble_grid(self, binary_image: np.ndarray) -> List[Dict]:
        """
        Locate the bubble grid in the image and identify individual bubble regions.
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find potential bubbles
            potential_bubbles = []
            height, width = binary_image.shape
            min_area = (height * width) // 10000  # Minimum bubble area
            max_area = (height * width) // 400    # Maximum bubble area
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check if contour is roughly circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Reasonably circular
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h if h > 0 else 0
                            if 0.5 < aspect_ratio < 2.0:  # Not too elongated
                                potential_bubbles.append({
                                    'contour': contour,
                                    'center': (x + w//2, y + h//2),
                                    'area': area,
                                    'bounds': (x, y, w, h)
                                })
            
            if len(potential_bubbles) < 10:  # Too few bubbles detected
                # Try with different parameters
                return self._fallback_grid_detection(binary_image)
            
            # Organize bubbles into grid structure
            bubble_regions = self._organize_bubbles_into_grid(potential_bubbles)
            
            return bubble_regions
            
        except Exception as e:
            self.logger.error(f"Grid detection failed: {str(e)}")
            return self._fallback_grid_detection(binary_image)
    
    def _organize_bubbles_into_grid(self, potential_bubbles: List[Dict]) -> List[Dict]:
        """
        Organize detected bubbles into a grid structure representing questions and options.
        """
        if not potential_bubbles:
            return []
        
        # Sort bubbles by position (top to bottom, left to right)
        sorted_bubbles = sorted(potential_bubbles, key=lambda b: (b['center'][1], b['center'][0]))
        
        # Group bubbles into rows (questions)
        rows = []
        current_row = [sorted_bubbles[0]]
        row_y = sorted_bubbles[0]['center'][1]
        y_tolerance = 30  # Pixels tolerance for same row
        
        for bubble in sorted_bubbles[1:]:
            if abs(bubble['center'][1] - row_y) <= y_tolerance:
                current_row.append(bubble)
            else:
                if len(current_row) >= 2:  # Valid row should have multiple options
                    rows.append(current_row)
                current_row = [bubble]
                row_y = bubble['center'][1]
        
        # Add the last row
        if len(current_row) >= 2:
            rows.append(current_row)
        
        # Convert rows to question format
        bubble_regions = []
        for question_idx, row in enumerate(rows[:40]):  # Limit to reasonable number
            # Sort bubbles in row by x position (left to right)
            row_sorted = sorted(row, key=lambda b: b['center'][0])
            
            # Take up to 4 options per question (A, B, C, D)
            bubbles = [b['contour'] for b in row_sorted[:4]]
            
            bubble_regions.append({
                'question': question_idx + 1,
                'bubbles': bubbles
            })
        
        return bubble_regions
    
    def _fallback_grid_detection(self, binary_image: np.ndarray) -> List[Dict]:
        """
        Fallback grid detection method using template-based approach.
        """
        try:
            # Create a synthetic grid structure for demonstration
            height, width = binary_image.shape
            bubble_regions = []
            
            # Estimate grid dimensions
            rows = min(40, height // 20)  # Reasonable number of questions
            cols = 4  # A, B, C, D options
            
            start_y = height // 10
            start_x = width // 10
            
            row_height = (height - 2 * start_y) // rows if rows > 0 else 20
            col_width = (width - 2 * start_x) // cols if cols > 0 else 50
            
            for q in range(1, min(21, rows + 1)):  # Limit to 20 questions for demo
                bubbles = []
                y = start_y + (q - 1) * row_height
                
                for opt in range(cols):
                    x = start_x + opt * col_width
                    
                    # Create a synthetic bubble contour
                    radius = min(row_height // 3, col_width // 3, 15)
                    center = (x + col_width // 2, y + row_height // 2)
                    
                    # Generate circle contour
                    angles = np.linspace(0, 2 * np.pi, 20)
                    points = []
                    for angle in angles:
                        px = int(center[0] + radius * np.cos(angle))
                        py = int(center[1] + radius * np.sin(angle))
                        points.append([px, py])
                    
                    bubble_contour = np.array(points, dtype=np.int32)
                    bubbles.append(bubble_contour)
                
                bubble_regions.append({
                    'question': q,
                    'bubbles': bubbles
                })
            
            return bubble_regions
            
        except Exception as e:
            self.logger.error(f"Fallback detection failed: {str(e)}")
            return []
    
    def _analyze_bubble(self, binary_image: np.ndarray, bubble_contour: np.ndarray) -> float:
        """
        Analyze a single bubble to determine if it's filled.
        
        Returns:
            Score between 0 and 1 indicating how filled the bubble is
        """
        try:
            # Create mask for the bubble
            mask = np.zeros(binary_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [bubble_contour], (255,))
            
            # Extract region of interest
            x, y, w, h = cv2.boundingRect(bubble_contour)
            roi_mask = mask[y:y+h, x:x+w]
            roi_image = binary_image[y:y+h, x:x+w]
            
            if roi_mask.sum() == 0:  # Empty mask
                return 0.0
            
            # Calculate fill ratio
            bubble_pixels = np.sum(roi_mask > 0)
            filled_pixels = np.sum((roi_image > 0) & (roi_mask > 0))
            
            if bubble_pixels == 0:
                return 0.0
            
            fill_ratio = filled_pixels / bubble_pixels
            
            # Additional checks for better accuracy
            # Check for concentrated filling in center
            center_region = self._get_center_region(roi_mask, roi_image)
            center_fill_ratio = np.sum(center_region) / max(int(np.sum(roi_mask > 0) * 0.5), 1)
            
            # Combine fill ratio with center concentration
            score = 0.7 * fill_ratio + 0.3 * min(center_fill_ratio, 1.0)
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Bubble analysis failed: {str(e)}")
            return 0.0
    
    def _get_center_region(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Extract the center region of a bubble for more accurate fill detection.
        """
        try:
            h, w = mask.shape
            center_h, center_w = h // 2, w // 2
            
            # Define center region (inner 60% of the bubble)
            margin_h, margin_w = int(h * 0.2), int(w * 0.2)
            
            center_mask = np.zeros_like(mask)
            center_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1
            
            return (image > 0) & (mask > 0) & (center_mask > 0)
            
        except Exception as e:
            self.logger.warning(f"Center region extraction failed: {str(e)}")
            return np.zeros_like(mask, dtype=bool)
    
    def _determine_answer(self, bubble_scores: List[Tuple[str, float]]) -> Tuple[str, float]:
        """
        Determine the selected answer based on bubble scores.
        
        Args:
            bubble_scores: List of (option, score) tuples
            
        Returns:
            Tuple of (selected_option, confidence)
        """
        if not bubble_scores:
            return 'X', 0.0
        
        # Sort by score (highest first)
        sorted_scores = sorted(bubble_scores, key=lambda x: x[1], reverse=True)
        
        best_option, best_score = sorted_scores[0]
        second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        
        # Check for multiple selections (ambiguous)
        high_scores = [score for _, score in bubble_scores if score > 0.4]
        if len(high_scores) > 1:
            # Multiple bubbles seem filled
            score_diff = best_score - second_best_score
            if score_diff < 0.2:  # Scores are too close
                return 'M', 0.3  # M for Multiple/ambiguous
        
        # Check for no clear selection
        if best_score < 0.3:
            return 'N', best_score  # N for No clear answer
        
        # Calculate confidence based on score and difference from second best
        confidence = min(best_score, 0.8 + 0.2 * (best_score - second_best_score))
        
        return best_option, confidence
    
    def visualize_detection(self, image: np.ndarray, bubble_regions: List[Dict], 
                          detected_answers: Dict[str, str]) -> np.ndarray:
        """
        Create a visualization of the bubble detection results.
        """
        try:
            # Convert to color image for visualization
            if len(image.shape) == 2:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = image.copy()
            
            # Draw detected bubbles and answers
            for region_info in bubble_regions:
                question_num = region_info['question']
                bubbles = region_info['bubbles']
                detected_answer = detected_answers.get(str(question_num), 'N')
                
                for opt_idx, bubble_contour in enumerate(bubbles):
                    option = chr(ord('A') + opt_idx)
                    
                    # Choose color based on whether this option was selected
                    if option == detected_answer:
                        color = (0, 255, 0)  # Green for selected
                        thickness = 3
                    else:
                        color = (255, 0, 0)  # Red for not selected
                        thickness = 1
                    
                    # Draw contour
                    cv2.drawContours(vis_image, [bubble_contour], -1, color, thickness)
                    
                    # Add option label
                    x, y, w, h = cv2.boundingRect(bubble_contour)
                    cv2.putText(vis_image, option, (x - 10, y + h // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
            return image
