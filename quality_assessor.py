"""
Quality assessment module for OMR images and detection results.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

class QualityAssessor:
    """Assesses image quality and detection confidence for OMR processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            'min_resolution': {'width': 400, 'height': 500},
            'max_rotation_angle': 15,  # degrees
            'min_contrast_ratio': 0.3,
            'blur_threshold': 100,  # Laplacian variance
            'noise_tolerance': 0.1,
            'min_bubble_count': 10,
            'min_avg_confidence': 0.4
        }
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive image quality assessment.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing quality metrics and assessment
        """
        try:
            assessment = {
                'overall_quality': 'good',
                'quality_score': 0.0,
                'issues': [],
                'recommendations': [],
                'metrics': {}
            }
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            assessment['metrics']['resolution'] = {'width': width, 'height': height}
            
            # Resolution check
            resolution_score = self._assess_resolution(width, height, assessment)
            
            # Contrast assessment
            contrast_score = self._assess_contrast(gray, assessment)
            
            # Blur assessment
            blur_score = self._assess_blur(gray, assessment)
            
            # Rotation assessment
            rotation_score = self._assess_rotation(gray, assessment)
            
            # Noise assessment
            noise_score = self._assess_noise(gray, assessment)
            
            # Calculate overall quality score (0-100)
            scores = [resolution_score, contrast_score, blur_score, rotation_score, noise_score]
            assessment['quality_score'] = np.mean(scores)
            
            # Determine overall quality level
            if assessment['quality_score'] >= 80:
                assessment['overall_quality'] = 'excellent'
            elif assessment['quality_score'] >= 65:
                assessment['overall_quality'] = 'good'
            elif assessment['quality_score'] >= 50:
                assessment['overall_quality'] = 'fair'
            else:
                assessment['overall_quality'] = 'poor'
            
            # Generate recommendations
            self._generate_quality_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {
                'overall_quality': 'unknown',
                'quality_score': 0.0,
                'issues': [f'Assessment failed: {str(e)}'],
                'recommendations': ['Manual review required'],
                'metrics': {}
            }
    
    def assess_detection_quality(self, detected_answers: Dict[str, str], 
                               confidence_scores: Dict[str, float],
                               problem_areas: List[Dict]) -> Dict[str, Any]:
        """
        Assess the quality of bubble detection results.
        
        Args:
            detected_answers: Dictionary of detected answers
            confidence_scores: Dictionary of confidence scores
            problem_areas: List of problem areas detected
            
        Returns:
            Dictionary containing detection quality metrics
        """
        try:
            assessment = {
                'detection_quality': 'good',
                'detection_score': 0.0,
                'issues': [],
                'recommendations': [],
                'metrics': {}
            }
            
            # Basic metrics
            total_detected = len(detected_answers)
            total_problems = len(problem_areas)
            confidences = list(confidence_scores.values()) if confidence_scores else []
            
            assessment['metrics'] = {
                'total_detected': total_detected,
                'total_problems': total_problems,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'min_confidence': np.min(confidences) if confidences else 0.0,
                'max_confidence': np.max(confidences) if confidences else 0.0,
                'std_confidence': np.std(confidences) if confidences else 0.0
            }
            
            # Detection count assessment
            detection_count_score = self._assess_detection_count(total_detected, assessment)
            
            # Confidence assessment
            confidence_score = self._assess_confidence_scores(confidences, assessment)
            
            # Problem areas assessment
            problem_score = self._assess_problem_areas(problem_areas, total_detected, assessment)
            
            # Answer pattern assessment
            pattern_score = self._assess_answer_patterns(detected_answers, assessment)
            
            # Calculate overall detection score
            scores = [detection_count_score, confidence_score, problem_score, pattern_score]
            assessment['detection_score'] = np.mean(scores)
            
            # Determine detection quality level
            if assessment['detection_score'] >= 80:
                assessment['detection_quality'] = 'excellent'
            elif assessment['detection_score'] >= 65:
                assessment['detection_quality'] = 'good'
            elif assessment['detection_score'] >= 50:
                assessment['detection_quality'] = 'fair'
            else:
                assessment['detection_quality'] = 'poor'
            
            # Generate detection recommendations
            self._generate_detection_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Detection assessment failed: {e}")
            return {
                'detection_quality': 'unknown',
                'detection_score': 0.0,
                'issues': [f'Assessment failed: {str(e)}'],
                'recommendations': ['Manual review required'],
                'metrics': {}
            }
    
    def _assess_resolution(self, width: int, height: int, assessment: Dict) -> float:
        """Assess image resolution quality."""
        min_width = self.thresholds['min_resolution']['width']
        min_height = self.thresholds['min_resolution']['height']
        
        if width < min_width or height < min_height:
            assessment['issues'].append(f'Low resolution: {width}x{height}')
            return 30.0
        elif width < min_width * 2 or height < min_height * 2:
            assessment['issues'].append(f'Moderate resolution: {width}x{height}')
            return 60.0
        else:
            return 100.0
    
    def _assess_contrast(self, gray: np.ndarray, assessment: Dict) -> float:
        """Assess image contrast."""
        contrast = np.std(gray) / 255.0
        assessment['metrics']['contrast'] = contrast
        
        if contrast < self.thresholds['min_contrast_ratio']:
            assessment['issues'].append(f'Low contrast: {contrast:.3f}')
            return 40.0
        elif contrast < self.thresholds['min_contrast_ratio'] * 1.5:
            return 70.0
        else:
            return 100.0
    
    def _assess_blur(self, gray: np.ndarray, assessment: Dict) -> float:
        """Assess image blur using Laplacian variance."""
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        assessment['metrics']['blur_score'] = blur_score
        
        if blur_score < self.thresholds['blur_threshold']:
            assessment['issues'].append(f'Blurry image: score {blur_score:.2f}')
            return 30.0
        elif blur_score < self.thresholds['blur_threshold'] * 2:
            return 70.0
        else:
            return 100.0
    
    def _assess_rotation(self, gray: np.ndarray, assessment: Dict) -> float:
        """Assess image rotation."""
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:min(10, len(lines)), 0]:
                    angle = theta * 180 / np.pi - 90
                    if abs(angle) < 45:
                        angles.append(angle)
                
                if angles:
                    estimated_rotation = np.median(angles)
                    assessment['metrics']['estimated_rotation'] = estimated_rotation
                    
                    if abs(estimated_rotation) > self.thresholds['max_rotation_angle']:
                        assessment['issues'].append(f'Significant rotation: {estimated_rotation:.1f}°')
                        return 50.0
                    elif abs(estimated_rotation) > self.thresholds['max_rotation_angle'] / 2:
                        return 80.0
            
            return 100.0
            
        except Exception:
            return 80.0  # Neutral score if assessment fails
    
    def _assess_noise(self, gray: np.ndarray, assessment: Dict) -> float:
        """Assess image noise levels."""
        try:
            # Use median filter to estimate noise
            filtered = cv2.medianBlur(gray, 5)
            noise = np.std(gray.astype(float) - filtered.astype(float)) / 255.0
            assessment['metrics']['noise_level'] = noise
            
            if noise > self.thresholds['noise_tolerance']:
                assessment['issues'].append(f'High noise level: {noise:.3f}')
                return 50.0
            elif noise > self.thresholds['noise_tolerance'] / 2:
                return 80.0
            else:
                return 100.0
                
        except Exception:
            return 80.0
    
    def _assess_detection_count(self, count: int, assessment: Dict) -> float:
        """Assess number of detected bubbles."""
        if count < self.thresholds['min_bubble_count']:
            assessment['issues'].append(f'Too few bubbles detected: {count}')
            return 30.0
        elif count < self.thresholds['min_bubble_count'] * 2:
            return 70.0
        else:
            return 100.0
    
    def _assess_confidence_scores(self, confidences: List[float], assessment: Dict) -> float:
        """Assess confidence score distribution."""
        if not confidences:
            assessment['issues'].append('No confidence scores available')
            return 0.0
        
        avg_confidence = np.mean(confidences)
        
        if avg_confidence < self.thresholds['min_avg_confidence']:
            assessment['issues'].append(f'Low average confidence: {avg_confidence:.2f}')
            return 40.0
        elif avg_confidence < self.thresholds['min_avg_confidence'] * 1.5:
            return 70.0
        else:
            return 100.0
    
    def _assess_problem_areas(self, problem_areas: List[Dict], total_detected: int, assessment: Dict) -> float:
        """Assess ratio of problem areas."""
        if total_detected == 0:
            return 0.0
        
        problem_ratio = len(problem_areas) / total_detected
        assessment['metrics']['problem_ratio'] = problem_ratio
        
        if problem_ratio > 0.5:
            assessment['issues'].append(f'High problem ratio: {problem_ratio:.1%}')
            return 30.0
        elif problem_ratio > 0.2:
            return 70.0
        else:
            return 100.0
    
    def _assess_answer_patterns(self, detected_answers: Dict[str, str], assessment: Dict) -> float:
        """Assess answer patterns for anomalies."""
        if not detected_answers:
            return 0.0
        
        # Count answer distribution
        answer_counts = {}
        for answer in detected_answers.values():
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Check for extreme bias (all same answer)
        max_count = max(answer_counts.values()) if answer_counts else 0
        total_answers = len(detected_answers)
        
        if max_count / total_answers > 0.8:  # More than 80% same answer
            assessment['issues'].append(f'Suspicious answer pattern: {max_count}/{total_answers} same answer')
            return 60.0
        
        return 100.0
    
    def _generate_quality_recommendations(self, assessment: Dict):
        """Generate recommendations based on quality issues."""
        for issue in assessment['issues']:
            if 'resolution' in issue.lower():
                assessment['recommendations'].append('Use higher resolution scanner or camera')
            elif 'contrast' in issue.lower():
                assessment['recommendations'].append('Improve lighting conditions or adjust camera settings')
            elif 'blurry' in issue.lower():
                assessment['recommendations'].append('Ensure camera is in focus and sheet is flat')
            elif 'rotation' in issue.lower():
                assessment['recommendations'].append('Align sheet properly before scanning')
            elif 'noise' in issue.lower():
                assessment['recommendations'].append('Clean scanner/camera lens and reduce environmental interference')
    
    def _generate_detection_recommendations(self, assessment: Dict):
        """Generate recommendations based on detection issues."""
        for issue in assessment['issues']:
            if 'few bubbles' in issue.lower():
                assessment['recommendations'].append('Check if template matches OMR sheet layout')
            elif 'confidence' in issue.lower():
                assessment['recommendations'].append('Improve image quality or adjust detection parameters')
            elif 'problem ratio' in issue.lower():
                assessment['recommendations'].append('Manual review recommended for problem areas')
            elif 'pattern' in issue.lower():
                assessment['recommendations'].append('Verify answers - unusual pattern detected')
    
    def get_overall_assessment(self, image_assessment: Dict, detection_assessment: Dict) -> Dict[str, Any]:
        """
        Combine image and detection assessments for overall quality.
        """
        combined_score = (image_assessment['quality_score'] + detection_assessment['detection_score']) / 2
        
        all_issues = image_assessment['issues'] + detection_assessment['issues']
        all_recommendations = image_assessment['recommendations'] + detection_assessment['recommendations']
        
        if combined_score >= 80:
            overall_quality = 'excellent'
        elif combined_score >= 65:
            overall_quality = 'good'
        elif combined_score >= 50:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'
        
        return {
            'overall_quality': overall_quality,
            'combined_score': combined_score,
            'image_assessment': image_assessment,
            'detection_assessment': detection_assessment,
            'all_issues': all_issues,
            'all_recommendations': list(set(all_recommendations))  # Remove duplicates
        }