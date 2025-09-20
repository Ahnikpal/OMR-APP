"""
Machine Learning enhanced bubble detection for improved accuracy.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from bubble_detector import BubbleDetector

class MLBubbleDetector(BubbleDetector):
    """
    Enhanced bubble detector using machine learning techniques for improved accuracy.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, use_ml_enhancement: bool = True):
        super().__init__(confidence_threshold)
        self.use_ml_enhancement = use_ml_enhancement
        self.logger = logging.getLogger(__name__)
        
        # ML enhancement parameters
        self.ml_params = {
            'clustering_enabled': True,
            'outlier_detection_enabled': True,
            'adaptive_thresholding': True,
            'pattern_analysis': True
        }
    
    def detect_answers(self, binary_image: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float], List[Dict]]:
        """
        Enhanced bubble detection with ML techniques.
        """
        if not self.use_ml_enhancement:
            return super().detect_answers(binary_image)
        
        try:
            # First, get basic detection results
            detected_answers, confidence_scores, problem_areas = super().detect_answers(binary_image)
            
            if not detected_answers:
                return detected_answers, confidence_scores, problem_areas
            
            # Apply ML enhancements
            enhanced_results = self._apply_ml_enhancements(
                binary_image, detected_answers, confidence_scores, problem_areas
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"ML-enhanced detection failed: {e}, falling back to basic detection")
            return super().detect_answers(binary_image)
    
    def _apply_ml_enhancements(self, binary_image: np.ndarray, 
                              detected_answers: Dict[str, str],
                              confidence_scores: Dict[str, float],
                              problem_areas: List[Dict]) -> Tuple[Dict[str, str], Dict[str, float], List[Dict]]:
        """Apply machine learning enhancements to improve detection accuracy."""
        
        enhanced_answers = detected_answers.copy()
        enhanced_confidence = confidence_scores.copy()
        enhanced_problems = problem_areas.copy()
        
        # 1. Clustering-based bubble refinement
        if self.ml_params['clustering_enabled']:
            enhanced_answers, enhanced_confidence = self._apply_clustering_refinement(
                binary_image, enhanced_answers, enhanced_confidence
            )
        
        # 2. Outlier detection for anomalous answers
        if self.ml_params['outlier_detection_enabled']:
            enhanced_answers, enhanced_confidence, outlier_problems = self._detect_outlier_answers(
                enhanced_answers, enhanced_confidence
            )
            enhanced_problems.extend(outlier_problems)
        
        # 3. Adaptive confidence adjustment
        if self.ml_params['adaptive_thresholding']:
            enhanced_confidence = self._adaptive_confidence_adjustment(
                enhanced_answers, enhanced_confidence
            )
        
        # 4. Pattern-based validation
        if self.ml_params['pattern_analysis']:
            enhanced_answers, enhanced_confidence, pattern_problems = self._analyze_answer_patterns(
                enhanced_answers, enhanced_confidence
            )
            enhanced_problems.extend(pattern_problems)
        
        return enhanced_answers, enhanced_confidence, enhanced_problems
    
    def _apply_clustering_refinement(self, binary_image: np.ndarray,
                                   answers: Dict[str, str],
                                   confidences: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Use clustering to refine bubble detection based on similar features."""
        try:
            # Extract features from detected bubble regions
            features = []
            question_keys = []
            
            for q_key in answers.keys():
                try:
                    # Get bubble regions and extract features
                    bubble_features = self._extract_bubble_features(binary_image, q_key)
                    if bubble_features is not None:
                        features.append(bubble_features)
                        question_keys.append(q_key)
                except Exception:
                    continue
            
            if len(features) < 3:  # Not enough data for clustering
                return answers, confidences
            
            features_array = np.array(features)
            
            # Apply K-means clustering to group similar bubbles
            n_clusters = min(4, len(features))  # Max 4 clusters (for A, B, C, D)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_array)
            
            # Analyze clusters and adjust confidence based on cluster consistency
            cluster_analysis = self._analyze_clusters(clusters, answers, confidences, question_keys)
            
            # Apply adjustments
            refined_answers = answers.copy()
            refined_confidences = confidences.copy()
            
            for q_key, adjustment in cluster_analysis.items():
                if q_key in refined_confidences:
                    refined_confidences[q_key] = min(1.0, max(0.0, 
                        refined_confidences[q_key] + adjustment['confidence_delta']))
            
            return refined_answers, refined_confidences
            
        except Exception as e:
            self.logger.warning(f"Clustering refinement failed: {e}")
            return answers, confidences
    
    def _extract_bubble_features(self, binary_image: np.ndarray, question_key: str) -> np.ndarray:
        """Extract features from bubble regions for ML analysis."""
        try:
            # This is a simplified feature extraction
            # In a real implementation, you'd extract more sophisticated features
            height, width = binary_image.shape
            
            # Create some basic features based on image characteristics
            features = []
            
            # Global image statistics
            features.extend([
                np.mean(binary_image),
                np.std(binary_image),
                np.sum(binary_image > 0) / (height * width)  # Fill ratio
            ])
            
            # Local region analysis (simplified)
            center_y, center_x = height // 2, width // 2
            local_region = binary_image[
                max(0, center_y - 50):min(height, center_y + 50),
                max(0, center_x - 50):min(width, center_x + 50)
            ]
            
            if local_region.size > 0:
                features.extend([
                    np.mean(local_region),
                    np.std(local_region),
                    np.sum(local_region > 0) / local_region.size
                ])
            else:
                features.extend([0, 0, 0])
            
            return np.array(features)
            
        except Exception:
            return None
    
    def _analyze_clusters(self, clusters: np.ndarray, answers: Dict[str, str], 
                         confidences: Dict[str, float], question_keys: List[str]) -> Dict[str, Dict]:
        """Analyze cluster consistency and generate adjustments."""
        cluster_analysis = {}
        
        # Group questions by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(question_keys[i])
        
        # Analyze each cluster
        for cluster_id, questions in cluster_groups.items():
            if len(questions) < 2:
                continue
            
            # Check answer consistency within cluster
            cluster_answers = [answers[q] for q in questions]
            cluster_confidences = [confidences[q] for q in questions]
            
            # If most answers in cluster are the same, boost confidence
            answer_counts = {}
            for ans in cluster_answers:
                answer_counts[ans] = answer_counts.get(ans, 0) + 1
            
            most_common_answer = max(answer_counts.keys(), key=lambda k: answer_counts[k])
            consistency_ratio = answer_counts[most_common_answer] / len(cluster_answers)
            
            for q in questions:
                adjustment = {
                    'confidence_delta': 0.0,
                    'cluster_consistency': consistency_ratio
                }
                
                if answers[q] == most_common_answer and consistency_ratio > 0.7:
                    # Boost confidence for consistent answers
                    adjustment['confidence_delta'] = 0.1
                elif answers[q] != most_common_answer and consistency_ratio > 0.8:
                    # Reduce confidence for inconsistent answers
                    adjustment['confidence_delta'] = -0.1
                
                cluster_analysis[q] = adjustment
        
        return cluster_analysis
    
    def _detect_outlier_answers(self, answers: Dict[str, str], 
                               confidences: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float], List[Dict]]:
        """Detect outlier answers using anomaly detection."""
        try:
            outlier_problems = []
            
            if len(answers) < 10:  # Need sufficient data for outlier detection
                return answers, confidences, outlier_problems
            
            # Create feature vectors based on answer patterns and confidences
            features = []
            question_keys = list(answers.keys())
            
            for q_key in question_keys:
                answer = answers[q_key]
                conf = confidences.get(q_key, 0.0)
                
                # Convert answer to numeric (A=1, B=2, C=3, D=4, others=0)
                answer_numeric = {'A': 1, 'B': 2, 'C': 3, 'D': 4}.get(answer, 0)
                
                features.append([answer_numeric, conf])
            
            features_array = np.array(features)
            
            # Apply Isolation Forest for outlier detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = isolation_forest.fit_predict(features_array)
            
            # Flag outliers as potential problems
            for i, is_outlier in enumerate(outliers):
                if is_outlier == -1:  # -1 indicates outlier
                    q_key = question_keys[i]
                    outlier_problems.append({
                        'question': q_key,
                        'reason': f'Anomalous answer pattern detected (answer: {answers[q_key]}, conf: {confidences.get(q_key, 0):.2f})'
                    })
            
            return answers, confidences, outlier_problems
            
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
            return answers, confidences, []
    
    def _adaptive_confidence_adjustment(self, answers: Dict[str, str], 
                                       confidences: Dict[str, float]) -> Dict[str, float]:
        """Adaptively adjust confidence scores based on overall detection quality."""
        try:
            if not confidences:
                return confidences
            
            adjusted_confidences = confidences.copy()
            conf_values = list(confidences.values())
            
            # Calculate statistics
            mean_conf = np.mean(conf_values)
            std_conf = np.std(conf_values)
            
            # Adaptive adjustment based on overall quality
            for q_key, conf in confidences.items():
                # Normalize confidence based on overall distribution
                if std_conf > 0:
                    z_score = (conf - mean_conf) / std_conf
                    
                    # Adjust based on z-score
                    if z_score > 1:  # High confidence relative to others
                        adjustment = 0.05
                    elif z_score < -1:  # Low confidence relative to others
                        adjustment = -0.05
                    else:
                        adjustment = 0.0
                    
                    adjusted_confidences[q_key] = min(1.0, max(0.0, conf + adjustment))
            
            return adjusted_confidences
            
        except Exception as e:
            self.logger.warning(f"Adaptive confidence adjustment failed: {e}")
            return confidences
    
    def _analyze_answer_patterns(self, answers: Dict[str, str], 
                                confidences: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float], List[Dict]]:
        """Analyze answer patterns for suspicious sequences."""
        try:
            pattern_problems = []
            
            if len(answers) < 5:
                return answers, confidences, pattern_problems
            
            # Convert to list for sequence analysis
            question_keys = sorted(answers.keys(), key=int)
            answer_sequence = [answers[q] for q in question_keys]
            
            # Detect suspicious patterns
            
            # 1. Long sequences of same answer
            current_answer = None
            sequence_length = 0
            max_sequence = 0
            
            for answer in answer_sequence:
                if answer == current_answer:
                    sequence_length += 1
                else:
                    max_sequence = max(max_sequence, sequence_length)
                    current_answer = answer
                    sequence_length = 1
            
            max_sequence = max(max_sequence, sequence_length)
            
            if max_sequence > 7:  # More than 7 consecutive same answers
                pattern_problems.append({
                    'question': 'sequence',
                    'reason': f'Suspicious pattern: {max_sequence} consecutive same answers'
                })
            
            # 2. Detect ABCD repeating patterns
            pattern_length = 4
            if len(answer_sequence) >= pattern_length * 3:  # Need at least 3 repetitions
                for start in range(len(answer_sequence) - pattern_length * 2):
                    pattern = answer_sequence[start:start + pattern_length]
                    
                    # Check if pattern repeats
                    repeats = 1
                    for i in range(start + pattern_length, len(answer_sequence) - pattern_length + 1, pattern_length):
                        if answer_sequence[i:i + pattern_length] == pattern:
                            repeats += 1
                        else:
                            break
                    
                    if repeats >= 3:  # Pattern repeats 3 or more times
                        pattern_problems.append({
                            'question': f'pattern_{start}',
                            'reason': f'Repeating pattern detected: {pattern} x{repeats}'
                        })
                        break
            
            return answers, confidences, pattern_problems
            
        except Exception as e:
            self.logger.warning(f"Pattern analysis failed: {e}")
            return answers, confidences, []
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get status of ML enhancements."""
        return {
            'ml_enabled': self.use_ml_enhancement,
            'features': self.ml_params,
            'description': 'Enhanced bubble detection using machine learning techniques'
        }