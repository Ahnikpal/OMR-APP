import numpy as np
import cv2
import pandas as pd
from typing import List, Dict, Any
import io
from datetime import datetime
import random

def create_sample_omr_image(width: int = 800, height: int = 1000) -> np.ndarray:
    """
    Create a sample OMR sheet image for demonstration purposes.
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(image, "SAMPLE OMR ANSWER SHEET", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add instructions
    cv2.putText(image, "Mark your answers clearly with a dark pen", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Create bubble grid
    start_y = 120
    start_x = 100
    
    questions_per_section = 20
    sections = ['MATHEMATICS', 'SCIENCE']
    
    current_y = start_y
    
    for section_idx, section_name in enumerate(sections):
        # Section header
        cv2.putText(image, section_name, (start_x, current_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        current_y += 40
        
        # Create questions for this section
        for q in range(1, questions_per_section + 1):
            question_num = q + (section_idx * questions_per_section)
            
            # Question number
            cv2.putText(image, f"{question_num:2d}.", (start_x, current_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw bubbles for options A, B, C, D
            for opt_idx, option in enumerate(['A', 'B', 'C', 'D']):
                bubble_x = start_x + 50 + (opt_idx * 80)
                bubble_y = current_y + 10
                
                # Draw option label
                cv2.putText(image, option, (bubble_x - 5, bubble_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw bubble circle
                cv2.circle(image, (bubble_x + 10, bubble_y + 10), 12, (0, 0, 0), 2)
                
                # Randomly fill some bubbles to simulate answers
                if random.random() < 0.25:  # 25% chance to fill each bubble
                    cv2.circle(image, (bubble_x + 10, bubble_y + 10), 8, (0, 0, 0), -1)
            
            current_y += 35
        
        current_y += 20  # Extra space between sections
    
    # Add some noise and slight rotation to make it more realistic
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Slight rotation
    center = (width // 2, height // 2)
    rotation_angle = random.uniform(-2, 2)  # Small rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                          borderValue=(255, 255, 255))
    
    return image

def export_results_to_excel(results: List[Dict[str, Any]]) -> io.BytesIO:
    """
    Export OMR processing results to Excel format.
    """
    buffer = io.BytesIO()
    
    # Type: ignore for BytesIO compatibility with pandas
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:  # type: ignore
        # Summary sheet
        summary_data = []
        for result in results:
            scores = result['scores']
            summary_data.append({
                'Filename': result['filename'],
                'Total Score (%)': scores['total_score'],
                'Grade': scores['grade'],
                'Correct Answers': scores['total_correct'],
                'Total Questions': scores['total_questions'],
                'Attempted': scores['total_attempted'],
                'Accuracy (%)': scores['accuracy'],
                'Performance Level': scores['performance_level'],
                'Timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed results for each file
        for idx, result in enumerate(results):
            sheet_name = f"Details_{idx+1}"[:31]  # Excel sheet name limit
            
            # Question-by-question analysis
            question_data = []
            question_analysis = result['scores']['question_analysis']
            
            for q_num, analysis in question_analysis.items():
                question_data.append({
                    'Question': q_num,
                    'Correct Answer': analysis['correct_answer'],
                    'Detected Answer': analysis['detected_answer'],
                    'Status': analysis['status'],
                    'Result': '✓' if analysis['is_correct'] else '✗',
                    'Confidence': result['confidence_scores'].get(q_num, 0)
                })
            
            question_df = pd.DataFrame(question_data)
            question_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Section-wise performance sheet
        section_data = []
        for result in results:
            filename = result['filename']
            section_scores = result['scores']['section_scores']
            
            for section_id, section_info in section_scores.items():
                section_data.append({
                    'Filename': filename,
                    'Section': section_info['name'],
                    'Score (%)': section_info['percentage'],
                    'Grade': section_info['grade'],
                    'Correct': section_info['correct'],
                    'Total': section_info['total'],
                    'Attempted': section_info['attempted'],
                    'Accuracy (%)': section_info['accuracy']
                })
        
        if section_data:
            section_df = pd.DataFrame(section_data)
            section_df.to_excel(writer, sheet_name='Section_Performance', index=False)
    
    buffer.seek(0)
    return buffer

def validate_answer_key(answer_key: Dict) -> List[str]:
    """
    Validate the structure and content of an answer key.
    """
    errors = []
    
    if not isinstance(answer_key, dict):
        errors.append("Answer key must be a dictionary")
        return errors
    
    if not answer_key:
        errors.append("Answer key cannot be empty")
        return errors
    
    valid_options = {'A', 'B', 'C', 'D'}
    
    for set_name, questions in answer_key.items():
        if not isinstance(questions, dict):
            errors.append(f"Questions in set '{set_name}' must be a dictionary")
            continue
        
        for q_num, answer in questions.items():
            if not str(q_num).isdigit():
                errors.append(f"Question number '{q_num}' in set '{set_name}' must be numeric")
            
            if answer not in valid_options:
                errors.append(f"Invalid answer '{answer}' for question {q_num} in set '{set_name}'. Must be A, B, C, or D")
    
    return errors

def preprocess_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Preprocess an uploaded image file for OMR processing.
    """
    try:
        # Read the uploaded file
        file_bytes = uploaded_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image file")
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
        
    except Exception as e:
        raise ValueError(f"Error processing uploaded image: {str(e)}")

def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about an image.
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'total_pixels': width * height,
        'aspect_ratio': width / height if height > 0 else 0,
        'dtype': str(image.dtype)
    }

def create_confidence_visualization(confidence_scores: Dict[str, float], 
                                  threshold: float = 0.6) -> Dict[str, List]:
    """
    Create data for visualizing confidence scores.
    """
    high_confidence = []
    low_confidence = []
    
    for question, confidence in confidence_scores.items():
        if confidence >= threshold:
            high_confidence.append((question, confidence))
        else:
            low_confidence.append((question, confidence))
    
    return {
        'high_confidence': high_confidence,
        'low_confidence': low_confidence,
        'threshold': [threshold]
    }

def calculate_processing_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall processing statistics across all results.
    """
    if not results:
        return {}
    
    total_sheets = len(results)
    total_questions = sum(r['scores']['total_questions'] for r in results)
    total_correct = sum(r['scores']['total_correct'] for r in results)
    total_attempted = sum(r['scores']['total_attempted'] for r in results)
    
    # Calculate averages
    avg_score = np.mean([r['scores']['total_score'] for r in results])
    avg_confidence = np.mean([
        np.mean(list(r['confidence_scores'].values())) 
        for r in results if r['confidence_scores']
    ])
    
    # Count problem areas
    total_problems = sum(len(r['problem_areas']) for r in results)
    
    return {
        'total_sheets_processed': total_sheets,
        'total_questions': total_questions,
        'total_correct_answers': total_correct,
        'total_attempted': total_attempted,
        'average_score': avg_score,
        'average_confidence': avg_confidence,
        'total_problem_areas': total_problems,
        'overall_accuracy': (total_correct / total_attempted * 100) if total_attempted > 0 else 0
    }
