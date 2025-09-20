from typing import Dict, List, Any, Optional
import logging

class ScoreCalculator:
    """
    Calculates scores by comparing detected answers with answer key.
    Supports section-wise scoring and multiple answer sets.
    """
    
    def __init__(self, answer_key: Dict[str, Dict[str, str]]):
        """
        Initialize score calculator with answer key.
        
        Args:
            answer_key: Dictionary with format:
                {'set_a': {'1': 'A', '2': 'B', ...}, 'set_b': {...}}
        """
        self.answer_key = answer_key
        self.logger = logging.getLogger(__name__)
        
        # Default section configuration - can be customized
        self.section_config = {
            'math': {'start': 1, 'end': 20, 'name': 'Mathematics'},
            'science': {'start': 21, 'end': 40, 'name': 'Science'}
        }
    
    def calculate_scores(self, detected_answers: Dict[str, str], 
                        answer_set: str = 'set_a') -> Dict[str, Any]:
        """
        Calculate comprehensive scores for the detected answers.
        
        Args:
            detected_answers: Dictionary mapping question numbers to detected answers
            answer_set: Which answer set to use ('set_a' or 'set_b')
            
        Returns:
            Dictionary containing various score metrics
        """
        try:
            if answer_set not in self.answer_key:
                self.logger.warning(f"Answer set '{answer_set}' not found, using first available")
                answer_set = list(self.answer_key.keys())[0]
            
            correct_answers = self.answer_key[answer_set]
            
            # Initialize score tracking
            total_questions = len(correct_answers)
            total_correct = 0
            total_attempted = 0
            section_scores = {}
            question_analysis = {}
            
            # Process each question
            for q_num_str, correct_answer in correct_answers.items():
                detected_answer = detected_answers.get(q_num_str, 'N')  # N for not answered
                
                is_correct = detected_answer == correct_answer
                is_attempted = detected_answer not in ['N', 'X', 'M']  # Not answered, no answer, or multiple
                
                if is_correct:
                    total_correct += 1
                if is_attempted:
                    total_attempted += 1
                
                question_analysis[q_num_str] = {
                    'correct_answer': correct_answer,
                    'detected_answer': detected_answer,
                    'is_correct': is_correct,
                    'is_attempted': is_attempted,
                    'status': self._get_question_status(detected_answer, correct_answer)
                }
            
            # Calculate section-wise scores
            for section_id, section_info in self.section_config.items():
                section_score = self._calculate_section_score(
                    question_analysis, section_info['start'], section_info['end']
                )
                section_score['name'] = section_info['name']
                section_scores[section_id] = section_score
            
            # Calculate overall metrics
            total_score_percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0
            attempt_percentage = (total_attempted / total_questions * 100) if total_questions > 0 else 0
            accuracy = (total_correct / total_attempted * 100) if total_attempted > 0 else 0
            
            # Compile final results
            results = {
                'total_score': total_score_percentage,
                'total_correct': total_correct,
                'total_questions': total_questions,
                'total_attempted': total_attempted,
                'attempt_percentage': attempt_percentage,
                'accuracy': accuracy,
                'section_scores': section_scores,
                'question_analysis': question_analysis,
                'answer_set_used': answer_set,
                'grade': self._calculate_grade(total_score_percentage),
                'performance_level': self._get_performance_level(total_score_percentage)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            return self._get_empty_results()
    
    def _calculate_section_score(self, question_analysis: Dict[str, Dict], 
                                start: int, end: int) -> Dict[str, Any]:
        """
        Calculate scores for a specific section.
        """
        section_correct = 0
        section_attempted = 0
        section_total = 0
        
        for q_num in range(start, end + 1):
            q_num_str = str(q_num)
            if q_num_str in question_analysis:
                analysis = question_analysis[q_num_str]
                section_total += 1
                
                if analysis['is_correct']:
                    section_correct += 1
                if analysis['is_attempted']:
                    section_attempted += 1
        
        percentage = (section_correct / section_total * 100) if section_total > 0 else 0
        attempt_rate = (section_attempted / section_total * 100) if section_total > 0 else 0
        accuracy = (section_correct / section_attempted * 100) if section_attempted > 0 else 0
        
        return {
            'correct': section_correct,
            'total': section_total,
            'attempted': section_attempted,
            'percentage': percentage,
            'attempt_rate': attempt_rate,
            'accuracy': accuracy,
            'grade': self._calculate_grade(percentage)
        }
    
    def _get_question_status(self, detected: str, correct: str) -> str:
        """
        Determine the status of a question response.
        """
        if detected == 'N':
            return 'not_answered'
        elif detected == 'M':
            return 'multiple_marked'
        elif detected == 'X':
            return 'unclear'
        elif detected == correct:
            return 'correct'
        else:
            return 'incorrect'
    
    def _calculate_grade(self, percentage: float) -> str:
        """
        Calculate letter grade based on percentage.
        """
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'
    
    def _get_performance_level(self, percentage: float) -> str:
        """
        Get performance level description.
        """
        if percentage >= 90:
            return 'Excellent'
        elif percentage >= 80:
            return 'Good'
        elif percentage >= 70:
            return 'Satisfactory'
        elif percentage >= 60:
            return 'Needs Improvement'
        else:
            return 'Unsatisfactory'
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """
        Return empty results structure for error cases.
        """
        return {
            'total_score': 0.0,
            'total_correct': 0,
            'total_questions': 0,
            'total_attempted': 0,
            'attempt_percentage': 0.0,
            'accuracy': 0.0,
            'section_scores': {},
            'question_analysis': {},
            'answer_set_used': 'none',
            'grade': 'F',
            'performance_level': 'Error'
        }
    
    def compare_answer_sets(self, detected_answers: Dict[str, str]) -> Dict[str, Dict]:
        """
        Compare detected answers against all available answer sets.
        Useful for determining which answer set was used.
        """
        results = {}
        
        for set_name in self.answer_key.keys():
            score_result = self.calculate_scores(detected_answers, set_name)
            results[set_name] = {
                'score': score_result['total_score'],
                'correct': score_result['total_correct'],
                'total': score_result['total_questions']
            }
        
        return results
    
    def get_detailed_analysis(self, detected_answers: Dict[str, str], 
                            answer_set: str = 'set_a') -> Dict[str, Any]:
        """
        Get detailed question-by-question analysis.
        """
        scores = self.calculate_scores(detected_answers, answer_set)
        
        # Add additional analysis
        problem_questions = []
        strong_areas = []
        weak_areas = []
        
        question_analysis = scores['question_analysis']
        
        for q_num, analysis in question_analysis.items():
            if analysis['status'] in ['multiple_marked', 'unclear']:
                problem_questions.append(q_num)
        
        # Analyze section performance
        for section_id, section_score in scores['section_scores'].items():
            if section_score['percentage'] >= 80:
                strong_areas.append(section_score['name'])
            elif section_score['percentage'] < 60:
                weak_areas.append(section_score['name'])
        
        return {
            'scores': scores,
            'problem_questions': problem_questions,
            'strong_areas': strong_areas,
            'weak_areas': weak_areas,
            'recommendations': self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores: Dict[str, Any]) -> List[str]:
        """
        Generate personalized recommendations based on performance.
        """
        recommendations = []
        
        total_score = scores['total_score']
        attempt_rate = scores['attempt_percentage']
        
        if attempt_rate < 80:
            recommendations.append("Try to attempt more questions to improve your overall score.")
        
        if total_score < 60:
            recommendations.append("Focus on fundamental concepts and practice more questions.")
        elif total_score < 80:
            recommendations.append("Good progress! Work on weak areas to achieve excellence.")
        
        # Section-specific recommendations
        for section_id, section_score in scores['section_scores'].items():
            if section_score['percentage'] < 50:
                recommendations.append(f"Need significant improvement in {section_score['name']}.")
            elif section_score['percentage'] < 70:
                recommendations.append(f"Practice more problems in {section_score['name']}.")
        
        return recommendations
