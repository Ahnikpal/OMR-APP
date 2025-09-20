import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import io
import base64
from datetime import datetime
import os

from omr_processor import OMRProcessor
from bubble_detector import BubbleDetector
from ml_bubble_detector import MLBubbleDetector
from score_calculator import ScoreCalculator
from utils import create_sample_omr_image, export_results_to_excel
from template_manager import TemplateManager
from quality_assessor import QualityAssessor

def main():
    st.title("📊 Automated OMR Evaluation & Scoring System")
    st.markdown("Upload OMR answer sheets to automatically detect filled bubbles and calculate scores")
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Answer key selection
        answer_key_source = st.radio(
            "Answer Key Source:",
            ["Upload JSON/CSV", "Use Sample Answer Key"]
        )
        
        answer_key = None
        if answer_key_source == "Upload JSON/CSV":
            uploaded_key = st.file_uploader(
                "Upload Answer Key", 
                type=['json', 'csv'],
                help="JSON format: {'set_a': {'1': 'A', '2': 'B', ...}, 'set_b': {...}}"
            )
            if uploaded_key:
                try:
                    if uploaded_key.name.endswith('.json'):
                        answer_key = json.load(uploaded_key)
                    else:
                        df = pd.read_csv(uploaded_key)
                        # Convert CSV to expected format
                        answer_key = {}
                        for _, row in df.iterrows():
                            set_name = str(row.get('set', 'set_a')).lower()
                            if set_name not in answer_key:
                                answer_key[set_name] = {}
                            answer_key[set_name][str(row['question'])] = row['answer']
                    st.success("Answer key loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading answer key: {str(e)}")
        else:
            # Load processed answer key (try both files)
            answer_key_files = ['sample_data/processed_answer_key.json', 'sample_data/answer_key.json']
            answer_key = None
            for file_path in answer_key_files:
                try:
                    with open(file_path, 'r') as f:
                        answer_key = json.load(f)
                    st.success(f"Answer key loaded from {file_path}")
                    break
                except FileNotFoundError:
                    continue
            
            if answer_key is None:
                st.error("No answer key found. Please upload your own.")
        
        # Processing parameters
        st.subheader("Processing Parameters")
        
        # Template selection
        template_manager = TemplateManager()
        template_names = template_manager.list_templates()
        template_info = template_manager.get_template_info()
        
        selected_template = st.selectbox(
            "OMR Sheet Template",
            template_names,
            index=0,
            help="Choose the template that matches your OMR sheet layout"
        )
        
        if selected_template:
            template = template_manager.get_template(selected_template)
            st.info(f"📋 {template.description}")
            
            # Show template details
            with st.expander("Template Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Page Size:** {template.page_dimensions['width']} x {template.page_dimensions['height']}")
                    st.write(f"**Sections:** {len(template.sections)}")
                with col2:
                    total_questions = sum(s.end_question - s.start_question + 1 for s in template.sections)
                    st.write(f"**Total Questions:** {total_questions}")
                    st.write(f"**Options:** {', '.join(template.bubble_config.options)}")
        
        confidence_threshold = st.slider("Bubble Detection Confidence", 0.1, 0.9, 0.6)
        perspective_correction = st.checkbox("Enable Perspective Correction", True)
        rotation_correction = st.checkbox("Enable Rotation Correction", True)
        
        # ML Enhancement option
        ml_enhanced = st.checkbox(
            "🤖 Enable ML-Enhanced Detection", 
            value=True,
            help="Use machine learning techniques for improved bubble detection accuracy"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload OMR Sheets")
        
        # Option to use sample image or upload
        image_source = st.radio(
            "Image Source:",
            ["Upload Images", "Generate Sample OMR"]
        )
        
        uploaded_files = None
        sample_image = None
        
        if image_source == "Upload Images":
            uploaded_files = st.file_uploader(
                "Choose OMR sheet images",
                type=['png', 'jpg', 'jpeg', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple OMR sheets for batch processing. Supported formats: PNG, JPG, JPEG, TIFF"
            )
            
            # Show batch processing info if multiple files uploaded
            if uploaded_files and len(uploaded_files) > 1:
                st.info(f"📁 {len(uploaded_files)} files selected for batch processing")
                
                # Batch processing options
                col1, col2 = st.columns(2)
                with col1:
                    batch_confidence = st.slider(
                        "Batch Confidence Threshold", 
                        0.1, 0.9, confidence_threshold,
                        help="Apply same confidence threshold to all images"
                    )
                with col2:
                    stop_on_error = st.checkbox(
                        "Stop on Error", 
                        value=False,
                        help="Stop batch processing if any image fails"
                    )
        else:
            if st.button("Generate Sample OMR Sheet"):
                sample_image = create_sample_omr_image()
                st.success("Sample OMR sheet generated!")
        
        # Process images
        if (uploaded_files or sample_image) and answer_key:
            process_button = st.button("🔍 Process OMR Sheets", type="primary")
            
            if process_button:
                # Initialize processors
                omr_processor = OMRProcessor()
                
                # Use ML-enhanced detector if enabled
                if ml_enhanced:
                    bubble_detector = MLBubbleDetector(confidence_threshold=confidence_threshold, use_ml_enhancement=True)
                else:
                    bubble_detector = BubbleDetector(confidence_threshold=confidence_threshold)
                
                score_calculator = ScoreCalculator(answer_key)
                quality_assessor = QualityAssessor()
                
                # Process each image
                images_to_process = []
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        images_to_process.append((np.array(image), uploaded_file.name))
                elif sample_image is not None:
                    images_to_process.append((sample_image, "sample_omr.png"))
                
                # Enhanced batch processing with detailed progress tracking
                total_files = len(images_to_process)
                
                # Create progress tracking UI
                progress_container = st.container()
                with progress_container:
                    overall_progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Batch statistics
                    if total_files > 1:
                        batch_cols = st.columns(4)
                        with batch_cols[0]:
                            processed_metric = st.empty()
                        with batch_cols[1]:
                            success_metric = st.empty()
                        with batch_cols[2]:
                            avg_score_metric = st.empty()
                        with batch_cols[3]:
                            avg_confidence_metric = st.empty()
                
                results = []
                successful_count = 0
                
                for idx, (image, filename) in enumerate(images_to_process):
                    # Update status
                    status_text.text(f"Processing {idx + 1}/{total_files}: {filename}")
                    
                    try:
                        # Use batch confidence if available
                        current_confidence = batch_confidence if 'batch_confidence' in locals() else confidence_threshold
                        bubble_detector.confidence_threshold = current_confidence
                        
                        # Preprocess image
                        processed_image = omr_processor.preprocess_image(
                            image,
                            enable_perspective_correction=perspective_correction,
                            enable_rotation_correction=rotation_correction
                        )
                        
                        # Detect bubbles
                        detected_answers, confidence_scores, problem_areas = bubble_detector.detect_answers(processed_image)
                        
                        # Calculate scores
                        scores = score_calculator.calculate_scores(detected_answers)
                        
                        # Quality assessment
                        image_quality = quality_assessor.assess_image_quality(image)
                        detection_quality = quality_assessor.assess_detection_quality(
                            detected_answers, confidence_scores, problem_areas
                        )
                        overall_quality = quality_assessor.get_overall_assessment(image_quality, detection_quality)
                        
                        result = {
                            'filename': filename,
                            'original_image': image,
                            'processed_image': processed_image,
                            'detected_answers': detected_answers,
                            'confidence_scores': confidence_scores,
                            'problem_areas': problem_areas,
                            'scores': scores,
                            'timestamp': datetime.now(),
                            'processing_index': idx + 1,
                            'quality_assessment': overall_quality
                        }
                        
                        results.append(result)
                        successful_count += 1
                        
                        # Update batch metrics for multiple files
                        if total_files > 1:
                            processed_metric.metric("Processed", f"{idx + 1}/{total_files}")
                            success_metric.metric("Successful", successful_count)
                            
                            if results:
                                avg_score = np.mean([r['scores']['total_score'] for r in results])
                                avg_conf = np.mean([np.mean(list(r['confidence_scores'].values())) 
                                                  for r in results if r['confidence_scores']])
                                avg_score_metric.metric("Avg Score", f"{avg_score:.1f}%")
                                avg_confidence_metric.metric("Avg Confidence", f"{avg_conf:.2f}")
                        
                    except Exception as e:
                        error_msg = f"Error processing {filename}: {str(e)}"
                        st.error(error_msg)
                        
                        if 'stop_on_error' in locals() and stop_on_error:
                            st.warning("Batch processing stopped due to error")
                            break
                        
                        continue
                    
                    # Update progress
                    overall_progress.progress((idx + 1) / total_files)
                
                # Final status update
                if total_files > 1:
                    status_text.text(f"Batch processing complete: {successful_count}/{total_files} files processed successfully")
                else:
                    status_text.text("Processing complete")
                
                # Store results in session state
                st.session_state.processed_results.extend(results)
                st.success(f"Processed {len(results)} OMR sheets successfully!")
    
    with col2:
        st.subheader("Answer Key Preview")
        if answer_key:
            # Show answer key summary
            total_sets = len(answer_key)
            st.metric("Number of Sets", total_sets)
            
            for set_name, answers in answer_key.items():
                with st.expander(f"Set {set_name.upper()}"):
                    questions_per_row = 5
                    questions = list(answers.keys())
                    
                    for i in range(0, len(questions), questions_per_row):
                        row_questions = questions[i:i+questions_per_row]
                        cols = st.columns(len(row_questions))
                        
                        for j, q in enumerate(row_questions):
                            with cols[j]:
                                st.write(f"Q{q}: **{answers[q]}**")
        else:
            st.info("Upload or select an answer key to see preview")
    
    # Display results
    if st.session_state.processed_results:
        # Analytics Dashboard
        st.header("📊 Analytics Dashboard")
        
        # Calculate overall statistics
        total_results = len(st.session_state.processed_results)
        all_scores = [r['scores']['total_score'] for r in st.session_state.processed_results]
        all_confidences = [np.mean(list(r['confidence_scores'].values())) 
                          for r in st.session_state.processed_results if r['confidence_scores']]
        total_problems = sum(len(r['problem_areas']) for r in st.session_state.processed_results)
        
        # Main metrics
        metrics_cols = st.columns(5)
        with metrics_cols[0]:
            st.metric("Total Sheets", total_results)
        with metrics_cols[1]:
            avg_score = np.mean(all_scores) if all_scores else 0
            st.metric("Average Score", f"{avg_score:.1f}%")
        with metrics_cols[2]:
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        with metrics_cols[3]:
            st.metric("Problem Areas", total_problems)
        with metrics_cols[4]:
            success_rate = (total_results - sum(1 for r in st.session_state.processed_results 
                           if r['scores']['total_score'] == 0)) / total_results * 100 if total_results > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Performance distribution charts
        if total_results > 1:
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                st.subheader("Score Distribution")
                score_data = pd.DataFrame({
                    'Filename': [r['filename'] for r in st.session_state.processed_results],
                    'Score': all_scores
                })
                st.bar_chart(score_data.set_index('Filename')['Score'])
            
            with chart_cols[1]:
                st.subheader("Confidence Distribution")
                if all_confidences:
                    conf_data = pd.DataFrame({
                        'Filename': [r['filename'] for r in st.session_state.processed_results if r['confidence_scores']],
                        'Confidence': all_confidences
                    })
                    st.bar_chart(conf_data.set_index('Filename')['Confidence'])
        
        # Quality insights
        st.subheader("📈 Processing Insights")
        insights_cols = st.columns(3)
        
        with insights_cols[0]:
            # High performers
            high_score_results = [r for r in st.session_state.processed_results if r['scores']['total_score'] >= 80]
            st.metric("High Performers (≥80%)", len(high_score_results))
            if high_score_results:
                st.write("**Top performers:**")
                for r in high_score_results[:3]:
                    st.write(f"• {r['filename']}: {r['scores']['total_score']:.1f}%")
        
        with insights_cols[1]:
            # Low confidence detections
            low_conf_results = [r for r in st.session_state.processed_results 
                               if r['confidence_scores'] and np.mean(list(r['confidence_scores'].values())) < 0.5]
            st.metric("Low Confidence (< 0.5)", len(low_conf_results))
            if low_conf_results:
                st.write("**Needs review:**")
                for r in low_conf_results[:3]:
                    avg_conf = np.mean(list(r['confidence_scores'].values()))
                    st.write(f"• {r['filename']}: {avg_conf:.2f}")
        
        with insights_cols[2]:
            # Problem areas summary
            problem_types = {}
            for result in st.session_state.processed_results:
                for problem in result['problem_areas']:
                    reason = problem.get('reason', 'Unknown')
                    problem_types[reason] = problem_types.get(reason, 0) + 1
            
            st.metric("Problem Types", len(problem_types))
            if problem_types:
                st.write("**Common issues:**")
                for reason, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                    st.write(f"• {reason}: {count}")
        
        st.header("📋 Detailed Results")
        
        for idx, result in enumerate(st.session_state.processed_results):
            with st.expander(f"📄 {result['filename']} - Score: {result['scores']['total_score']:.1f}%", expanded=True):
                
                # Create columns for layout
                img_col, details_col = st.columns([1, 2])
                
                with img_col:
                    st.subheader("Images")
                    
                    # Original image
                    st.write("**Original:**")
                    st.image(result['original_image'], width=200)
                    
                    # Processed image
                    st.write("**Processed:**")
                    st.image(result['processed_image'], width=200)
                
                with details_col:
                    st.subheader("Detection Results")
                    
                    # Score summary
                    scores = result['scores']
                    score_cols = st.columns(3)
                    
                    with score_cols[0]:
                        st.metric("Total Score", f"{scores['total_score']:.1f}%")
                    with score_cols[1]:
                        st.metric("Correct Answers", f"{scores['total_correct']}/{scores['total_questions']}")
                    with score_cols[2]:
                        avg_confidence = np.mean(list(result['confidence_scores'].values())) if result['confidence_scores'] else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                    
                    # Section-wise scores
                    if 'section_scores' in scores:
                        st.write("**Section-wise Performance:**")
                        section_data = []
                        for section, section_score in scores['section_scores'].items():
                            section_data.append({
                                'Section': section.title(),
                                'Score (%)': f"{section_score['percentage']:.1f}",
                                'Correct': f"{section_score['correct']}/{section_score['total']}"
                            })
                        
                        st.dataframe(pd.DataFrame(section_data), hide_index=True)
                    
                    # Answer comparison
                    st.write("**Answer Comparison:**")
                    
                    # Create answer comparison dataframe
                    comparison_data = []
                    detected = result['detected_answers']
                    correct_answers = answer_key.get('set_a', {}) if answer_key else {}  # Default to set_a
                    
                    for q_num in sorted(detected.keys(), key=int):
                        correct_ans = correct_answers.get(q_num, 'N/A')
                        detected_ans = detected[q_num]
                        is_correct = detected_ans == correct_ans
                        confidence = result['confidence_scores'].get(q_num, 0)
                        
                        comparison_data.append({
                            'Question': q_num,
                            'Detected': detected_ans,
                            'Correct': correct_ans,
                            'Status': '✅' if is_correct else '❌',
                            'Confidence': f"{confidence:.2f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, hide_index=True)
                    
                    # Problem areas
                    if result['problem_areas']:
                        st.warning("**Areas flagged for manual review:**")
                        for area in result['problem_areas']:
                            st.write(f"- Question {area['question']}: {area['reason']}")
        
        # Export functionality
        st.header("📤 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to Excel", type="secondary"):
                try:
                    excel_buffer = export_results_to_excel(st.session_state.processed_results)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")
        
        with col2:
            if st.button("📋 Export to CSV"):
                try:
                    # Create summary CSV
                    csv_data = []
                    for result in st.session_state.processed_results:
                        row = {
                            'Filename': result['filename'],
                            'Total Score (%)': result['scores']['total_score'],
                            'Correct Answers': result['scores']['total_correct'],
                            'Total Questions': result['scores']['total_questions'],
                            'Timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Add detected answers
                        for q, ans in result['detected_answers'].items():
                            row[f'Q{q}_Detected'] = ans
                        
                        csv_data.append(row)
                    
                    csv_df = pd.DataFrame(csv_data)
                    csv_buffer = io.StringIO()
                    csv_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_buffer.getvalue(),
                        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating CSV file: {str(e)}")
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            st.session_state.processed_results = []
            st.rerun()
    
    # Instructions
    with st.expander("📖 How to Use This System"):
        st.markdown("""
        ### Step-by-Step Instructions:
        
        1. **Prepare Answer Key:**
           - Upload a JSON file with format: `{"set_a": {"1": "A", "2": "B", ...}, "set_b": {...}}`
           - Or upload a CSV with columns: `question, answer, set`
           - Or use the provided sample answer key
        
        2. **Upload OMR Sheets:**
           - Supported formats: PNG, JPG, JPEG, TIFF
           - Images should be clear and well-lit
           - Multiple sheets can be processed at once
        
        3. **Configure Processing:**
           - Adjust confidence threshold for bubble detection
           - Enable/disable perspective and rotation correction
        
        4. **Review Results:**
           - Check individual answer comparisons
           - Review flagged areas that may need manual verification
           - View section-wise performance breakdown
        
        5. **Export Results:**
           - Download detailed Excel reports
           - Export summary data as CSV
        
        ### Tips for Best Results:
        - Ensure OMR sheets are well-lit and in focus
        - Avoid shadows and reflections
        - Keep the sheet flat against the scanning surface
        - Use high-resolution images (at least 300 DPI)
        """)

if __name__ == "__main__":
    main()
