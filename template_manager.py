"""
Template manager for different OMR sheet layouts and configurations.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class BubbleConfig:
    """Configuration for individual bubbles."""
    radius: int = 12
    inner_radius: int = 8
    spacing_horizontal: int = 80
    spacing_vertical: int = 35
    options: List[str] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = ["A", "B", "C", "D"]

@dataclass
class SectionConfig:
    """Configuration for a section of questions."""
    name: str
    start_question: int
    end_question: int
    start_position: Dict[str, int]  # {"x": 100, "y": 160}
    columns: int = 1  # Number of columns for questions
    
@dataclass
class OMRTemplate:
    """Complete OMR sheet template configuration."""
    name: str
    description: str
    page_dimensions: Dict[str, int]  # {"width": 800, "height": 1000}
    sections: List[SectionConfig]
    bubble_config: BubbleConfig
    detection_params: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'page_dimensions': self.page_dimensions,
            'sections': [
                {
                    'name': section.name,
                    'start_question': section.start_question,
                    'end_question': section.end_question,
                    'start_position': section.start_position,
                    'columns': section.columns
                }
                for section in self.sections
            ],
            'bubble_config': {
                'radius': self.bubble_config.radius,
                'inner_radius': self.bubble_config.inner_radius,
                'spacing_horizontal': self.bubble_config.spacing_horizontal,
                'spacing_vertical': self.bubble_config.spacing_vertical,
                'options': self.bubble_config.options
            },
            'detection_params': self.detection_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OMRTemplate':
        """Create template from dictionary."""
        sections = [
            SectionConfig(
                name=s['name'],
                start_question=s['start_question'],
                end_question=s['end_question'],
                start_position=s['start_position'],
                columns=s.get('columns', 1)
            )
            for s in data['sections']
        ]
        
        bubble_config = BubbleConfig(
            radius=data['bubble_config']['radius'],
            inner_radius=data['bubble_config']['inner_radius'],
            spacing_horizontal=data['bubble_config']['spacing_horizontal'],
            spacing_vertical=data['bubble_config']['spacing_vertical'],
            options=data['bubble_config']['options']
        )
        
        return cls(
            name=data['name'],
            description=data['description'],
            page_dimensions=data['page_dimensions'],
            sections=sections,
            bubble_config=bubble_config,
            detection_params=data['detection_params']
        )

class TemplateManager:
    """Manages OMR sheet templates."""
    
    def __init__(self):
        self.templates: Dict[str, OMRTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default template configurations."""
        
        # Standard template (current default)
        standard_template = OMRTemplate(
            name="standard",
            description="Standard 40-question OMR sheet with 2 sections",
            page_dimensions={"width": 800, "height": 1000},
            sections=[
                SectionConfig(
                    name="Mathematics",
                    start_question=1,
                    end_question=20,
                    start_position={"x": 100, "y": 160}
                ),
                SectionConfig(
                    name="Science",
                    start_question=21,
                    end_question=40,
                    start_position={"x": 100, "y": 860}
                )
            ],
            bubble_config=BubbleConfig(),
            detection_params={
                "min_bubble_area_ratio": 0.0001,
                "max_bubble_area_ratio": 0.0025,
                "circularity_threshold": 0.3,
                "fill_threshold": 0.6,
                "confidence_threshold": 0.6
            }
        )
        
        # Compact template
        compact_template = OMRTemplate(
            name="compact",
            description="Compact 50-question OMR sheet with smaller spacing",
            page_dimensions={"width": 600, "height": 800},
            sections=[
                SectionConfig(
                    name="Section A",
                    start_question=1,
                    end_question=25,
                    start_position={"x": 50, "y": 100}
                ),
                SectionConfig(
                    name="Section B",
                    start_question=26,
                    end_question=50,
                    start_position={"x": 350, "y": 100}
                )
            ],
            bubble_config=BubbleConfig(
                radius=10,
                inner_radius=6,
                spacing_horizontal=60,
                spacing_vertical=25
            ),
            detection_params={
                "min_bubble_area_ratio": 0.00008,
                "max_bubble_area_ratio": 0.002,
                "circularity_threshold": 0.25,
                "fill_threshold": 0.55,
                "confidence_threshold": 0.55
            }
        )
        
        # Multi-column template
        multicolumn_template = OMRTemplate(
            name="multicolumn",
            description="100-question OMR sheet with multiple columns",
            page_dimensions={"width": 1000, "height": 1200},
            sections=[
                SectionConfig(
                    name="Part I",
                    start_question=1,
                    end_question=50,
                    start_position={"x": 50, "y": 100},
                    columns=2
                ),
                SectionConfig(
                    name="Part II",
                    start_question=51,
                    end_question=100,
                    start_position={"x": 50, "y": 700},
                    columns=2
                )
            ],
            bubble_config=BubbleConfig(
                spacing_horizontal=70,
                spacing_vertical=20
            ),
            detection_params={
                "min_bubble_area_ratio": 0.00005,
                "max_bubble_area_ratio": 0.0015,
                "circularity_threshold": 0.3,
                "fill_threshold": 0.6,
                "confidence_threshold": 0.6
            }
        )
        
        self.templates["standard"] = standard_template
        self.templates["compact"] = compact_template
        self.templates["multicolumn"] = multicolumn_template
    
    def get_template(self, name: str) -> Optional[OMRTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self) -> Dict[str, str]:
        """Get template names and descriptions."""
        return {name: template.description for name, template in self.templates.items()}
    
    def add_custom_template(self, template: OMRTemplate):
        """Add a custom template."""
        self.templates[template.name] = template
    
    def save_template(self, template: OMRTemplate, filepath: str):
        """Save template to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def load_template(self, filepath: str) -> OMRTemplate:
        """Load template from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return OMRTemplate.from_dict(data)
    
    def calculate_expected_positions(self, template_name: str) -> Dict[str, Dict]:
        """Calculate expected bubble positions for a template."""
        template = self.get_template(template_name)
        if not template:
            return {}
        
        positions = {}
        
        for section in template.sections:
            questions_in_section = section.end_question - section.start_question + 1
            questions_per_column = questions_in_section // section.columns
            
            for i, q_num in enumerate(range(section.start_question, section.end_question + 1)):
                # Calculate column and row within column
                column = i // questions_per_column if section.columns > 1 else 0
                row_in_column = i % questions_per_column
                
                # Calculate question position
                question_x = section.start_position["x"] + (column * 400)  # 400px between columns
                question_y = section.start_position["y"] + (row_in_column * template.bubble_config.spacing_vertical)
                
                # Calculate bubble positions for this question
                bubble_positions = []
                for j, option in enumerate(template.bubble_config.options):
                    bubble_x = question_x + 50 + (j * template.bubble_config.spacing_horizontal)
                    bubble_y = question_y + 10
                    
                    bubble_positions.append({
                        "option": option,
                        "center": (bubble_x + template.bubble_config.radius, bubble_y + template.bubble_config.radius),
                        "radius": template.bubble_config.radius
                    })
                
                positions[str(q_num)] = {
                    "section": section.name,
                    "bubbles": bubble_positions
                }
        
        return positions
    
    def validate_template(self, template: OMRTemplate) -> List[str]:
        """Validate template configuration."""
        errors = []
        
        # Check page dimensions
        if template.page_dimensions["width"] < 200 or template.page_dimensions["height"] < 200:
            errors.append("Page dimensions too small")
        
        # Check sections
        if not template.sections:
            errors.append("No sections defined")
        
        for section in template.sections:
            if section.start_question >= section.end_question:
                errors.append(f"Invalid question range in section {section.name}")
            
            if section.start_position["x"] < 0 or section.start_position["y"] < 0:
                errors.append(f"Invalid start position in section {section.name}")
        
        # Check bubble configuration
        if template.bubble_config.radius <= 0:
            errors.append("Bubble radius must be positive")
        
        if not template.bubble_config.options:
            errors.append("No bubble options defined")
        
        return errors