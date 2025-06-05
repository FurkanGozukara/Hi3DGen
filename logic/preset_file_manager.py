import json
import os
import shutil
import datetime
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import traceback

from .preset_manager import PresetManager, PresetParameters, Preset, get_preset_manager


class PresetFileManager:
    """Advanced file management utilities for preset system"""
    
    def __init__(self, presets_dir: str = "presets"):
        self.presets_dir = Path(presets_dir)
        self.backup_dir = self.presets_dir / "backups"
        self.templates_dir = self.presets_dir / "templates"
        
    def setup_folder_structure(self) -> Tuple[bool, str]:
        """
        Create complete folder structure for preset system
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Create main directories
            directories = [
                self.presets_dir,
                self.backup_dir,
                self.templates_dir
            ]
            
            created_dirs = []
            for directory in directories:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(directory))
                    print(f"✓ Created directory: {directory}")
            
            # Create .gitignore if it doesn't exist
            gitignore_path = self.presets_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_content = """# Preset system files
*.tmp
*.bak
.DS_Store
Thumbs.db

# Keep structure but ignore temporary files
backups/*.tmp
templates/*.tmp
"""
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write(gitignore_content)
                print(f"✓ Created .gitignore: {gitignore_path}")
            
            # Create README if it doesn't exist
            readme_path = self.presets_dir / "README.md"
            if not readme_path.exists():
                readme_content = """# Hi3DGen Presets

This folder contains saved presets for Hi3DGen parameters.

## Structure
- `*.json` - Individual preset files
- `last_used_preset.txt` - Tracks the last used preset
- `backups/` - Automatic backups of presets
- `templates/` - Preset templates for different use cases

## Usage
Presets are automatically managed by the Hi3DGen application.
You can manually backup important presets by copying them to the backups folder.

## File Format
Presets are stored as JSON files with the following structure:
```json
{
  "name": "Preset Name",
  "created_at": "2024-01-01T12:00:00",
  "description": "Preset description",
  "version": "1.0",
  "parameters": {
    "seed": -1,
    "ss_guidance_strength": 3.0,
    ...
  }
}
```
"""
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                print(f"✓ Created README: {readme_path}")
            
            if created_dirs:
                message = f"Preset folder structure created: {', '.join(created_dirs)}"
            else:
                message = "Preset folder structure already exists"
                
            print(f"✅ Preset folder structure ready at: {self.presets_dir}")
            return True, message
            
        except Exception as e:
            error_msg = f"Error setting up folder structure: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, error_msg
    
    def create_preset_templates(self) -> Tuple[bool, str]:
        """
        Create default preset templates for common use cases
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            templates = {
                "High_Quality": {
                    "description": "High quality settings for detailed 3D models (slower processing)",
                    "parameters": PresetParameters(
                        ss_guidance_strength=4.0,
                        ss_sampling_steps=50,
                        slat_guidance_strength=4.0,
                        slat_sampling_steps=8,
                        poly_count_pcnt=0.8,
                        xatlas_max_cost=6.0,
                        xatlas_normal_seam_weight=0.8,
                        xatlas_resolution=2048,
                        xatlas_padding=4,
                        normal_map_resolution=1024,
                        normal_match_input_resolution=True
                    )
                },
                "Fast_Preview": {
                    "description": "Fast settings for quick previews (lower quality)",
                    "parameters": PresetParameters(
                        ss_guidance_strength=2.5,
                        ss_sampling_steps=25,
                        slat_guidance_strength=2.5,
                        slat_sampling_steps=4,
                        poly_count_pcnt=0.3,
                        xatlas_max_cost=10.0,
                        xatlas_normal_seam_weight=1.5,
                        xatlas_resolution=512,
                        xatlas_padding=1,
                        normal_map_resolution=512,
                        normal_match_input_resolution=False
                    )
                },
                "Balanced": {
                    "description": "Balanced settings for good quality and reasonable speed",
                    "parameters": PresetParameters(
                        ss_guidance_strength=3.0,
                        ss_sampling_steps=40,
                        slat_guidance_strength=3.0,
                        slat_sampling_steps=6,
                        poly_count_pcnt=0.6,
                        xatlas_max_cost=8.0,
                        xatlas_normal_seam_weight=1.0,
                        xatlas_resolution=1024,
                        xatlas_padding=2,
                        normal_map_resolution=768,
                        normal_match_input_resolution=True
                    )
                },
                "Low_Memory": {
                    "description": "Settings optimized for systems with limited GPU memory",
                    "parameters": PresetParameters(
                        ss_guidance_strength=2.5,
                        ss_sampling_steps=30,
                        slat_guidance_strength=2.5,
                        slat_sampling_steps=4,
                        poly_count_pcnt=0.4,
                        xatlas_max_cost=9.0,
                        xatlas_normal_seam_weight=1.2,
                        xatlas_resolution=512,
                        xatlas_padding=1,
                        normal_map_resolution=512,
                        normal_match_input_resolution=False
                    )
                }
            }
            
            created_templates = []
            
            for template_name, template_data in templates.items():
                template_path = self.templates_dir / f"{template_name}.json"
                
                # Only create if doesn't exist
                if not template_path.exists():
                    preset = Preset(
                        name=template_name,
                        created_at=datetime.datetime.now().isoformat(),
                        parameters=template_data["parameters"],
                        description=template_data["description"],
                        version="1.0"
                    )
                    
                    # Convert to dict for JSON serialization
                    preset_dict = {
                        'name': preset.name,
                        'created_at': preset.created_at,
                        'description': preset.description,
                        'version': preset.version,
                        'parameters': {
                            'seed': preset.parameters.seed,
                            'ss_guidance_strength': preset.parameters.ss_guidance_strength,
                            'ss_sampling_steps': preset.parameters.ss_sampling_steps,
                            'slat_guidance_strength': preset.parameters.slat_guidance_strength,
                            'slat_sampling_steps': preset.parameters.slat_sampling_steps,
                            'poly_count_pcnt': preset.parameters.poly_count_pcnt,
                            'xatlas_max_cost': preset.parameters.xatlas_max_cost,
                            'xatlas_normal_seam_weight': preset.parameters.xatlas_normal_seam_weight,
                            'xatlas_resolution': preset.parameters.xatlas_resolution,
                            'xatlas_padding': preset.parameters.xatlas_padding,
                            'normal_map_resolution': preset.parameters.normal_map_resolution,
                            'normal_match_input_resolution': preset.parameters.normal_match_input_resolution,
                            'auto_save_obj': preset.parameters.auto_save_obj,
                            'auto_save_glb': preset.parameters.auto_save_glb,
                            'auto_save_ply': preset.parameters.auto_save_ply,
                            'auto_save_stl': preset.parameters.auto_save_stl
                        }
                    }
                    
                    # Save template
                    with open(template_path, 'w', encoding='utf-8') as f:
                        json.dump(preset_dict, f, indent=2, ensure_ascii=False)
                    
                    created_templates.append(template_name)
                    print(f"✓ Created template: {template_name}")
            
            if created_templates:
                message = f"Created {len(created_templates)} preset templates: {', '.join(created_templates)}"
            else:
                message = "Preset templates already exist"
                
            return True, message
            
        except Exception as e:
            error_msg = f"Error creating preset templates: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, error_msg
    
    def validate_preset_file(self, preset_path: Path) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate a preset file for correct format and data
        
        Args:
            preset_path: Path to preset file
            
        Returns:
            Tuple of (is_valid: bool, message: str, preset_data: Optional[Dict])
        """
        try:
            if not preset_path.exists():
                return False, "File does not exist", None
            
            if not preset_path.suffix.lower() == '.json':
                return False, "File is not a JSON file", None
            
            # Try to load JSON
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            # Validate structure
            required_fields = ['name', 'created_at', 'parameters']
            missing_fields = [field for field in required_fields if field not in preset_data]
            
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}", preset_data
            
            # Validate parameters structure
            parameters = preset_data.get('parameters', {})
            if not isinstance(parameters, dict):
                return False, "Parameters field must be a dictionary", preset_data
            
            # Check for some key parameters (not all required due to backward compatibility)
            expected_params = ['seed', 'ss_guidance_strength', 'slat_guidance_strength']
            has_expected = any(param in parameters for param in expected_params)
            
            if not has_expected:
                return False, "Parameters missing expected fields", preset_data
            
            return True, "Valid preset file", preset_data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except Exception as e:
            return False, f"Error validating file: {str(e)}", None
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about preset storage usage
        
        Returns:
            Dictionary with storage information
        """
        try:
            info = {
                "presets_dir": str(self.presets_dir),
                "exists": self.presets_dir.exists(),
                "preset_files": 0,
                "backup_files": 0,
                "template_files": 0,
                "total_size": 0,
                "breakdown": {}
            }
            
            if not self.presets_dir.exists():
                return info
            
            # Count preset files
            preset_files = list(self.presets_dir.glob('*.json'))
            info["preset_files"] = len(preset_files)
            
            preset_size = sum(f.stat().st_size for f in preset_files)
            info["breakdown"]["presets"] = preset_size
            
            # Count backup files
            if self.backup_dir.exists():
                backup_files = list(self.backup_dir.rglob('*'))
                backup_files = [f for f in backup_files if f.is_file()]
                info["backup_files"] = len(backup_files)
                
                backup_size = sum(f.stat().st_size for f in backup_files)
                info["breakdown"]["backups"] = backup_size
            else:
                info["breakdown"]["backups"] = 0
            
            # Count template files
            if self.templates_dir.exists():
                template_files = list(self.templates_dir.glob('*.json'))
                info["template_files"] = len(template_files)
                
                template_size = sum(f.stat().st_size for f in template_files)
                info["breakdown"]["templates"] = template_size
            else:
                info["breakdown"]["templates"] = 0
            
            info["total_size"] = sum(info["breakdown"].values())
            
            return info
            
        except Exception as e:
            print(f"❌ Error getting storage info: {e}")
            return {"error": str(e)}


def test_preset_file_management() -> bool:
    """
    Test the preset file management system
    
    Returns:
        bool: True if all tests pass
    """
    print("🧪 Testing Preset File Management System...")
    
    try:
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_preset_dir = Path(temp_dir) / "test_presets"
            
            # Initialize file manager
            file_manager = PresetFileManager(str(test_preset_dir))
            
            # Test 1: Setup folder structure
            print("\n📁 Test 1: Folder Structure Setup")
            success, message = file_manager.setup_folder_structure()
            if not success:
                print(f"❌ Folder setup failed: {message}")
                return False
            print(f"✅ {message}")
            
            # Test 2: Create templates
            print("\n📋 Test 2: Template Creation")
            success, message = file_manager.create_preset_templates()
            if not success:
                print(f"❌ Template creation failed: {message}")
                return False
            print(f"✅ {message}")
            
            # Test 3: Validate created templates
            print("\n✅ Test 3: Template Validation")
            template_files = list(file_manager.templates_dir.glob('*.json'))
            all_valid = True
            for template_file in template_files:
                is_valid, validation_message, _ = file_manager.validate_preset_file(template_file)
                if not is_valid:
                    print(f"❌ Template {template_file.name} validation failed: {validation_message}")
                    all_valid = False
                else:
                    print(f"✅ Template {template_file.name} is valid")
            
            if not all_valid:
                return False
            
            # Test 4: Storage info
            print("\n💾 Test 4: Storage Information")
            storage_info = file_manager.get_storage_info()
            if "error" in storage_info:
                print(f"❌ Storage info failed: {storage_info['error']}")
                return False
            
            print(f"✅ Storage info collected:")
            print(f"   Preset files: {storage_info['preset_files']}")
            print(f"   Template files: {storage_info['template_files']}")
            print(f"   Total size: {storage_info['total_size']} bytes")
            
            print("\n🎉 All preset file management tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        traceback.print_exc()
        return False


# Global file manager instance
_file_manager = None

def get_preset_file_manager(presets_dir: str = "presets") -> PresetFileManager:
    """Get or create the global preset file manager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = PresetFileManager(presets_dir)
    return _file_manager


def initialize_preset_file_system(presets_dir: str = "presets") -> Tuple[bool, str]:
    """
    Initialize the complete preset file system
    
    Args:
        presets_dir: Directory for presets
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        print("🔧 Initializing preset file system...")
        
        file_manager = get_preset_file_manager(presets_dir)
        
        # Setup folder structure
        success, message = file_manager.setup_folder_structure()
        if not success:
            return False, f"Folder setup failed: {message}"
        
        # Create templates
        success, template_message = file_manager.create_preset_templates()
        if not success:
            print(f"⚠️ Template creation failed: {template_message}")
        
        # Get storage info
        storage_info = file_manager.get_storage_info()
        
        print(f"✅ Preset file system initialized")
        print(f"   Directory: {storage_info.get('presets_dir', 'Unknown')}")
        print(f"   Preset files: {storage_info.get('preset_files', 0)}")
        print(f"   Template files: {storage_info.get('template_files', 0)}")
        
        return True, "Preset file system ready"
        
    except Exception as e:
        error_msg = f"Failed to initialize preset file system: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg 