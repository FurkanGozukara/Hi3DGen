import json
import os
import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import traceback


@dataclass
class PresetParameters:
    """Data class containing all user-configurable parameters"""
    # Basic generation settings
    seed: int = -1
    
    # Stage 1: Sparse Structure Generation
    ss_guidance_strength: float = 3.0
    ss_sampling_steps: int = 50
    
    # Stage 2: Structured Latent Generation  
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 6
    
    # Mesh Simplification Settings
    poly_count_pcnt: float = 0.5
    
    # UV Unwrapping (xatlas) Settings
    xatlas_max_cost: float = 8.0
    xatlas_normal_seam_weight: float = 1.0
    xatlas_resolution: int = 1024
    xatlas_padding: int = 2
    
    # Normal Map Generation Settings
    normal_map_resolution: int = 768
    normal_match_input_resolution: bool = True
    
    # Auto-Save Settings
    auto_save_obj: bool = True
    auto_save_glb: bool = True
    auto_save_ply: bool = True
    auto_save_stl: bool = True


@dataclass
class Preset:
    """Complete preset data structure"""
    name: str
    created_at: str
    parameters: PresetParameters
    description: str = ""
    version: str = "1.0"


class PresetManager:
    """Manages saving, loading, and organizing user presets"""
    
    def __init__(self, presets_dir: str = "presets"):
        self.presets_dir = presets_dir
        self.last_used_file = os.path.join(presets_dir, "last_used_preset.txt")
        self._ensure_presets_directory()
        
    def _ensure_presets_directory(self):
        """Create presets directory if it doesn't exist"""
        try:
            os.makedirs(self.presets_dir, exist_ok=True)
            print(f"✓ Presets directory ready: {self.presets_dir}")
        except Exception as e:
            print(f"❌ Error creating presets directory: {e}")
            raise
    
    def _get_preset_file_path(self, preset_name: str) -> str:
        """Get the full file path for a preset"""
        safe_name = self._sanitize_filename(preset_name)
        return os.path.join(self.presets_dir, f"{safe_name}.json")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and strip whitespace
        filename = filename.strip()[:50]  # Limit to 50 characters
        
        # Ensure it's not empty
        if not filename:
            filename = "unnamed_preset"
            
        return filename
    
    def create_default_preset(self) -> Preset:
        """Create a default preset with standard values"""
        try:
            default_params = PresetParameters()  # Uses default values from dataclass
            preset = Preset(
                name="Default",
                created_at=datetime.datetime.now().isoformat(),
                parameters=default_params,
                description="Default Hi3DGen settings with balanced quality and performance"
            )
            
            print("✓ Created default preset")
            return preset
            
        except Exception as e:
            print(f"❌ Error creating default preset: {e}")
            traceback.print_exc()
            raise
    
    def save_preset(self, preset_name: str, parameters: PresetParameters, 
                   description: str = "", overwrite: bool = False) -> Tuple[bool, str]:
        """
        Save a preset to disk
        
        Args:
            preset_name: Name of the preset
            parameters: PresetParameters object with all settings
            description: Optional description for the preset
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not preset_name or not preset_name.strip():
                return False, "Preset name cannot be empty"
            
            preset_name = preset_name.strip()
            preset_path = self._get_preset_file_path(preset_name)
            
            # Check if preset already exists
            if os.path.exists(preset_path) and not overwrite:
                return False, f"Preset '{preset_name}' already exists. Use overwrite=True to replace it."
            
            # Create preset object
            preset = Preset(
                name=preset_name,
                created_at=datetime.datetime.now().isoformat(),
                parameters=parameters,
                description=description
            )
            
            # Convert to dictionary for JSON serialization
            preset_dict = asdict(preset)
            
            # Save to file
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_dict, f, indent=2, ensure_ascii=False)
            
            # Update last used preset
            self._set_last_used_preset(preset_name)
            
            print(f"✓ Saved preset '{preset_name}' to {preset_path}")
            return True, f"Preset '{preset_name}' saved successfully"
            
        except Exception as e:
            error_msg = f"Error saving preset '{preset_name}': {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, error_msg
    
    def load_preset(self, preset_name: str) -> Tuple[Optional[Preset], str]:
        """
        Load a preset from disk
        
        Args:
            preset_name: Name of the preset to load
            
        Returns:
            Tuple of (preset: Optional[Preset], message: str)
        """
        try:
            if not preset_name or not preset_name.strip():
                return None, "Preset name cannot be empty"
            
            preset_path = self._get_preset_file_path(preset_name.strip())
            
            if not os.path.exists(preset_path):
                return None, f"Preset '{preset_name}' not found"
            
            # Load from file
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_dict = json.load(f)
            
            # Validate and convert to preset object
            preset = self._dict_to_preset(preset_dict)
            
            if preset is None:
                return None, f"Invalid preset format in '{preset_name}'"
            
            # Update last used preset
            self._set_last_used_preset(preset_name)
            
            print(f"✓ Loaded preset '{preset_name}' from {preset_path}")
            return preset, f"Preset '{preset_name}' loaded successfully"
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in preset '{preset_name}': {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Error loading preset '{preset_name}': {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return None, error_msg
    
    def _dict_to_preset(self, preset_dict: Dict[str, Any]) -> Optional[Preset]:
        """Convert dictionary to Preset object with error handling and defaults"""
        try:
            # Extract basic preset info
            name = preset_dict.get('name', 'Unnamed')
            created_at = preset_dict.get('created_at', datetime.datetime.now().isoformat())
            description = preset_dict.get('description', '')
            version = preset_dict.get('version', '1.0')
            
            # Extract parameters with defaults
            params_dict = preset_dict.get('parameters', {})
            
            # Create PresetParameters with defaults for missing values
            default_params = PresetParameters()
            parameters = PresetParameters(
                seed=params_dict.get('seed', default_params.seed),
                ss_guidance_strength=params_dict.get('ss_guidance_strength', default_params.ss_guidance_strength),
                ss_sampling_steps=params_dict.get('ss_sampling_steps', default_params.ss_sampling_steps),
                slat_guidance_strength=params_dict.get('slat_guidance_strength', default_params.slat_guidance_strength),
                slat_sampling_steps=params_dict.get('slat_sampling_steps', default_params.slat_sampling_steps),
                poly_count_pcnt=params_dict.get('poly_count_pcnt', default_params.poly_count_pcnt),
                xatlas_max_cost=params_dict.get('xatlas_max_cost', default_params.xatlas_max_cost),
                xatlas_normal_seam_weight=params_dict.get('xatlas_normal_seam_weight', default_params.xatlas_normal_seam_weight),
                xatlas_resolution=params_dict.get('xatlas_resolution', default_params.xatlas_resolution),
                xatlas_padding=params_dict.get('xatlas_padding', default_params.xatlas_padding),
                normal_map_resolution=params_dict.get('normal_map_resolution', default_params.normal_map_resolution),
                normal_match_input_resolution=params_dict.get('normal_match_input_resolution', default_params.normal_match_input_resolution),
                auto_save_obj=params_dict.get('auto_save_obj', default_params.auto_save_obj),
                auto_save_glb=params_dict.get('auto_save_glb', default_params.auto_save_glb),
                auto_save_ply=params_dict.get('auto_save_ply', default_params.auto_save_ply),
                auto_save_stl=params_dict.get('auto_save_stl', default_params.auto_save_stl)
            )
            
            preset = Preset(
                name=name,
                created_at=created_at,
                parameters=parameters,
                description=description,
                version=version
            )
            
            return preset
            
        except Exception as e:
            print(f"❌ Error converting dict to preset: {e}")
            traceback.print_exc()
            return None
    
    def get_available_presets(self) -> List[str]:
        """Get list of available preset names"""
        try:
            if not os.path.exists(self.presets_dir):
                return []
            
            preset_files = [
                f[:-5]  # Remove .json extension
                for f in os.listdir(self.presets_dir)
                if f.endswith('.json') and f != 'last_used_preset.txt'
            ]
            
            # Sort presets, with 'Default' first if it exists
            preset_files.sort()
            if 'Default' in preset_files:
                preset_files.remove('Default')
                preset_files.insert(0, 'Default')
            
            print(f"✓ Found {len(preset_files)} presets: {preset_files}")
            return preset_files
            
        except Exception as e:
            print(f"❌ Error getting available presets: {e}")
            traceback.print_exc()
            return []
    
    def delete_preset(self, preset_name: str) -> Tuple[bool, str]:
        """
        Delete a preset from disk
        
        Args:
            preset_name: Name of the preset to delete
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not preset_name or not preset_name.strip():
                return False, "Preset name cannot be empty"
            
            preset_name = preset_name.strip()
            
            # Don't allow deleting the default preset
            if preset_name.lower() == 'default':
                return False, "Cannot delete the Default preset"
            
            preset_path = self._get_preset_file_path(preset_name)
            
            if not os.path.exists(preset_path):
                return False, f"Preset '{preset_name}' not found"
            
            # Delete the file
            os.remove(preset_path)
            
            # If this was the last used preset, clear it
            last_used = self.get_last_used_preset()
            if last_used == preset_name:
                self._set_last_used_preset("Default")
            
            print(f"✓ Deleted preset '{preset_name}'")
            return True, f"Preset '{preset_name}' deleted successfully"
            
        except Exception as e:
            error_msg = f"Error deleting preset '{preset_name}': {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, error_msg
    
    def get_last_used_preset(self) -> str:
        """Get the name of the last used preset"""
        try:
            if os.path.exists(self.last_used_file):
                with open(self.last_used_file, 'r', encoding='utf-8') as f:
                    last_used = f.read().strip()
                    if last_used:
                        return last_used
            
            # Default fallback
            return "Default"
            
        except Exception as e:
            print(f"❌ Error reading last used preset: {e}")
            return "Default"
    
    def _set_last_used_preset(self, preset_name: str):
        """Set the last used preset name"""
        try:
            with open(self.last_used_file, 'w', encoding='utf-8') as f:
                f.write(preset_name.strip())
            print(f"✓ Set last used preset to '{preset_name}'")
            
        except Exception as e:
            print(f"❌ Error setting last used preset: {e}")
    
    def initialize_presets(self) -> Tuple[bool, str, List[str]]:
        """
        Initialize preset system on app startup
        
        Returns:
            Tuple of (success: bool, message: str, available_presets: List[str])
        """
        try:
            print("🔧 Initializing preset system...")
            
            # Ensure directory exists
            self._ensure_presets_directory()
            
            # Check if default preset exists, create if not
            available_presets = self.get_available_presets()
            if "Default" not in available_presets:
                print("📝 Creating default preset...")
                default_preset = self.create_default_preset()
                success, msg = self.save_preset("Default", default_preset.parameters, 
                                              default_preset.description, overwrite=True)
                if not success:
                    return False, f"Failed to create default preset: {msg}", []
            
            # Refresh available presets
            available_presets = self.get_available_presets()
            
            # Validate last used preset
            last_used = self.get_last_used_preset()
            if last_used not in available_presets:
                print(f"⚠️ Last used preset '{last_used}' not found, defaulting to 'Default'")
                self._set_last_used_preset("Default")
                last_used = "Default"
            
            print(f"✅ Preset system initialized successfully")
            print(f"   Available presets: {available_presets}")
            print(f"   Last used preset: {last_used}")
            
            return True, f"Preset system ready. Last used: {last_used}", available_presets
            
        except Exception as e:
            error_msg = f"Failed to initialize preset system: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, error_msg, []
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a preset without fully loading it"""
        try:
            preset_path = self._get_preset_file_path(preset_name)
            
            if not os.path.exists(preset_path):
                return {"exists": False, "error": "Preset not found"}
            
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_dict = json.load(f)
            
            return {
                "exists": True,
                "name": preset_dict.get('name', preset_name),
                "created_at": preset_dict.get('created_at', 'Unknown'),
                "description": preset_dict.get('description', ''),
                "version": preset_dict.get('version', '1.0'),
                "file_size": os.path.getsize(preset_path)
            }
            
        except Exception as e:
            return {"exists": False, "error": str(e)}


# Global preset manager instance
_preset_manager = None

def get_preset_manager(presets_dir: str = "presets") -> PresetManager:
    """Get or create the global preset manager instance"""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager(presets_dir)
    return _preset_manager


# Convenience functions for external use
def create_preset_parameters_from_values(
    seed: int = -1,
    ss_guidance_strength: float = 3.0,
    ss_sampling_steps: int = 50,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 6,
    poly_count_pcnt: float = 0.5,
    xatlas_max_cost: float = 8.0,
    xatlas_normal_seam_weight: float = 1.0,
    xatlas_resolution: int = 1024,
    xatlas_padding: int = 2,
    normal_map_resolution: int = 768,
    normal_match_input_resolution: bool = True,
    auto_save_obj: bool = True,
    auto_save_glb: bool = True,
    auto_save_ply: bool = True,
    auto_save_stl: bool = True
) -> PresetParameters:
    """Create a PresetParameters object from individual values"""
    return PresetParameters(
        seed=seed,
        ss_guidance_strength=ss_guidance_strength,
        ss_sampling_steps=ss_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        slat_sampling_steps=slat_sampling_steps,
        poly_count_pcnt=poly_count_pcnt,
        xatlas_max_cost=xatlas_max_cost,
        xatlas_normal_seam_weight=xatlas_normal_seam_weight,
        xatlas_resolution=xatlas_resolution,
        xatlas_padding=xatlas_padding,
        normal_map_resolution=normal_map_resolution,
        normal_match_input_resolution=normal_match_input_resolution,
        auto_save_obj=auto_save_obj,
        auto_save_glb=auto_save_glb,
        auto_save_ply=auto_save_ply,
        auto_save_stl=auto_save_stl
    )


# UI Integration Functions
# These functions provide direct interface between Gradio UI and preset system

def collect_parameters_from_ui(
    seed, ss_guidance_strength, ss_sampling_steps,
    slat_guidance_strength, slat_sampling_steps, poly_count_pcnt,
    xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
    normal_map_resolution, normal_match_input_resolution,
    auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl
) -> PresetParameters:
    """
    Collect parameters from Gradio UI components and create PresetParameters object
    
    Args:
        All the individual parameter values from UI components
        
    Returns:
        PresetParameters object with all current UI values
    """
    return PresetParameters(
        seed=int(seed) if seed is not None else -1,
        ss_guidance_strength=float(ss_guidance_strength) if ss_guidance_strength is not None else 3.0,
        ss_sampling_steps=int(ss_sampling_steps) if ss_sampling_steps is not None else 50,
        slat_guidance_strength=float(slat_guidance_strength) if slat_guidance_strength is not None else 3.0,
        slat_sampling_steps=int(slat_sampling_steps) if slat_sampling_steps is not None else 6,
        poly_count_pcnt=float(poly_count_pcnt) if poly_count_pcnt is not None else 0.5,
        xatlas_max_cost=float(xatlas_max_cost) if xatlas_max_cost is not None else 8.0,
        xatlas_normal_seam_weight=float(xatlas_normal_seam_weight) if xatlas_normal_seam_weight is not None else 1.0,
        xatlas_resolution=int(xatlas_resolution) if xatlas_resolution is not None else 1024,
        xatlas_padding=int(xatlas_padding) if xatlas_padding is not None else 2,
        normal_map_resolution=int(normal_map_resolution) if normal_map_resolution is not None else 768,
        normal_match_input_resolution=bool(normal_match_input_resolution) if normal_match_input_resolution is not None else True,
        auto_save_obj=bool(auto_save_obj) if auto_save_obj is not None else True,
        auto_save_glb=bool(auto_save_glb) if auto_save_glb is not None else True,
        auto_save_ply=bool(auto_save_ply) if auto_save_ply is not None else True,
        auto_save_stl=bool(auto_save_stl) if auto_save_stl is not None else True
    )


def extract_parameters_for_ui(preset: Preset) -> Tuple:
    """
    Extract parameters from preset for updating Gradio UI components
    
    Args:
        preset: Preset object to extract parameters from
        
    Returns:
        Tuple of all parameter values in the same order as UI components
    """
    params = preset.parameters
    return (
        params.seed,
        params.ss_guidance_strength,
        params.ss_sampling_steps,
        params.slat_guidance_strength,
        params.slat_sampling_steps,
        params.poly_count_pcnt,
        params.xatlas_max_cost,
        params.xatlas_normal_seam_weight,
        params.xatlas_resolution,
        params.xatlas_padding,
        params.normal_map_resolution,
        params.normal_match_input_resolution,
        params.auto_save_obj,
        params.auto_save_glb,
        params.auto_save_ply,
        params.auto_save_stl
    )


def save_preset_from_ui(
    preset_name: str,
    description: str,
    seed, ss_guidance_strength, ss_sampling_steps,
    slat_guidance_strength, slat_sampling_steps, poly_count_pcnt,
    xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
    normal_map_resolution, normal_match_input_resolution,
    auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl,
    overwrite: bool = False
) -> Tuple[bool, str, List[str], str]:
    """
    Save preset from current UI values
    
    Args:
        preset_name: Name for the new preset
        description: Optional description
        All UI parameter values
        overwrite: Whether to overwrite existing preset
        
    Returns:
        Tuple of (success: bool, message: str, updated_preset_list: List[str], selected_preset: str)
    """
    try:
        preset_manager = get_preset_manager()
        
        # Collect parameters from UI
        parameters = collect_parameters_from_ui(
            seed, ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps, poly_count_pcnt,
            xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
            normal_map_resolution, normal_match_input_resolution,
            auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl
        )
        
        # Save preset
        success, message = preset_manager.save_preset(preset_name, parameters, description, overwrite)
        
        # Get updated preset list
        updated_presets = preset_manager.get_available_presets()
        
        # Return the preset name as selected if successful
        selected_preset = preset_name if success else preset_manager.get_last_used_preset()
        
        return success, message, updated_presets, selected_preset
        
    except Exception as e:
        error_msg = f"Error saving preset from UI: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        # Try to get current state for fallback
        try:
            preset_manager = get_preset_manager()
            fallback_presets = preset_manager.get_available_presets()
            fallback_selected = preset_manager.get_last_used_preset()
        except:
            fallback_presets = ["Default"]
            fallback_selected = "Default"
            
        return False, error_msg, fallback_presets, fallback_selected


def load_preset_for_ui(preset_name: str) -> Tuple[bool, str, List[str], str, Tuple]:
    """
    Load preset and return values for updating UI with validation
    
    Args:
        preset_name: Name of preset to load
        
    Returns:
        Tuple of (success: bool, message: str, updated_preset_list: List[str], 
                 selected_preset: str, ui_values: Tuple)
    """
    try:
        preset_manager = get_preset_manager()
        
        # Load preset
        preset, load_message = preset_manager.load_preset(preset_name)
        
        if preset is None:
            # Failed to load, return current state
            updated_presets = preset_manager.get_available_presets()
            current_selected = preset_manager.get_last_used_preset()
            
            # Return default values for UI
            default_params = PresetParameters()
            ui_values = extract_parameters_for_ui(Preset("Default", "", default_params))
            
            return False, load_message, updated_presets, current_selected, ui_values
        
        # Validate parameters before applying to UI
        try:
            from .preset_validation import validate_preset_for_ui
            all_valid, validated_params, validation_message = validate_preset_for_ui(preset.parameters)
            
            if not all_valid:
                print(f"⚠️ Parameter validation issues for preset '{preset_name}': {validation_message}")
                # Use validated parameters anyway, but update the message
                preset.parameters = validated_params
                message = f"{load_message} (Parameters adjusted: {validation_message})"
            else:
                message = load_message
                if "adjustment" in validation_message.lower():
                    message += f" ({validation_message})"
                    
        except ImportError:
            # Validation module not available, proceed without validation
            print("⚠️ Preset validation not available, proceeding without validation")
            message = load_message
        except Exception as validation_error:
            print(f"⚠️ Validation error for preset '{preset_name}': {validation_error}")
            message = f"{load_message} (Validation warning: {str(validation_error)})"
        
        # Extract values for UI
        ui_values = extract_parameters_for_ui(preset)
        updated_presets = preset_manager.get_available_presets()
        
        return True, message, updated_presets, preset_name, ui_values
        
    except Exception as e:
        error_msg = f"Error loading preset for UI: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        # Fallback to default
        try:
            preset_manager = get_preset_manager()
            fallback_presets = preset_manager.get_available_presets()
            fallback_selected = preset_manager.get_last_used_preset()
        except:
            fallback_presets = ["Default"]
            fallback_selected = "Default"
            
        default_params = PresetParameters()
        fallback_ui_values = extract_parameters_for_ui(Preset("Default", "", default_params))
        
        return False, error_msg, fallback_presets, fallback_selected, fallback_ui_values


def delete_preset_from_ui(preset_name: str) -> Tuple[bool, str, List[str], str]:
    """
    Delete preset and return updated UI state
    
    Args:
        preset_name: Name of preset to delete
        
    Returns:
        Tuple of (success: bool, message: str, updated_preset_list: List[str], selected_preset: str)
    """
    try:
        preset_manager = get_preset_manager()
        
        # Delete preset
        success, message = preset_manager.delete_preset(preset_name)
        
        # Get updated preset list
        updated_presets = preset_manager.get_available_presets()
        
        # Get current selected preset (will be Default if deleted preset was last used)
        selected_preset = preset_manager.get_last_used_preset()
        
        return success, message, updated_presets, selected_preset
        
    except Exception as e:
        error_msg = f"Error deleting preset from UI: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        # Fallback
        try:
            preset_manager = get_preset_manager()
            fallback_presets = preset_manager.get_available_presets()
            fallback_selected = preset_manager.get_last_used_preset()
        except:
            fallback_presets = ["Default"]
            fallback_selected = "Default"
            
        return False, error_msg, fallback_presets, fallback_selected


def initialize_presets_for_ui() -> Tuple[bool, str, List[str], str, Tuple]:
    """
    Initialize preset system and return initial UI state
    
    Returns:
        Tuple of (success: bool, message: str, preset_list: List[str], 
                 selected_preset: str, ui_values: Tuple)
    """
    try:
        preset_manager = get_preset_manager()
        
        # Initialize preset system
        success, message, available_presets = preset_manager.initialize_presets()
        
        if not success:
            # Initialization failed, return minimal fallback
            default_params = PresetParameters()
            fallback_ui_values = extract_parameters_for_ui(Preset("Default", "", default_params))
            return False, message, ["Default"], "Default", fallback_ui_values
        
        # Get last used preset
        last_used = preset_manager.get_last_used_preset()
        
        # Load the last used preset for UI
        preset, load_message = preset_manager.load_preset(last_used)
        
        if preset is None:
            # Fallback to default if last used preset failed to load
            print(f"⚠️ Failed to load last used preset '{last_used}': {load_message}")
            default_preset = preset_manager.create_default_preset()
            ui_values = extract_parameters_for_ui(default_preset)
            last_used = "Default"
        else:
            ui_values = extract_parameters_for_ui(preset)
        
        init_message = f"Preset system initialized. Loaded: {last_used}"
        return True, init_message, available_presets, last_used, ui_values
        
    except Exception as e:
        error_msg = f"Error initializing presets for UI: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        # Complete fallback
        default_params = PresetParameters()
        fallback_ui_values = extract_parameters_for_ui(Preset("Default", "", default_params))
        return False, error_msg, ["Default"], "Default", fallback_ui_values


def get_preset_choices_for_ui() -> List[str]:
    """
    Get current list of available presets for dropdown choices
    
    Returns:
        List of preset names for UI dropdown
    """
    try:
        preset_manager = get_preset_manager()
        presets = preset_manager.get_available_presets()
        return presets if presets else ["Default"]
    except Exception as e:
        print(f"❌ Error getting preset choices: {e}")
        return ["Default"]


def validate_preset_name(preset_name: str) -> Tuple[bool, str]:
    """
    Validate preset name for UI input
    
    Args:
        preset_name: Name to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not preset_name or not preset_name.strip():
        return False, "Preset name cannot be empty"
    
    preset_name = preset_name.strip()
    
    if len(preset_name) > 50:
        return False, "Preset name too long (max 50 characters)"
    
    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        if char in preset_name:
            return False, f"Preset name cannot contain: {invalid_chars}"
    
    return True, "Valid preset name"


def get_preset_status_for_ui() -> str:
    """
    Get current preset system status for UI display
    
    Returns:
        Status string for UI
    """
    try:
        preset_manager = get_preset_manager()
        
        available_presets = preset_manager.get_available_presets()
        last_used = preset_manager.get_last_used_preset()
        
        status = f"📋 Presets: {len(available_presets)} available"
        if last_used:
            status += f" | Current: {last_used}"
        
        return status
        
    except Exception as e:
        return f"❌ Preset system error: {str(e)}"


def copy_template_to_presets(template_name: str) -> Tuple[bool, str]:
    """
    Copy a template preset to the main presets folder
    
    Args:
        template_name: Name of the template to copy
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        from .preset_file_manager import get_preset_file_manager
        
        preset_manager = get_preset_manager()
        file_manager = get_preset_file_manager()
        
        # Check if template exists
        template_path = file_manager.templates_dir / f"{template_name}.json"
        if not template_path.exists():
            available_templates = [f.stem for f in file_manager.templates_dir.glob('*.json')]
            return False, f"Template '{template_name}' not found. Available: {', '.join(available_templates)}"
        
        # Load template
        preset, load_message = preset_manager.load_preset(template_name)
        if preset is None:
            # Try loading directly from template file
            try:
                import json
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                preset = preset_manager._dict_to_preset(template_data)
                if preset is None:
                    return False, f"Failed to parse template '{template_name}'"
                    
            except Exception as e:
                return False, f"Error reading template '{template_name}': {str(e)}"
        
        # Save to main presets folder
        success, save_message = preset_manager.save_preset(
            template_name, preset.parameters, preset.description, overwrite=True
        )
        
        if success:
            return True, f"Template '{template_name}' copied to presets successfully"
        else:
            return False, f"Failed to copy template '{template_name}': {save_message}"
            
    except Exception as e:
        error_msg = f"Error copying template '{template_name}': {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg


def get_available_templates() -> List[str]:
    """Get list of available template presets"""
    try:
        from .preset_file_manager import get_preset_file_manager
        file_manager = get_preset_file_manager()
        
        if not file_manager.templates_dir.exists():
            return []
        
        templates = [f.stem for f in file_manager.templates_dir.glob('*.json')]
        return sorted(templates)
        
    except Exception as e:
        print(f"❌ Error getting available templates: {e}")
        return []


def copy_all_templates_to_presets() -> Tuple[bool, str, int]:
    """
    Copy all template presets to main presets folder
    
    Returns:
        Tuple of (success: bool, message: str, copied_count: int)
    """
    try:
        templates = get_available_templates()
        if not templates:
            return True, "No templates found to copy", 0
        
        copied = []
        failed = []
        
        for template_name in templates:
            success, message = copy_template_to_presets(template_name)
            if success:
                copied.append(template_name)
                print(f"✓ Copied template: {template_name}")
            else:
                failed.append(f"{template_name}: {message}")
                print(f"❌ Failed to copy template {template_name}: {message}")
        
        if failed:
            summary = f"Copied {len(copied)}/{len(templates)} templates. Failed: {len(failed)}"
            return False, summary, len(copied)
        else:
            summary = f"Successfully copied all {len(copied)} templates to presets"
            return True, summary, len(copied)
            
    except Exception as e:
        error_msg = f"Error copying templates: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg, 0 