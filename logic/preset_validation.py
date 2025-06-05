import traceback
from typing import Tuple, Dict, Any, Optional
from .preset_manager import PresetParameters


class ParameterValidator:
    """Validates and sanitizes preset parameters for UI compatibility"""
    
    def __init__(self):
        # Define parameter constraints based on UI component limits
        self.constraints = {
            'seed': {'min': -1, 'max': 2147483647, 'type': int},
            'ss_guidance_strength': {'min': 0.0, 'max': 10.0, 'type': float},
            'ss_sampling_steps': {'min': 1, 'max': 100, 'type': int},
            'slat_guidance_strength': {'min': 0.0, 'max': 10.0, 'type': float},
            'slat_sampling_steps': {'min': 1, 'max': 100, 'type': int},
            'poly_count_pcnt': {'min': 0.05, 'max': 1.0, 'type': float},
            'xatlas_max_cost': {'min': 1.0, 'max': 20.0, 'type': float},
            'xatlas_normal_seam_weight': {'min': 0.1, 'max': 5.0, 'type': float},
            'xatlas_resolution': {'min': 256, 'max': 4096, 'type': int, 'step': 256},
            'xatlas_padding': {'min': 0, 'max': 16, 'type': int},
            'normal_map_resolution': {'min': 256, 'max': 1024, 'type': int, 'step': 128},
            'normal_match_input_resolution': {'type': bool},
            'auto_save_obj': {'type': bool},
            'auto_save_glb': {'type': bool},
            'auto_save_ply': {'type': bool},
            'auto_save_stl': {'type': bool}
        }
    
    def validate_parameter(self, param_name: str, value: Any) -> Tuple[bool, Any, str]:
        """
        Validate and sanitize a single parameter
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            
        Returns:
            Tuple of (is_valid: bool, sanitized_value: Any, message: str)
        """
        try:
            if param_name not in self.constraints:
                return False, value, f"Unknown parameter: {param_name}"
            
            constraint = self.constraints[param_name]
            param_type = constraint['type']
            
            # Type conversion
            try:
                if param_type == bool:
                    if isinstance(value, bool):
                        sanitized_value = value
                    elif isinstance(value, (int, float)):
                        sanitized_value = bool(value)
                    elif isinstance(value, str):
                        sanitized_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        sanitized_value = bool(value)
                elif param_type == int:
                    sanitized_value = int(float(value))  # Handle string numbers
                elif param_type == float:
                    sanitized_value = float(value)
                else:
                    sanitized_value = value
            except (ValueError, TypeError) as e:
                return False, value, f"Invalid type for {param_name}: {str(e)}"
            
            # Range validation for numeric types
            if param_type in (int, float):
                if 'min' in constraint and sanitized_value < constraint['min']:
                    sanitized_value = constraint['min']
                    message = f"Parameter {param_name} clamped to minimum value {constraint['min']}"
                elif 'max' in constraint and sanitized_value > constraint['max']:
                    sanitized_value = constraint['max']
                    message = f"Parameter {param_name} clamped to maximum value {constraint['max']}"
                else:
                    message = f"Parameter {param_name} validated successfully"
                
                # Step validation for stepped parameters
                if 'step' in constraint and constraint['step'] > 0:
                    min_val = constraint.get('min', 0)
                    step = constraint['step']
                    steps_from_min = round((sanitized_value - min_val) / step)
                    sanitized_value = min_val + (steps_from_min * step)
                    sanitized_value = param_type(sanitized_value)  # Ensure correct type
            else:
                message = f"Parameter {param_name} validated successfully"
            
            return True, sanitized_value, message
            
        except Exception as e:
            error_msg = f"Error validating parameter {param_name}: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False, value, error_msg
    
    def validate_preset_parameters(self, parameters: PresetParameters) -> Tuple[bool, PresetParameters, Dict[str, str]]:
        """
        Validate all parameters in a preset
        
        Args:
            parameters: PresetParameters object to validate
            
        Returns:
            Tuple of (all_valid: bool, sanitized_parameters: PresetParameters, messages: Dict[str, str])
        """
        try:
            messages = {}
            sanitized_values = {}
            all_valid = True
            
            # Get all parameter values as dict
            param_dict = {
                'seed': parameters.seed,
                'ss_guidance_strength': parameters.ss_guidance_strength,
                'ss_sampling_steps': parameters.ss_sampling_steps,
                'slat_guidance_strength': parameters.slat_guidance_strength,
                'slat_sampling_steps': parameters.slat_sampling_steps,
                'poly_count_pcnt': parameters.poly_count_pcnt,
                'xatlas_max_cost': parameters.xatlas_max_cost,
                'xatlas_normal_seam_weight': parameters.xatlas_normal_seam_weight,
                'xatlas_resolution': parameters.xatlas_resolution,
                'xatlas_padding': parameters.xatlas_padding,
                'normal_map_resolution': parameters.normal_map_resolution,
                'normal_match_input_resolution': parameters.normal_match_input_resolution,
                'auto_save_obj': parameters.auto_save_obj,
                'auto_save_glb': parameters.auto_save_glb,
                'auto_save_ply': parameters.auto_save_ply,
                'auto_save_stl': parameters.auto_save_stl
            }
            
            # Validate each parameter
            for param_name, value in param_dict.items():
                is_valid, sanitized_value, message = self.validate_parameter(param_name, value)
                
                sanitized_values[param_name] = sanitized_value
                messages[param_name] = message
                
                if not is_valid:
                    all_valid = False
                    print(f"⚠️ Parameter validation issue: {message}")
            
            # Create sanitized parameters object
            sanitized_parameters = PresetParameters(
                seed=sanitized_values['seed'],
                ss_guidance_strength=sanitized_values['ss_guidance_strength'],
                ss_sampling_steps=sanitized_values['ss_sampling_steps'],
                slat_guidance_strength=sanitized_values['slat_guidance_strength'],
                slat_sampling_steps=sanitized_values['slat_sampling_steps'],
                poly_count_pcnt=sanitized_values['poly_count_pcnt'],
                xatlas_max_cost=sanitized_values['xatlas_max_cost'],
                xatlas_normal_seam_weight=sanitized_values['xatlas_normal_seam_weight'],
                xatlas_resolution=sanitized_values['xatlas_resolution'],
                xatlas_padding=sanitized_values['xatlas_padding'],
                normal_map_resolution=sanitized_values['normal_map_resolution'],
                normal_match_input_resolution=sanitized_values['normal_match_input_resolution'],
                auto_save_obj=sanitized_values['auto_save_obj'],
                auto_save_glb=sanitized_values['auto_save_glb'],
                auto_save_ply=sanitized_values['auto_save_ply'],
                auto_save_stl=sanitized_values['auto_save_stl']
            )
            
            return all_valid, sanitized_parameters, messages
            
        except Exception as e:
            error_msg = f"Error validating preset parameters: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            # Return original parameters on error
            return False, parameters, {'error': error_msg}
    
    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get constraint information for a parameter"""
        return self.constraints.get(param_name, {})
    
    def get_all_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameter constraints"""
        return self.constraints.copy()


# Global validator instance
_validator = None

def get_parameter_validator() -> ParameterValidator:
    """Get or create the global parameter validator instance"""
    global _validator
    if _validator is None:
        _validator = ParameterValidator()
    return _validator


def validate_preset_for_ui(parameters: PresetParameters) -> Tuple[bool, PresetParameters, str]:
    """
    Validate preset parameters for UI compatibility
    
    Args:
        parameters: PresetParameters object to validate
        
    Returns:
        Tuple of (all_valid: bool, sanitized_parameters: PresetParameters, summary_message: str)
    """
    try:
        validator = get_parameter_validator()
        all_valid, sanitized_parameters, messages = validator.validate_preset_parameters(parameters)
        
        # Create summary message
        issues = [msg for msg in messages.values() if 'clamped' in msg.lower() or 'error' in msg.lower()]
        
        if all_valid:
            if issues:
                summary = f"✅ Preset validated with {len(issues)} parameter adjustments"
            else:
                summary = "✅ Preset validated successfully"
        else:
            summary = f"⚠️ Preset validation completed with {len(issues)} issues"
        
        return all_valid, sanitized_parameters, summary
        
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, parameters, error_msg


def validate_ui_parameters(**kwargs) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validate individual UI parameter values
    
    Args:
        **kwargs: Parameter name-value pairs
        
    Returns:
        Tuple of (all_valid: bool, sanitized_values: Dict, summary_message: str)
    """
    try:
        validator = get_parameter_validator()
        sanitized_values = {}
        issues = []
        
        for param_name, value in kwargs.items():
            is_valid, sanitized_value, message = validator.validate_parameter(param_name, value)
            sanitized_values[param_name] = sanitized_value
            
            if not is_valid or 'clamped' in message.lower():
                issues.append(f"{param_name}: {message}")
        
        all_valid = len(issues) == 0
        
        if all_valid:
            summary = "✅ All parameters validated successfully"
        else:
            summary = f"⚠️ {len(issues)} parameter issues: {'; '.join(issues[:3])}"
            if len(issues) > 3:
                summary += f" and {len(issues) - 3} more"
        
        return all_valid, sanitized_values, summary
        
    except Exception as e:
        error_msg = f"Parameter validation error: {str(e)}"
        print(f"❌ {error_msg}")
        return False, kwargs, error_msg 