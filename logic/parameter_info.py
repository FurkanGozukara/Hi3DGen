"""
Parameter Information and Descriptions for Hi3DGen Interface
This module contains detailed descriptions of all user-controllable parameters.
"""

PARAMETER_INFO = {
    "image_input": {
        "name": "Image Input",
        "type": "Image Upload",
        "description": "The source image for 3D generation",
        "details": """
        **Purpose**: This is your input image that gets processed through normal map generation and then converted to a 3D model.
        
        **Format**: RGBA mode, PIL format
        
        **Examples**: 
        - Upload a photo of an object, character, or scene you want to convert to 3D
        - Works best with clear, well-lit subjects
        - Supports transparent backgrounds (RGBA)
        
        **Impact**: Quality and characteristics of input directly affect final 3D model quality.
        """,
        "tips": "Use high-resolution images with good lighting for best results."
    },
    
    "seed": {
        "name": "Seed",
        "type": "Slider",
        "range": "-1 to 2,147,483,647",
        "default": "0",
        "description": "Controls randomness in the generation process",
        "details": """
        **Purpose**: Determines the random number generation seed for reproducible results.
        
        **Settings**:
        - **-1**: Uses random seed each time (different results every generation)
        - **Specific number**: Produces reproducible results
        
        **Examples**:
        - Seed = 12345: Always generates the same 3D model from same input
        - Seed = -1: Each generation will be slightly different
        
        **Use case**: Set specific seed when you find a good result and want to reproduce it.
        """,
        "tips": "Use -1 for experimentation, set specific values to reproduce good results."
    },
    
    "ss_guidance_strength": {
        "name": "Sparse Structure Guidance Strength",
        "type": "Slider", 
        "range": "0.0 to 10.0",
        "default": "3.0",
        "step": "0.1",
        "stage": "Stage 1: Sparse Structure Generation",
        "description": "Controls how closely the AI follows the input during initial 3D structure creation",
        "details": """
        **Purpose**: Determines how strictly the AI adheres to your input image during the first generation stage.
        
        **Settings**:
        - **Low values (0.5-2.0)**: More creative/abstract interpretation, less faithful to input
        - **Medium values (2.5-4.0)**: Balanced approach (recommended for most cases)
        - **High values (5.0-10.0)**: Very strict adherence to input, may be over-constrained
        
        **Example**: For a portrait photo, value of 3.0 gives good balance between accuracy and natural 3D form.
        """,
        "tips": "Start with 3.0 and adjust based on whether you want more creativity (lower) or accuracy (higher)."
    },
    
    "ss_sampling_steps": {
        "name": "Sparse Structure Sampling Steps",
        "type": "Slider",
        "range": "1 to 50", 
        "default": "50",
        "stage": "Stage 1: Sparse Structure Generation",
        "description": "Number of refinement iterations for initial 3D structure",
        "details": """
        **Purpose**: Controls how many refinement steps are used to create the initial 3D structure.
        
        **Settings**:
        - **Low steps (10-20)**: Faster but less detailed initial structure
        - **Medium steps (30-40)**: Good balance of speed and quality  
        - **High steps (45-50)**: Best quality but slower processing
        
        **Time impact**: Each step adds processing time. 50 steps might take 2-3x longer than 20 steps.
        """,
        "tips": "Use 50 for best quality, reduce to 30-40 if you need faster processing."
    },
    
    "slat_guidance_strength": {
        "name": "Structured Latent Guidance Strength", 
        "type": "Slider",
        "range": "0.0 to 10.0",
        "default": "3.0",
        "step": "0.1", 
        "stage": "Stage 2: Structured Latent Generation",
        "description": "Controls guidance during final 3D mesh refinement",
        "details": """
        **Purpose**: Similar to Stage 1 guidance but affects final mesh detail and accuracy.
        
        **Settings**: Same as Sparse Structure Guidance Strength
        - **Low values**: More creative interpretation in final mesh
        - **Medium values**: Balanced approach
        - **High values**: Strict adherence to structure
        
        **Recommendation**: Keep similar to Stage 1 guidance for consistency.
        """,
        "tips": "Usually best to match this with your Stage 1 guidance strength."
    },
    
    "slat_sampling_steps": {
        "name": "Structured Latent Sampling Steps",
        "type": "Slider", 
        "range": "1 to 50",
        "default": "6",
        "stage": "Stage 2: Structured Latent Generation", 
        "description": "Final refinement iterations for the 3D mesh",
        "details": """
        **Purpose**: Number of refinement steps for final mesh generation.
        
        **Settings**: Lower default (6) because this stage is more computationally expensive
        - **Low steps (4-6)**: Faster processing, good for most cases
        - **High steps (8-12)**: Better quality but significantly longer processing
        
        **Performance note**: Each step significantly impacts generation time.
        """,
        "tips": "6 is usually sufficient; only increase if you need maximum quality and don't mind longer processing."
    },
    
    "poly_count_pcnt": {
        "name": "Polygon Count Percentage",
        "type": "Slider",
        "range": "0.05 to 1.0 (5% to 100%)", 
        "default": "0.5 (50%)",
        "step": "0.05",
        "stage": "Mesh Simplification",
        "description": "Reduces mesh complexity for performance and file size",
        "details": """
        **Purpose**: Controls how many polygons are retained after mesh simplification.
        
        **Settings**:
        - **0.1 (10%)**: Very simplified mesh, small file, lower detail
        - **0.5 (50%)**: Half the polygons, good balance for most uses  
        - **0.9 (90%)**: High detail, larger files, may slow down viewing
        
        **Example**: 
        - Original mesh: 100,000 triangles
        - At 0.5: Result has ~50,000 triangles
        - At 0.1: Result has ~10,000 triangles
        """,
        "tips": "Use 0.5 for general use, 0.7-0.9 for high detail, 0.2-0.4 for web/mobile use."
    },
    
    "xatlas_max_cost": {
        "name": "Max Chart Cost",
        "type": "Slider",
        "range": "1.0 to 10.0",
        "default": "8.0", 
        "step": "0.1",
        "stage": "UV Unwrapping (xatlas)",
        "description": "Controls texture mapping quality vs. efficiency",
        "details": """
        **Purpose**: Determines how much texture distortion is acceptable for UV mapping.
        
        **Settings**:
        - **Low values (1.0-3.0)**: More UV seams, less stretching, better texture quality
        - **High values (7.0-10.0)**: Fewer seams, more stretching, may distort textures
        
        **Example**: For detailed textures, use lower values; for simple colors, higher is fine.
        """,
        "tips": "Use 3-5 for detailed textures, 7-9 for simple materials or solid colors."
    },
    
    "xatlas_normal_seam_weight": {
        "name": "Normal Seam Weight", 
        "type": "Slider",
        "range": "0.1 to 5.0",
        "default": "1.0",
        "step": "0.1",
        "stage": "UV Unwrapping (xatlas)",
        "description": "How much surface normal changes affect UV seam placement", 
        "details": """
        **Purpose**: Controls how surface angle changes influence where UV seams are placed.
        
        **Settings**:
        - **Low values (0.1-0.5)**: Fewer seams based on surface angles
        - **High values (2.0-5.0)**: More seams at surface angle changes
        
        **Use case**: Lower for smooth objects, higher for complex geometry.
        """,
        "tips": "Use 0.5-1.0 for organic shapes, 1.5-3.0 for mechanical objects."
    },
    
    "xatlas_resolution": {
        "name": "UV Atlas Resolution",
        "type": "Slider", 
        "range": "256 to 4096",
        "default": "1024",
        "step": "256",
        "stage": "UV Unwrapping (xatlas)",
        "description": "Texture resolution for UV mapping",
        "details": """
        **Purpose**: Sets the resolution of the UV texture atlas.
        
        **Settings**:
        - **256**: Low-res textures, good for simple models
        - **1024**: Standard resolution, good for most uses
        - **2048/4096**: High-res textures, large file sizes
        
        **File size impact**: 4096 creates much larger texture files than 1024.
        """,
        "tips": "1024 is good for most uses, 2048+ for high-detail texturing, 512 for web optimization."
    },
    
    "xatlas_padding": {
        "name": "UV Chart Padding",
        "type": "Slider",
        "range": "0 to 16 pixels", 
        "default": "2",
        "step": "1",
        "stage": "UV Unwrapping (xatlas)",
        "description": "Space between UV islands in texture atlas",
        "details": """
        **Purpose**: Sets padding between UV islands to prevent texture bleeding.
        
        **Settings**:
        - **0-1 pixels**: Minimal padding, risk of bleeding
        - **2-4 pixels**: Safe padding for most uses
        - **8+ pixels**: Extra safe, wastes UV space
        
        **Recommendation**: 2-4 pixels is usually sufficient.
        """,
        "tips": "Use 2 for most cases, increase to 4-6 if you see texture bleeding artifacts."
    },
    
    "normal_map_resolution": {
        "name": "Normal Map Resolution",
        "type": "Slider",
        "range": "256 to 1024", 
        "default": "768",
        "step": "128",
        "stage": "Normal Map Generation",
        "description": "Resolution of generated normal map",
        "details": """
        **Purpose**: Controls the resolution of the normal map generated from your input image.
        
        **Settings**:
        - **256-512**: Lower detail, faster processing  
        - **768**: Good balance (default)
        - **1024**: Maximum detail, slower processing
        
        **Performance**: Higher resolutions take more processing time and memory.
        """,
        "tips": "768 works well for most images, use 1024 for very detailed inputs."
    },
    
    "normal_match_input_resolution": {
        "name": "Match Input Resolution",
        "type": "Checkbox",
        "default": "True (Checked)",
        "stage": "Normal Map Generation", 
        "description": "Automatically adjust normal map resolution based on input image",
        "details": """
        **Purpose**: Intelligently sets normal map resolution based on your input image size.
        
        **Settings**:
        - **Checked**: If input is 512px, uses 512px instead of slider value
        - **Unchecked**: Always uses slider value regardless of input size
        
        **Recommendation**: Keep checked for optimal quality/performance balance.
        """,
        "tips": "Leave checked unless you specifically want to override the automatic sizing."
    },
    
    "auto_save_formats": {
        "name": "Auto-Save Format Checkboxes", 
        "type": "Checkboxes",
        "default": "All checked (True)",
        "stage": "Auto-Save Settings",
        "description": "Automatically saves generated models in selected formats",
        "details": """
        **Purpose**: Automatically save your generated 3D models in multiple formats.
        
        **Formats**:
        - **OBJ**: Wavefront OBJ format (widely compatible, supports textures)
        - **GLB**: Binary glTF format (modern, compact, web-friendly)  
        - **PLY**: Stanford Triangle format (good for 3D printing, scientific use)
        - **STL**: Stereolithography format (3D printing standard, no textures)
        
        **File locations**: Saved to organized folders in outputs directory.
        """,
        "tips": "Keep all enabled unless you specifically don't need certain formats."
    },
    
    "export_format": {
        "name": "Export Format",
        "type": "Dropdown",
        "options": "obj, glb, ply, stl", 
        "default": "glb",
        "stage": "Export Settings",
        "description": "Choose format for manual download",
        "details": """
        **Purpose**: Select the format for manual download of your generated model.
        
        **What each format is good for**:
        - **OBJ**: Universal compatibility, texture support, easy to edit
        - **GLB**: Modern web applications, VR/AR, game engines
        - **PLY**: Scientific applications, 3D scanning, point clouds
        - **STL**: 3D printing (no texture/color information)
        """,
        "tips": "Use GLB for web/games, OBJ for 3D software, STL for 3D printing."
    }
}

RECOMMENDED_SETTINGS = {
    "high_quality": {
        "name": "High Quality Results",
        "description": "Best quality output, slower processing",
        "settings": {
            "Guidance Strengths": "3.0-4.0",
            "Sampling Steps": "SS=50, SLAT=8-10", 
            "Polygon Count": "0.7-0.9",
            "UV Resolution": "2048"
        }
    },
    
    "fast_processing": {
        "name": "Fast Processing", 
        "description": "Balanced quality and speed",
        "settings": {
            "Guidance Strengths": "2.5-3.0",
            "Sampling Steps": "SS=30, SLAT=4-6",
            "Polygon Count": "0.3-0.5", 
            "UV Resolution": "1024"
        }
    },
    
    "3d_printing": {
        "name": "3D Printing Optimized",
        "description": "Optimized for 3D printing applications", 
        "settings": {
            "Enable STL auto-save": "True",
            "Polygon Count": "0.6-0.8 (good detail but printable)",
            "Higher mesh quality settings": "Use higher sampling steps"
        }
    }
}

def get_parameter_info(param_name: str) -> dict:
    """Get information for a specific parameter."""
    return PARAMETER_INFO.get(param_name, {})

def get_all_parameters() -> dict:
    """Get all parameter information."""
    return PARAMETER_INFO

def get_recommended_settings() -> dict:
    """Get recommended setting presets."""
    return RECOMMENDED_SETTINGS

def format_parameter_info_html() -> str:
    """Format all parameter information as HTML for display in Gradio."""
    html_content = """
    <div style="width: 100%; height: 100%; min-height: 80vh; overflow-y: auto; padding: 20px; font-family: system-ui, -apple-system, sans-serif; box-sizing: border-box; margin: 0;">
        <h1 style="text-align: center; margin-bottom: 30px; font-size: 2em;">📋 Hi3DGen Parameter Guide</h1>
    """
    
    # Group parameters by stage
    stages = {}
    for param_key, param_info in PARAMETER_INFO.items():
        stage = param_info.get("stage", "General Settings")
        if stage not in stages:
            stages[stage] = []
        stages[stage].append((param_key, param_info))
    
    # Add each stage section
    for stage_name, params in stages.items():
        html_content += f"""
        <div style="margin-bottom: 30px;">
            <h2 style="border-bottom: 2px solid; padding-bottom: 10px; font-size: 1.5em; margin-top: 30px;">
                {stage_name}
            </h2>
        """
        
        for param_key, param_info in params:
            name = param_info.get("name", "Unknown Parameter")
            param_type = param_info.get("type", "Unknown")
            range_info = param_info.get("range", "")
            default = param_info.get("default", "")
            description = param_info.get("description", "")
            details = param_info.get("details", "").replace("\n", "<br>")
            # Properly handle markdown bold formatting
            import re
            details = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', details)
            tips = param_info.get("tips", "")
            
            html_content += f"""
            <div style="margin: 15px 0; padding: 20px; border: 1px solid; border-radius: 5px; width: 100%; box-sizing: border-box;">
                <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 1.3em; font-weight: 600;">
                    {name} <span style="font-size: 0.85em; opacity: 0.7; font-weight: normal;">({param_type})</span>
                </h3>
                
                {f'<p style="margin: 8px 0;"><strong>Range:</strong> {range_info}</p>' if range_info else ''}
                {f'<p style="margin: 8px 0;"><strong>Default:</strong> {default}</p>' if default else ''}
                
                <p style="margin: 12px 0; font-size: 1.05em;"><strong>Description:</strong> {description}</p>
                
                <div style="margin: 18px 0; line-height: 1.6; font-size: 1.02em;">
                    {details}
                </div>
                
                {f'<div style="border-left: 4px solid; padding: 12px; margin-top: 18px; padding-left: 20px; font-size: 1.02em;"><strong>💡 Tip:</strong> {tips}</div>' if tips else ''}
            </div>
            """
        
        html_content += "</div>"
    
    # Add recommended settings section
    html_content += """
    <div style="margin-top: 40px;">
        <h2 style="border-bottom: 2px solid; padding-bottom: 10px; font-size: 1.5em;">
            🎯 Recommended Settings for Different Use Cases
        </h2>
    """
    
    for preset_key, preset_info in RECOMMENDED_SETTINGS.items():
        name = preset_info["name"]
        description = preset_info["description"] 
        settings = preset_info["settings"]
        
        html_content += f"""
        <div style="margin: 15px 0; padding: 20px; border: 1px solid; border-radius: 5px; width: 100%; box-sizing: border-box;">
            <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 1.3em; font-weight: 600;">{name}</h3>
            <p style="font-style: italic; opacity: 0.8; font-size: 1.05em; margin-bottom: 15px;">{description}</p>
            <ul style="margin-top: 10px; line-height: 1.6; font-size: 1.02em; padding-left: 25px;">
        """
        
        for setting_name, setting_value in settings.items():
            html_content += f"<li><strong>{setting_name}:</strong> {setting_value}</li>"
        
        html_content += """
            </ul>
        </div>
        """
    
    html_content += """
        </div>
    </div>
    """
    
    return html_content 