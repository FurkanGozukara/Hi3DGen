import os
import shutil
import platform
import subprocess
import datetime
import trimesh
from typing import Optional, List, Dict, Any
import tempfile

def get_outputs_folder() -> str:
    """Get the outputs folder path, create if it doesn't exist"""
    # Get the directory where the main app is located
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_folder = os.path.join(current_dir, 'outputs')
    os.makedirs(outputs_folder, exist_ok=True)
    return outputs_folder

def get_next_folder_number(outputs_folder: str) -> str:
    """Get the next available folder number (0001, 0002, etc.)"""
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder, exist_ok=True)
        return "0001"
    
    # Get all existing numbered folders
    existing_folders = []
    for item in os.listdir(outputs_folder):
        item_path = os.path.join(outputs_folder, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
            existing_folders.append(int(item))
    
    if not existing_folders:
        return "0001"
    
    # Get the next number
    next_number = max(existing_folders) + 1
    return f"{next_number:04d}"

def create_generation_folder(outputs_folder: str) -> str:
    """Create a new numbered folder for the generation"""
    folder_number = get_next_folder_number(outputs_folder)
    folder_path = os.path.join(outputs_folder, folder_number)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, folder_number

def save_mesh_in_formats(mesh_path: str, folder_path: str, folder_number: str, 
                        enabled_formats: Dict[str, bool], source_normal_image=None) -> Dict[str, str]:
    """Save mesh in all enabled formats"""
    if not mesh_path or not os.path.exists(mesh_path):
        print(f"save_mesh_in_formats: Invalid mesh path: {mesh_path}")
        return {}
    
    saved_files = {}
    
    try:
        # Load the mesh once
        mesh = trimesh.load_mesh(mesh_path)
        
        for format_name, is_enabled in enabled_formats.items():
            if not is_enabled:
                continue
                
            try:
                output_filename = f"{folder_number}.{format_name.lower()}"
                output_path = os.path.join(folder_path, output_filename)
                
                print(f"Saving {format_name.upper()} format to: {output_path}")
                
                # Special handling for GLB (copy if source is already GLB)
                if format_name.lower() == "glb" and mesh_path.lower().endswith(".glb"):
                    shutil.copy2(mesh_path, output_path)
                else:
                    mesh.export(output_path, file_type=format_name.lower())
                
                # Verify file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    saved_files[format_name] = output_path
                    print(f"✓ Successfully saved {format_name.upper()}: {output_path} ({os.path.getsize(output_path)} bytes)")
                else:
                    print(f"✗ Failed to create valid {format_name.upper()} file")
                    
            except Exception as e:
                print(f"Error saving {format_name.upper()} format: {e}")
                continue
        
        # Save the normal image if provided
        if source_normal_image is not None:
            try:
                normal_filename = f"{folder_number}_normal.png"
                normal_path = os.path.join(folder_path, normal_filename)
                source_normal_image.save(normal_path)
                saved_files['normal'] = normal_path
                print(f"✓ Successfully saved normal image: {normal_path}")
            except Exception as e:
                print(f"Error saving normal image: {e}")
        
        # Create a generation info file
        try:
            info_filename = f"{folder_number}_info.txt"
            info_path = os.path.join(folder_path, info_filename)
            with open(info_path, 'w') as f:
                f.write(f"Generation Information\n")
                f.write(f"=====================\n")
                f.write(f"Folder: {folder_number}\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source mesh: {os.path.basename(mesh_path)}\n")
                f.write(f"Enabled formats: {', '.join([k for k, v in enabled_formats.items() if v])}\n")
                f.write(f"Saved files:\n")
                for format_name, file_path in saved_files.items():
                    if format_name != 'normal':
                        f.write(f"  - {format_name.upper()}: {os.path.basename(file_path)}\n")
                if 'normal' in saved_files:
                    f.write(f"  - Normal image: {os.path.basename(saved_files['normal'])}\n")
            saved_files['info'] = info_path
            print(f"✓ Successfully saved generation info: {info_path}")
        except Exception as e:
            print(f"Error saving generation info: {e}")
            
    except Exception as e:
        print(f"Error loading mesh for format conversion: {e}")
        return {}
    
    return saved_files

def auto_save_generation(mesh_path: str, normal_image=None, enabled_formats: Dict[str, bool] = None) -> Optional[Dict[str, Any]]:
    """Main auto-save function"""
    if enabled_formats is None:
        enabled_formats = {"obj": True, "glb": True, "ply": True, "stl": True}
    
    # Check if any format is enabled
    if not any(enabled_formats.values()):
        print("Auto-save: No formats enabled, skipping save")
        return None
    
    try:
        outputs_folder = get_outputs_folder()
        folder_path, folder_number = create_generation_folder(outputs_folder)
        
        print(f"Auto-save: Created generation folder: {folder_path}")
        
        saved_files = save_mesh_in_formats(mesh_path, folder_path, folder_number, enabled_formats, normal_image)
        
        result = {
            'folder_path': folder_path,
            'folder_number': folder_number,
            'saved_files': saved_files,
            'outputs_folder': outputs_folder
        }
        
        print(f"Auto-save completed: {len(saved_files)} files saved in folder {folder_number}")
        return result
        
    except Exception as e:
        print(f"Auto-save error: {e}")
        import traceback
        traceback.print_exc()
        return None

def open_outputs_folder():
    """Open the outputs folder in the system file manager"""
    outputs_folder = get_outputs_folder()
    
    try:
        system = platform.system()
        if system == "Windows":
            # Windows
            os.startfile(outputs_folder)
        elif system == "Darwin":
            # macOS
            subprocess.Popen(["open", outputs_folder])
        else:
            # Linux and other Unix-like systems
            subprocess.Popen(["xdg-open", outputs_folder])
        
        print(f"Opened outputs folder: {outputs_folder}")
        return True
        
    except Exception as e:
        print(f"Error opening outputs folder: {e}")
        # Fallback: print the path
        print(f"Please manually open: {outputs_folder}")
        return False

def get_generation_summary(outputs_folder: str = None) -> Dict[str, Any]:
    """Get a summary of all generations"""
    if outputs_folder is None:
        outputs_folder = get_outputs_folder()
    
    summary = {
        'total_generations': 0,
        'folders': [],
        'outputs_folder': outputs_folder
    }
    
    try:
        if not os.path.exists(outputs_folder):
            return summary
        
        for item in os.listdir(outputs_folder):
            item_path = os.path.join(outputs_folder, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                folder_info = {
                    'number': item,
                    'path': item_path,
                    'files': []
                }
                
                # List files in the folder
                try:
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if os.path.isfile(file_path):
                            folder_info['files'].append({
                                'name': file,
                                'path': file_path,
                                'size': os.path.getsize(file_path)
                            })
                except Exception as e:
                    print(f"Error reading folder {item}: {e}")
                
                summary['folders'].append(folder_info)
                summary['total_generations'] += 1
        
        # Sort folders by number
        summary['folders'].sort(key=lambda x: x['number'])
        
    except Exception as e:
        print(f"Error getting generation summary: {e}")
    
    return summary 