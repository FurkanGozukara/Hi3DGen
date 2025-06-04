# Auto-Save Feature Documentation

## Overview
The Hi3DGen SECourses app now includes automatic saving functionality that saves generated 3D models and normal maps to organized folders.

## Features

### 🗂️ Organized Storage
- All generations are saved in the `outputs/` folder
- Each generation gets its own numbered folder (0001, 0002, 0003, etc.)
- Automatic folder numbering ensures no conflicts

### 📁 Open Outputs Folder Button
- Click the "📁 Open Outputs Folder" button to quickly access your saved files
- Works on Windows, macOS, and Linux
- Opens the folder in your system's default file manager

### ✅ Auto-Save Checkboxes
Four format options are available, all enabled by default:
- **Auto-save OBJ** - Wavefront OBJ format
- **Auto-save GLB** - Binary glTF format (recommended for web/AR/VR)
- **Auto-save PLY** - Stanford Triangle format
- **Auto-save STL** - Stereolithography format (3D printing)

### 📊 What Gets Saved
For each generation, the following files are created:
- `XXXX.obj` - OBJ format (if enabled)
- `XXXX.glb` - GLB format (if enabled)  
- `XXXX.ply` - PLY format (if enabled)
- `XXXX.stl` - STL format (if enabled)
- `XXXX_normal.png` - Generated normal map
- `XXXX_info.txt` - Generation information and settings

Where `XXXX` is the folder number (e.g., 0001, 0002, etc.)

## Folder Structure Example
```
outputs/
├── 0001/
│   ├── 0001.obj
│   ├── 0001.glb
│   ├── 0001.ply
│   ├── 0001.stl
│   ├── 0001_normal.png
│   └── 0001_info.txt
├── 0002/
│   ├── 0002.obj
│   ├── 0002.glb
│   ├── 0002_normal.png
│   └── 0002_info.txt
└── 0003/
    └── ...
```

## Usage
1. **Configure Auto-Save**: Use the checkboxes to select which formats you want to save automatically
2. **Generate**: Click "Generate Shape" as usual
3. **Auto-Save**: Files are automatically saved to the outputs folder during generation
4. **Access Files**: Click "📁 Open Outputs Folder" to view your saved files
5. **Export**: Use the regular export functionality for additional format conversions

## Technical Details
- Auto-save runs after successful 3D generation
- No performance impact on generation process
- Automatic error handling and logging
- Cross-platform folder opening support
- Incremental folder numbering system

## Benefits
- **Never lose your work** - All generations are automatically preserved
- **Organized storage** - Each generation in its own numbered folder
- **Multiple formats** - Save in all the formats you need simultaneously
- **Easy access** - One-click folder opening
- **Metadata preservation** - Generation settings saved with each model

## Troubleshooting
- If auto-save fails, check console output for error messages
- Ensure you have write permissions in the application directory
- The outputs folder is created automatically if it doesn't exist
- If folder opening doesn't work, manually navigate to the `outputs/` folder in your file manager 