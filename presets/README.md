# Hi3DGen Presets

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
