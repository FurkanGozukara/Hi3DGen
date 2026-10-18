"""Mesh post-processing used by the single image, multi image and batch pipelines.

Quadric decimation (Open3D) followed by UV unwrapping (xatlas). Moved out of secourses_app.py unchanged so that
logic/processing_core.py can import them directly instead of reaching into the __main__ module.
"""
import traceback

import numpy as np
import open3d as o3d
import trimesh
import xatlas


def simplify_mesh_open3d(in_mesh: trimesh.Trimesh, poly_count_pcnt: float=0.5) -> trimesh.Trimesh:
    if not isinstance(in_mesh, trimesh.Trimesh):
        print('Simplify ERR: Invalid input type.')
        return in_mesh
    if not 0.0 < poly_count_pcnt < 1.0:
        print(f'Simplify skip: poly_count_pcnt {poly_count_pcnt:.2f} out of (0,1) range.')
        return in_mesh
    current_tris = len(in_mesh.faces)
    if current_tris == 0:
        return in_mesh
    target_tris = int(current_tris * poly_count_pcnt)
    target_tris = max(1, target_tris)
    if target_tris >= current_tris:
        print(f'Simplify skip: Target {target_tris} >= current {current_tris}.')
        return in_mesh
    print(f'Simplifying: {current_tris} faces -> ~{target_tris} faces ({(1.0 - poly_count_pcnt) * 100:.0f}% original).')
    o3d_m = o3d.geometry.TriangleMesh()
    o3d_m.vertices = o3d.utility.Vector3dVector(in_mesh.vertices)
    o3d_m.triangles = o3d.utility.Vector3iVector(in_mesh.faces)
    try:
        simplified_o3d_m = o3d_m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        s_verts = np.asarray(simplified_o3d_m.vertices)
        s_faces = np.asarray(simplified_o3d_m.triangles)
        if s_faces.size == 0 and current_tris > 0:
            print('Simplify WARN: Empty mesh result. Reverting.')
            return in_mesh
        return trimesh.Trimesh(vertices=s_verts, faces=s_faces, process=True)
    except Exception as e:
        print(f'Simplify ERR: Open3D failed ({e}). Reverting.')
        traceback.print_exc()
        return in_mesh

def unwrap_mesh_with_xatlas(input_mesh: trimesh.Trimesh, max_cost_param: float=8.0, normal_seam_weight_param: float=1.0, resolution_param: int=1024, padding_param: int=2) -> trimesh.Trimesh:
    import time
    import sys
    print('UV Unwrapping with xatlas: Starting process...')
    start_time = time.time()
    input_vertices_orig = input_mesh.vertices.astype(np.float32)
    input_faces_orig = input_mesh.faces.astype(np.uint32)
    vertex_normals_from_trimesh = input_mesh.vertex_normals
    input_normals_orig = np.ascontiguousarray(vertex_normals_from_trimesh, dtype=np.float32)
    print(f'  Input mesh: {input_vertices_orig.shape[0]} vertices, {input_faces_orig.shape[0]} faces.')
    face_count = input_faces_orig.shape[0]
    print(f"  Complexity level: {('Low' if face_count < 100000 else 'Medium' if face_count < 500000 else 'High')} ({face_count:,} faces)")
    sys.stdout.flush()
    print('  Phase 1/4: Setting up xatlas Atlas...')
    phase_start = time.time()
    atlas = xatlas.Atlas()
    atlas.add_mesh(input_vertices_orig, input_faces_orig, input_normals_orig)
    print(f'  Phase 1/4: Complete ({time.time() - phase_start:.2f}s)')
    sys.stdout.flush()
    print('  Phase 2/4: Configuring xatlas options...')
    phase_start = time.time()
    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = max_cost_param
    chart_options.normal_seam_weight = normal_seam_weight_param
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution_param
    pack_options.padding = padding_param
    print(f'  Phase 2/4: Complete ({time.time() - phase_start:.2f}s)')
    print(f'    -> ChartOptions: max_cost={chart_options.max_cost:.2f}, normal_seam_weight={chart_options.normal_seam_weight:.2f}')
    print(f'    -> PackOptions: resolution={pack_options.resolution}, padding={pack_options.padding}')
    sys.stdout.flush()
    print('  Phase 3/4: Running xatlas.generate() - This is the time-consuming step...')
    print('    -> Processing mesh charts and UV packing...')
    print(f'    -> This may take some time so patiently wait for {face_count:,} faces')
    print('    -> xatlas will now run without progress updates (this is normal)')
    print('    -> Please wait... processing is happening in the background')
    sys.stdout.flush()
    generation_start = time.time()
    try:
        print('    -> Starting xatlas.generate()...')
        sys.stdout.flush()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        generation_time = time.time() - generation_start
        actual_performance = face_count / generation_time if generation_time > 0 else 0
        print(f'    -> ✓ xatlas.generate() completed successfully in {generation_time:.2f}s')
        print(f'    -> Actual performance: {actual_performance:.0f} faces/second')
    except Exception as e:
        generation_time = time.time() - generation_start
        print(f'    -> ✗ xatlas.generate() failed after {generation_time:.2f}s: {e}')
        raise
    print(f'  Phase 3/4: Complete ({generation_time:.2f}s)')
    print(f'    -> xatlas generated atlas with dimensions: width={atlas.width}, height={atlas.height}')
    sys.stdout.flush()
    print('  Phase 4/4: Processing xatlas results...')
    phase_start = time.time()
    v_out_xref_data, f_out_indices, uv_coords_from_xatlas = atlas.get_mesh(0)
    print(f'    -> Retrieved mesh data: {uv_coords_from_xatlas.shape[0]} UV vertices, {f_out_indices.shape[0]} faces')
    num_new_vertices = uv_coords_from_xatlas.shape[0]
    if v_out_xref_data.shape == (num_new_vertices,):
        xref_indices = v_out_xref_data.astype(np.uint32)
        if np.any(xref_indices >= input_vertices_orig.shape[0]) or np.any(xref_indices < 0):
            raise ValueError('Invalid xref values from xatlas - out of bounds for original input vertices.')
        final_vertices_spatial = input_vertices_orig[xref_indices]
    elif v_out_xref_data.shape == (num_new_vertices, 3):
        print('    -> Warning: xatlas.get_mesh() returned 3D vertex data directly, which is unexpected for add_mesh workflow.')
        final_vertices_spatial = v_out_xref_data.astype(np.float32)
    else:
        raise ValueError(f'Unexpected shape for vertex/xref data from xatlas.get_mesh: {v_out_xref_data.shape}.')
    print(f'    -> Processed vertex mapping: {final_vertices_spatial.shape[0]} final vertices')
    final_uvs = uv_coords_from_xatlas.astype(np.float32)
    if np.any(final_uvs > 1.5):
        print('    -> UVs appear to be in pixel coordinates. Normalizing...')
        if atlas.width > 0 and atlas.height > 0:
            final_uvs /= np.array([atlas.width, atlas.height], dtype=np.float32)
        else:
            print('    -> WARNING: Atlas width/height is 0, cannot normalize pixel UVs. Using unnormalized.')
    else:
        min_uv = final_uvs.min(axis=0) if final_uvs.size > 0 else 'N/A'
        max_uv = final_uvs.max(axis=0) if final_uvs.size > 0 else 'N/A'
        print(f'    -> UVs appear to be normalized. Min: {min_uv}, Max: {max_uv}')
    print(f'    -> Creating final Trimesh object...')
    output_mesh = trimesh.Trimesh(vertices=final_vertices_spatial, faces=f_out_indices, process=False)
    if final_uvs.shape != (final_vertices_spatial.shape[0], 2):
        raise ValueError(f'Shape mismatch for final UVs before Trimesh assignment.')
    material = trimesh.visual.material.PBRMaterial(name='defaultXatlasMat')
    output_mesh.visual = trimesh.visual.TextureVisuals(uv=final_uvs, material=material)
    if hasattr(output_mesh.visual, 'uv') and output_mesh.visual.uv is not None:
        print(f'    -> Trimesh object successfully created with UVs, Shape: {output_mesh.visual.uv.shape}')
    else:
        print('    -> ERROR: Trimesh object does NOT have UVs assigned after TextureVisuals call.')
        raise RuntimeError('Failed to assign UVs to the Trimesh object.')
    print(f'  Phase 4/4: Complete ({time.time() - phase_start:.2f}s)')
    total_time = time.time() - start_time
    print(f'UV Unwrapping with xatlas: Process complete! Total time: {total_time:.2f}s')
    print(f'  -> Performance: {input_faces_orig.shape[0] / total_time:.0f} faces/second')
    sys.stdout.flush()
    return output_mesh
