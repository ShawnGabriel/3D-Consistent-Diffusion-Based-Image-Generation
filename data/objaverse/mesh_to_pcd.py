import open3d as o3d
import os

def mesh_to_pointcloud(mesh_path, num_points=50000):
    """
    Convert a mesh file into a point cloud by uniform surface sampling.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {mesh_path}")
    
    mesh.compute_vertex_normals()
    
    # Uniformly sample points on the mesh surface
    pcd = mesh.sample_points_uniformly(
        number_of_points=num_points
    )
    
    return pcd


def main():
    mesh_dir = "data/objaverse/raw_meshes"
    out_dir = "data/objaverse/pointclouds"
    
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in os.listdir(mesh_dir):
        if not fname.lower().endswith((".glb", ".gltf", ".obj", ".ply")):
            continue
        
        mesh_path = os.path.join(mesh_dir, fname)
        print(f"Processing {mesh_path}")
        
        pcd = mesh_to_pointcloud(mesh_path)
        
        out_path = os.path.join(
            out_dir,
            os.path.splitext(fname)[0] + ".ply"
        )
        
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"Saved point cloud to {out_path}")
        
        # Uncomment to test for validity
        # pcd = o3d.io.read_point_cloud("data/objaverse/pointclouds/chair.ply")
        # o3d.visualization.draw_geometries([pcd])
        
if __name__ == "__main__":
    main()