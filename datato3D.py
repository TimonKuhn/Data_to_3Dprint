import open3d as o3d #for Mesh operations
import numpy as np #for numerical operations
from shapely.geometry import Polygon #for geometric operations
import trimesh #for mesh operations
import rasterio
from rasterio.mask import mask
import rasterio.merge
import requests
import pandas as pd

"""interesting options"""

def read_lv95_do_rectangular_bbox(lv95_file):
    # Load LV95 coordinates from a .txt file
    lv95_coordinates = np.loadtxt(lv95_file, delimiter=',')

    # Split the coordinates into two sets
    set1 = lv95_coordinates[:1]
    set2 = lv95_coordinates[1:]

    # Define the coordinates of the rectangular polygon
    x_min = set1[0][0]
    y_min = set1[0][1]
    x_max = set2[0][0]
    y_max = set2[0][1]

    # Create the rectangular polygon
    polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    return polygon
    # Use the polygon to cut the raster with GDAL
    # gdal.CutRaster(raster_file, output_file, polygon)

    # Use the polygon to cut the mesh with Open3D
    # o3d.CutMesh(mesh, polygon)

def buffer_polygon(polygon, buffer_percentage=0.2):
    # Calculate buffer_pertentage of the polygon's length
    buffer_distance = buffer_percentage * polygon.length

    # Buffer the polygon
    buffered_polygon = polygon.buffer(buffer_distance, cap_style=3)
    return buffered_polygon

def download_raster_files(csv_path, output_directory):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create an empty list to store the downloaded raster files
    raster_files = []
    # Iterate over each row in the CSV file
    for index, url in df.iterrows():
        # Download the TIFF file
        response = requests.get(url.iloc[0])
        filename = f"{output_directory}/raster_{index}.tif"
        with open(filename, "wb") as file:
            file.write(response.content)
        
        # Add the downloaded raster file to the list
        raster_files.append(filename)
    
    return raster_files

def stitch_raster_files(raster_files, output_filename): 
    # Open the first raster file to get the spatial extent
    with rasterio.open(raster_files[0]) as src:
        profile = src.profile

    # Merge the raster files into a single raster file
    merged_raster, out_transform = rasterio.merge.merge(raster_files)

    # Update the profile with the correct spatial extent
    profile.update(transform=out_transform, width=merged_raster.shape[2], height=merged_raster.shape[1])

    # Save the merged raster as a TIFF file
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(merged_raster)
    

def read_raster_dem_cut_to_bbox(dem_path, polygon):
    with rasterio.open(dem_path) as src:
        geo = [polygon.__geo_interface__]  # Convert the bbox to GeoJSON
        out_image, out_transform = mask(src, geo, crop=True)
    return out_image, out_transform, src

def slice_out_image(out_image, smallest_value=0, largest_value=10000):
    """cut/clip the dem at specific values to have valeys filled up as lakes or leave buildings at specific height"""
    out_image = out_image[0,:-1,1:]  # slice it to needed size
    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            if out_image[i][j] < smallest_value:
                out_image[i][j] = smallest_value # set negative values to 0 by default
            elif out_image[i][j] > largest_value:
                out_image[i][j] = largest_value # set values above 10000 to 10000
            else:
                pass
    return out_image

def dem_to_mesh(out_image, out_transform, KDTreeSearchParamHybrid_radius=0.1, KDTreeSearchParamHybrid_max_nn=30, create_from_point_cloud_poisson_depth=8, z_scale=1):
    """
    KDTreeSearchParamHybrid_radius: This parameter is used when estimating the normals of the point cloud. 
    It specifies the radius within which to search for neighboring points. 
    The KDTree data structure is used to efficiently find the nearest neighbors of each point. 
    The radius parameter determines how far from each point the algorithm will search for neighbors. 
    A larger radius will result in more points being considered as neighbors, which can smooth out the estimated normals but may also blur sharp features.
    
    KDTreeSearchParamHybrid_max_nn: This parameter is also used when estimating the normals.
    It specifies the maximum number of nearest neighbors to consider within the search radius.
    If there are more points within the search radius than this number, only the closest ones will be considered.
    This can be used to limit the influence of distant points when estimating the normals.
    
    create_from_point_cloud_poisson_depth: This parameter is used when creating the mesh from the point cloud using the Poisson surface reconstruction method.
    The depth parameter controls the depth of the octree that is used in the reconstruction. 
    A higher depth will result in a more detailed surface but will also require more computation.
    The optimal value depends on the complexity of the surface and the level of detail required.
    """


    # Get the elements of the out_transform
    a, b, c, d, e, f,g,h,i = out_transform

    # Create a 4x4 transformation matrix
    affine_transformation = np.array([
        [a, b, 0, c],
        [d, e, 0, f],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Get the x and y coordinates
    x, y = np.meshgrid(np.arange(out_image.shape[1]), np.arange(out_image.shape[0]))

    # Stack the coordinates together
    points = np.stack((x, y, out_image*z_scale), axis=-1)

    # Flatten the array
    points = points.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Transform the point cloud
    pcd.transform(affine_transformation)

    # Estimate normals
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=KDTreeSearchParamHybrid_radius, max_nn=KDTreeSearchParamHybrid_max_nn))

    # Create the triangle mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=create_from_point_cloud_poisson_depth)
    return mesh, pcd

def cutting_mesh_with_bbox(mesh, bbox_product, cutting_height_min=100 ,cutting_height_max=10000):
    # Convert the Polygon to a list of points
    bounding_points = np.array(bbox_product.exterior.coords)

    # Append a zero to make the points 3D
    bounding_points = np.hstack((bounding_points, np.zeros((bounding_points.shape[0], 1))))

    # double them for upper and lower z limits
    bounding_points2 = bounding_points.copy()
    bounding_points[:, 2] += cutting_height_min
    bounding_points2[:, 2] += cutting_height_max
    bounding_points=np.vstack((bounding_points, bounding_points2))


    # Create an AxisAlignedBoundingBox object
    min_bound = np.append(bounding_points.min(axis=0), 0)[:3]
    max_bound = np.append(bounding_points.max(axis=0), 0)[:3]
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    # Crop the mesh
    mesh_cropped = mesh.crop(bbox)
    return mesh_cropped

def surface_to_volume(mesh_surface, thickness=0):

    """
    Thickness from lowest point of DEM to bottom of the output stl,
    standard value = 0 = is calculated to be at sea level
    """
    # Convert legacy mesh to tensor-based mesh
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_surface)

    # Get the highest and lowest vertices
    highest_vertex = np.max(mesh_surface.vertices, axis=0)
    lowest_vertex = np.min(mesh_surface.vertices, axis=0)

    if thickness == 0: 
        thickness = lowest_vertex[2]

    # Calculate the distance between the highest and lowest vertices
    vertical_distance = highest_vertex[2] - lowest_vertex[2]

    # Define the extrusion distance
    extrusion_distance = - thickness - 10 - vertical_distance

    # Extrude the mesh along the z-axis
    extruded_mesh = tensor_mesh.extrude_linear([0, 0, extrusion_distance])

    # Convert tensor-based mesh back to legacy format for visualization
    extruded_mesh_legacy = extruded_mesh.to_legacy()
    extruded_mesh_legacy.compute_triangle_normals()

    # Correct the orientation of the triangle normals
    extruded_mesh_legacy.orient_triangles()

    # Get the vertices of the mesh
    vertices = np.asarray(extruded_mesh_legacy.vertices)

    # Find the minimum z-coordinate value
    min_z = np.min(vertices[:, 2])

    # Find the indices of the vertices with the minimum z-coordinate value
    bottom_layer_indices = np.where(vertices[:, 2] <= min_z+vertical_distance*1.0001)[0]

    # Convert bottom_layer_indices to integer type
    bottom_layer_indices = bottom_layer_indices.astype(int)

    # Set the z-coordinate of the bottom layer vertices to 0
    vertices[bottom_layer_indices, 2] = lowest_vertex[2] - thickness

    # Update the vertices in the mesh
    extruded_mesh_legacy.vertices = o3d.utility.Vector3dVector(vertices)

    # repair Mesh, but it can cause the kernel to die
    # extruded_mesh_legacy.merge_close_vertices(0.01)
    # extruded_mesh_legacy.remove_degenerate_triangles()
    # extruded_mesh_legacy.remove_duplicated_triangles()
    # extruded_mesh_legacy.remove_unreferenced_vertices()

    return extruded_mesh_legacy

############################################################################################################
def shrink_bbox(bbox_product, shrink_factor=0):
    minx, miny, maxx, maxy = bbox_product.bounds
    dx = (maxx - minx) * shrink_factor / 2
    dy = (maxy - miny) * shrink_factor / 2
    bbox_product = Polygon([(minx + dx, miny + dy), (minx + dx, maxy - dy), (maxx - dx, maxy - dy), (maxx - dx, miny + dy)])
    return bbox_product

def cutting_mesh_through_middle_of_faces(mesh, bbox_product):
    # Check if the mesh is a scene
    if isinstance(mesh, trimesh.Scene):
        # Convert the scene to a single trimesh
        mesh = mesh.dump(concatenate=True)

    ### getting info
    low_y = bbox_product.bounds[1]
    max_y = bbox_product.bounds[3]
    low_x = bbox_product.bounds[0]
    max_x = bbox_product.bounds[2]

    # Define the plane to slice the mesh: point on plane and normal vector

    plane_ypos_origin = [0, low_y, 0]
    plane_ypos_normal = [0, 1, 0]  # ypos-axis


    plane_yneg_origin = [0, max_y, 0]
    plane_yneg_normal = [0, -1, 0]  # yneg-axis


    plane_xpos_origin = [low_x, 0, 0]
    plane_xpos_normal = [1, 0, 0]  # y-axis


    plane_xneg_origin = [max_x, 0, 0]
    plane_xneg_normal = [-1, 0, 0]  # yneg-axis

    # Slice the mesh
    sliced_mesh = mesh.slice_plane(plane_ypos_origin, plane_ypos_normal).slice_plane(plane_yneg_origin, plane_yneg_normal).slice_plane(plane_xpos_origin, plane_xpos_normal).slice_plane(plane_xneg_origin, plane_xneg_normal)

    # The result is a list of polygons that form the cross section
    # We can convert these polygons to a new mesh for visualization
    cross_section_mesh = trimesh.Trimesh(vertices=sliced_mesh.vertices, faces=sliced_mesh.faces)

    # Visualize the sliced mesh
    cross_section_mesh.fill_holes()

    # cuts through the middle of the faces, so the roofs are sliced as expected, but the walls often wont be printed as they do not count as volume as they are not closed on the other side.
    return cross_section_mesh

############################################################################################################
def filter_faces_based_on_normal(mesh, direction):
    # Calculate the dot product of the face normals and the direction
    dot_product = mesh.face_normals.dot(direction)
    # Filter the faces where the dot product is positive
    mesh.update_faces(dot_product > 0)
    return mesh

def process_missing_side_walls(vertices, bbox_product, combined_mesh, side_name, tolerance = 0.01): 


    # Get the corners of the bbox
    bbox_corners = np.array(bbox_product.exterior.coords)

    # Define the minimum and maximum coordinates of the bounding box
    min_x = bbox_corners[0, 0]
    max_x = bbox_corners[2, 0]
    min_y = bbox_corners[0, 1]
    max_y = bbox_corners[2, 1]

    # Define the direction for each side
    directions = {
        'east': np.array([-1, 0, 0]),
        'west': np.array([1, 0, 0]),
        'north': np.array([0, 1, 0]),
        'south': np.array([0, -1, 0])
    }

    if side_name == 'east':
        side_vertices = vertices[np.abs(vertices[:, 0] - min_x) < tolerance]
    elif side_name == 'west':
        side_vertices = vertices[np.abs(vertices[:, 0] - max_x) < tolerance]
    elif side_name == 'north':
        side_vertices = vertices[np.abs(vertices[:, 1] - max_y) < tolerance]
    elif side_name == 'south':
        side_vertices = vertices[np.abs(vertices[:, 1] - min_y) < tolerance]
    else:
        return combined_mesh

    if side_vertices.size > 0:
        lowest_z = np.min(side_vertices[:, 2])
        additional_vertices = np.array([[i[0], i[1], lowest_z-1] for i in side_vertices])
        side_vertices = np.concatenate((side_vertices, additional_vertices))
        side_face = trimesh.convex.convex_hull(side_vertices, qhull_options='QbB Pp Qt') 
        side_normal = side_face.face_normals[0]
        side_face = filter_faces_based_on_normal(side_face, directions[side_name])
        combined_mesh = trimesh.util.concatenate([combined_mesh, side_face])
    
    return combined_mesh



    