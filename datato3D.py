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
    # Load LV95 coordinates from a .txt file with the following structure:
    #
    #    left_LV95, bottom_LV95
    #    right_LV95, top_LV95

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
    # Calculate buffer_percentage of the polygon's length
    buffer_distance = buffer_percentage * polygon.length

    # Buffer the polygon
    buffered_polygon = polygon.buffer(buffer_distance, cap_style=3)
    return buffered_polygon

def download_raster_files(csv_path, output_directory):
    """
    Downloads raster files from URLs specified in a CSV file and saves them in the output directory.

    Args:
        csv_path (str): The path to the CSV file containing the URLs of the raster files.
        output_directory (str): The directory where the downloaded raster files will be saved.

    Returns:
        list: A list of file paths of the downloaded raster files.
    """
    
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
    """
    Stitch multiple raster files into a single raster file.

    Args:
        raster_files (list): A list of file paths to the raster files to be stitched.
        output_filename (str): The file path of the output merged raster file.

    Returns:
        the merged raster file as a TIFF file at the specified output path.
    """
    
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
    """
    Reads a raster DEM file and cuts it to the specified bounding box defined by a polygon.

    Args:
        dem_path (str): The file path of the raster DEM file.
        polygon (Polygon): The polygon defining the bounding box to cut the raster.

    Returns:
        tuple: A tuple containing the following elements:
            - out_image (ndarray): The cut raster image.
            - out_transform (Affine): The affine transformation matrix of the cut raster.
            - src (rasterio.DatasetReader): The original raster dataset.
    """

    with rasterio.open(dem_path) as src:
        geo = [polygon.__geo_interface__]  # Convert the bbox to GeoJSON
        out_image, out_transform = mask(src, geo, crop=True)
    return out_image, out_transform, src

def slice_out_image(out_image, smallest_value=0, largest_value=10000):
    """cut/clip the dem at specific values to have valeys filled up as lakes or leave buildings at specific height"""
    out_image = out_image[0,:-1,1:]  # slice it to needed size as there is one axis to much and border pixels on the left and bottom that are incorrect
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
    Convert a digital elevation model (DEM) to a triangle mesh using Open3D library.

    Args:
        out_image (numpy.ndarray): The input DEM as a 2D numpy array.
        out_transform (tuple): The affine transformation parameters as a tuple (a, b, c, d, e, f, g, h, i).
        KDTreeSearchParamHybrid_radius (float, optional): The radius used when estimating the normals of the point cloud. 
            Defaults to 0.1.
        KDTreeSearchParamHybrid_max_nn (int, optional): The maximum number of nearest neighbors to consider within the search radius. 
            Defaults to 30.
        create_from_point_cloud_poisson_depth (int, optional): The depth of the octree used in the Poisson surface reconstruction method. 
            Defaults to 8.
        z_scale (float, optional): The scaling factor applied to the z-coordinate of the point cloud. 
            Defaults to 1.

    Returns:
        tuple: A tuple containing the triangle mesh and the point cloud.
            - mesh (open3d.geometry.TriangleMesh): The resulting triangle mesh.
            - pcd (open3d.geometry.PointCloud): The resulting point cloud.
    """
    
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
    """
    Cuts a mesh using a bounding box.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to be cropped.
        bbox_product (shapely.geometry.Polygon): The bounding box polygon representing the cutting area.
        cutting_height_min (float, optional): The minimum cutting height. Defaults to 100.
        cutting_height_max (float, optional): The maximum cutting height. Defaults to 10000.

    Returns:
        o3d.geometry.TriangleMesh: The cropped mesh.
    """
    
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
    Converts a surface mesh to a volume mesh by extruding it along the z-axis.

    Args:
        mesh_surface (open3d.geometry.TriangleMesh): The input surface mesh.
        thickness (float, optional): Thickness from the lowest point of the input mesh to the bottom of the output mesh.
            If not provided, it is calculated to be at sea level.

    Returns:
        open3d.geometry.TriangleMesh: The extruded volume mesh.

    Note:
        The input `mesh_surface` should be an instance of `open3d.geometry.TriangleMesh` representing a surface mesh.
        The output `extruded_mesh_legacy` is also an instance of `open3d.geometry.TriangleMesh` representing a volume mesh.
    """

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
    """
    Shrink the bounding box of a product by a given factor.

    Parameters:
    - bbox_product (Polygon): The original bounding box of the product.
    - shrink_factor (float): The factor by which the bounding box should be shrunk. Default is 0.

    Returns:
    - bbox_product (Polygon): The shrunk bounding box of the product.
    """
    minx, miny, maxx, maxy = bbox_product.bounds
    dx = (maxx - minx) * shrink_factor / 2
    dy = (maxy - miny) * shrink_factor / 2
    bbox_product = Polygon([(minx + dx, miny + dy), (minx + dx, maxy - dy), (maxx - dx, maxy - dy), (maxx - dx, miny + dy)])
    return bbox_product

def cutting_mesh_through_middle_of_faces(mesh, bbox_product):
    """
    Cuts a mesh through the middle of its faces using slicing planes defined by the bounding box of a product.

    Parameters:
        mesh (trimesh.Trimesh or trimesh.Scene): The input mesh to be sliced. If a trimesh.Scene is provided, it will be converted to a single trimesh.
        bbox_product (trimesh.primitives.Box): The bounding box of the product used to define the slicing planes.

    Returns:
        trimesh.Trimesh: The sliced mesh representing the cross section of the input mesh.

    Note:
        The input mesh can be either a trimesh.Trimesh object or a trimesh.Scene object. The bounding box of the product should be a trimesh.primitives.Box object.
        The output is a trimesh.Trimesh object representing the cross section of the input mesh.
    """

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
    """
    Filters the faces of a mesh based on the dot product of their normals and a given direction.

    Args:
        mesh (Mesh): The input mesh object.
        direction (ndarray): The direction vector used for filtering.

    Returns:
        Mesh: The filtered mesh object with only the faces where the dot product is positive.
    """
    # Calculate the dot product of the face normals and the direction
    dot_product = mesh.face_normals.dot(direction)
    # Filter the faces where the dot product is positive
    mesh.update_faces(dot_product > 0)
    return mesh

def process_missing_side_walls(vertices, bbox_product, combined_mesh, side_name, tolerance=0.01):
    """
    Process missing side walls of a 3D mesh.

    Parameters:
        vertices (numpy.ndarray): Array of shape (N, 3) representing the vertices of the 3D mesh.
        bbox_product (trimesh.primitives.Box): Bounding box of the product.
        combined_mesh (trimesh.Trimesh): Combined mesh of the product.
        side_name (str): Name of the side to process. Can be one of 'east', 'west', 'north', or 'south'.
        tolerance (float, optional): Tolerance value for vertex selection. Defaults to 0.01.

    Returns:
        trimesh.Trimesh: Combined mesh with the missing side walls processed.
    """
    
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