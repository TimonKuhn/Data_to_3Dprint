import open3d as o3d
import trimesh
import numpy as np

import datato3D as d3d

from datetime import datetime
print("start", datetime.now())
### adjust all paths and change the parameters as needed, where the comments are


### Bounding Box ############################################################################################################


### non rectangular bbox are only working for DOM and not for building models!
import geopandas as gpd

shapefile_path = "Zuerich\Zürich_Grenze.shp"
# Geometrie aus der Shapefile lesen
shape_data = gpd.read_file(shapefile_path)
geometry = shape_data.geometry[0]

bbox_product = geometry


###rest of the code as always

path = "Zuerich\Zuerich_extent.txt" ### input coordinates in LV95 (.txt)
bbox_working_region = d3d.buffer_polygon(bbox_product, 0.1) ### buffer around the product bbox so you do not have calculation errors at the edges

print("bboxes", datetime.now())
### DEM #####################################################################################################################

## if your DEM-file does not cover the area, try d3d.download_raster_files() and stitch_raster_files() to get a DEM that covers the area
## automatically download and stich raster files from the csv you get when you draw a polygon on https://www.swisstopo.admin.ch/de/hoehenmodell-swissalti3d

csv_path = "Zuerich\swissSurfaceRaster_Zürich.csv"
output_directory = "Zuerich\DEM"
raster_files = d3d.download_raster_files(csv_path, output_directory)

output_filename = "Zuerich\DEM\merged_raster.tif"
d3d.stitch_raster_files(raster_files, output_filename)

print("raster donwloaded", datetime.now())

dem_path = output_filename ### input DEM file, can also be a DSM (.tif)
out_image, out_transform, src = d3d.read_raster_dem_cut_to_bbox(dem_path, bbox_product)
out_image = d3d.slice_out_image(out_image) ### largest value in the DEM, can be found in the metadata of the DEM

print("raster einlesen", datetime.now())

mesh, pcd = d3d.dem_to_mesh(out_image, out_transform, 0.1, 30, 12)  ### 3 advanced parameters, explained in detail in the Module

print("mesh erstellen", datetime.now())

mesh_cropped = d3d.cutting_mesh_with_bbox(mesh, bbox_product, 1, 4500) ### min and max height

print("mesh zuschneiden", datetime.now())

mesh_smooth = mesh_cropped.filter_smooth_taubin(number_of_iterations=3) ### number of iterations in smoothing
mesh_smooth.compute_triangle_normals()

print("mesh nachbearbeiten", datetime.now())

output_file = 'Zuerich\output\surface_Zuerich.ply' ### output Surface (.ply) file, normaly not used
o3d.io.write_triangle_mesh(output_file, mesh_smooth)

print("mesh oberfläche to file", datetime.now())

thickness = 300 ### thickness of the thinnest part of the solid mesh
solid_mesh = d3d.surface_to_volume(mesh_smooth, thickness)

print("mesh to volume", datetime.now())

output_file = "Zuerich\output\output_solid_mesh.stl" ### output Solid (.stl) file 
o3d.io.write_triangle_mesh(output_file, solid_mesh) 

print("mesh solid to file", datetime.now())

### Buildings ###############################################################################################################

if False: ### change to False if you do not want to process the Buildings
    shrink_factor = 0 ### shrinking % the boundries of the product bbox so the buildings will fit properly, can cause problems
    bbox_product = d3d.shrink_bbox(bbox_product, shrink_factor)

    mesh_path = 'testfiles/3DStadtmodell_beschränkt.obj' ### input 3D model Buildings file
    mesh = trimesh.load_mesh(mesh_path)
    mesh_cut = d3d.cutting_mesh_through_middle_of_faces(mesh, bbox_product)

    edge_vertices = mesh_cut.outline().vertices

    coordinates = np.array(edge_vertices[:, :3])

    combined_mesh = mesh_cut

    combined_mesh = d3d.process_missing_side_walls(coordinates, bbox_product, combined_mesh, 'east')
    combined_mesh = d3d.process_missing_side_walls(coordinates, bbox_product, combined_mesh, 'west')
    combined_mesh = d3d.process_missing_side_walls(coordinates, bbox_product, combined_mesh, 'north')
    combined_mesh = d3d.process_missing_side_walls(coordinates, bbox_product, combined_mesh, 'south')

    trimesh.repair.fill_holes(combined_mesh)

    output_file = 'testfiles/stadtmodell_bbox.stl' ### output Buildings file
    combined_mesh.export(output_file)
