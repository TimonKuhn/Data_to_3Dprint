import open3d as o3d
import trimesh
import numpy as np

import datato3D as d3d

from datetime import datetime
print("start", datetime.now())
### adjust all paths and change the parameters as needed, where the comments are


### Bounding Box ############################################################################################################
path = 'Zuerich\Zuerich_kern2.txt'
bbox_product = d3d.read_lv95_do_rectangular_bbox(path)

print("bboxes", datetime.now())


ply_file_path = 'Zuerich\output\surface_Zuerich_delauney12.ply'
mesh = o3d.io.read_triangle_mesh(ply_file_path)

print("mesh gelesen", datetime.now())

mesh_cropped = d3d.cutting_mesh_with_bbox(mesh, bbox_product, 1, 4500) ### min and max height

print("mesh zuschneiden", datetime.now())

# output_file = 'Zuerich\output\surface_Zuerich.ply' ### output Surface (.ply) file, normaly not used
# o3d.io.write_triangle_mesh(output_file, mesh_smooth)

print("mesh oberfläche to file", datetime.now())

thickness = 300 ### thickness of the thinnest part of the solid mesh
solid_mesh = d3d.surface_to_volume(mesh_cropped, thickness)

print("mesh to volume", datetime.now())

output_file = "Zuerich\output\Stadtkern_north.stl" ### output Solid (.stl) file 
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
