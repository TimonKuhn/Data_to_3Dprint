# import open3d as o3d
# triangle = o3d.t.geometry.TriangleMesh([[1.0,1.0,0.0], [0,1,0], [1,0,0]], [[0,1,2]])
# o3d.visualization.draw([{'name': 'wedge', 'geometry': triangle}])
# wedge = triangle.extrude_linear([0,0,1])
# o3d.visualization.draw([{'name': 'wedge', 'geometry': wedge}])

# import open3d as o3d

# mesh = o3d.t.geometry.TriangleMesh([[1,1,0], [0.7,1,0], [1,0.7,0]], [[0,1,2]])
# o3d.visualization.draw([{'name': 'spring', 'geometry': mesh}])
# spring = mesh.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=1.1)
# o3d.visualization.draw([{'name': 'spring', 'geometry': spring}])

