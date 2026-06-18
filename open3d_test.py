import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window()

opt = vis.get_render_option()

print(type(opt))
print(dir(opt))