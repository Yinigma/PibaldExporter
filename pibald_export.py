bl_info = {
    "name": "Pibald Exporter",
    "author": "Antler Shed",
    "version": (1, 0),
    "blender": (4, 0, 1),
    "location": "File > Export > Pibald",
    "description": "Exports models in a format that can be read by the work-in-progress Pibald Renderer. Depends on Pibald colors to export colors.",
    "warning": "",
    "doc_url": "",
    "category": "Import-Export",
}

import bpy

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty

from bpy.types import (
    Action,
    Armature,
    Mesh,
    Operator
)
import mathutils
from mathutils import(
    Vector,
    Matrix,
    Quaternion
)
import os
from os import path as os_path
import struct

XLOC = 'xloc'
YLOC = 'yloc'
ZLOC = 'zloc'

WROT = 'wrot'
XROT = 'xrot'
YROT = 'yrot'
ZROT = 'zrot'

XSCALE = 'xscale'
YSCALE = 'yscale'
ZSCALE = 'zscale'

RECOLOR_ATTR_NAME = 'recolor_props'
PALETTE_ATTR_NAME = 'PaletteId'

'''
Constants for more easily converting the animation curves from one coordinate system to another
'''
KEY_TABLE = {'location': [XLOC, YLOC, ZLOC], 'rotation_quaternion': [WROT, XROT, YROT, ZROT], 'scale': [XSCALE, YSCALE, ZSCALE]}
LOC_INDEX_TABLE = [XLOC, YLOC, ZLOC]
ROT_INDEX_TABLE = [XROT, YROT, ZROT]
SCALE_INDEX_TABLE = [XSCALE, YSCALE, ZSCALE]
WRITE_KEY_TABLE = [XLOC, YLOC, ZLOC, XROT, YROT, ZROT, WROT, XSCALE, YSCALE, ZSCALE]

def action_in_armature(action: Action, armature: Armature) -> bool:
    return set([curve.data_path.split('\"')[1] for curve in action.fcurves]).issubset(set([bone.name for bone in armature.bones]))
    

def insert_and_get_index(v: Vector, l: list[Vector]):
    if any((m - v).length == 0.0 for m in l):
        return next(i for i, m in enumerate(l) if (m - v).length < 0.00001)
    else:
        l.append(v)
        return len(l) - 1
    
'''
Exports a mesh in the following byte format

pibm format
all byte sequences are stored in little endian format
<size_flag-u8: 00000wcn - n: has normal data, c: has color data, w: has weight data>
<number_of_unique_color_values-u16>
<color_value-f32, f32, f32> * number_of_unique_color_values
<number_of_palettes-u16>
<size_of_palette-u16> 
<color_index-u16> * size_of_palette * number_of_palettes
<number_of_unique_points-u32>
<location-f32, f32, f32> * number_of_unique_points
if size_flag w
[
    <length_of_armature_id-u8>
    <name_character-utf8> * length_of_armature_id
    [
        <bone_id-u16>
        <weight-f32>
    ] * number_of_unique_points
]
<number_of_unique_normals-u32>
<normal-f32, f32, f32> * number_of_unique_normals
<number_of_triangles-u32>
[
    [
        <location_index-u32>
        if size_flag n <normal_index-u32> 
        if size_flag c <palette_index-u8>
    ] * 3
] * number_of_triangles
<number_of_polygons-u32>
<poly_tri_count-u16> * number_of_polygons
<min_bound-f32, f32, f32>
<max_bound-f32, f32, f32>

'''
def pibald_model_export(mesh : Mesh, colors: list[list[Vector]], bbox_min: Vector, bbox_max: Vector, armature_id: str, bones: list[str], groups: list[str], name: str, filepath : str, write_normals: bool, write_colors: bool, write_weights: bool):
    full_path = os_path.join(filepath, name.replace('.', '') + '.pibm')
    coord_tf = Matrix([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)])
    has_weights = write_weights and bones != None and groups != None
    has_colors = write_colors and colors != None and len(colors) > 0
    
    #this big old block of code just shoves everything into a series of tuples and lists so we have a copy of the model data that won't mess with the runtime model in blender
    mesh.calc_normals_split()
    mesh.calc_loop_triangles()
    with open(full_path, 'wb') as f:
        flags = int(write_normals) | ( int(has_colors) << 1 ) | ( int(has_weights) << 2 )
        f.write(flags.to_bytes(1, 'little'))
        if has_colors:
            col_vals = []
            pals = []
            for palette in colors:
                pal = []
                for color in palette:
                    pal.append(insert_and_get_index(color, col_vals))
                pals.append(pal)
            
            f.write(len(col_vals).to_bytes(2, 'little'))
            for col in col_vals:
                f.write( struct.pack('<fff', *col[:])) #unique color values
            f.write(len(pals).to_bytes(2, 'little')) #number of palettes
            if len(pals) > 0:
                f.write(len(pals[0]).to_bytes(2, 'little'))#number of colors in palette
                for pal in pals:
                    for col_dex in pal:
                        f.write(col_dex.to_bytes(2, 'little'))
            else:
                f.write(b'\x00') #no palette size
        
        verts = []
        norms = []
        tris = []
        polys = []
        weights = {}
        cols = mesh.attributes[PALETTE_ATTR_NAME].data if (mesh.color_attributes.active_color and PALETTE_ATTR_NAME in mesh.attributes) else None
        
        #build a polygon to triangle map because if not the exporter chugs
        tri_map = {}
        for tri in mesh.loop_triangles:
            poly_dex = mesh.loop_triangle_polygons[tri.index].value
            if poly_dex not in tri_map:
                tri_map[poly_dex] = []
            tri_map[poly_dex].append(tri)
            
        vert_offset = 0
        for p in mesh.polygons:
            poly = []
            mesh_tris = tri_map[p.index]
            for mesh_tri in mesh_tris:
                tri = []
                for loop_idx in mesh_tri.loops:
                    #have to rip out the location and normal of these, preferably removing dupes along the way
                    #this is also where the coordinate system gets transformed
                    co = insert_and_get_index(coord_tf @ mesh.vertices[mesh.loops[loop_idx].vertex_index].co, verts)
                    norm = insert_and_get_index(coord_tf @ mesh.loops[loop_idx].normal, norms) if write_normals else None
                    col = cols[loop_idx].value if cols else None
                    if has_weights:
                        weights[co] = [(bones.index(groups[group.group]), group.weight) for group in mesh.vertices[mesh.loops[loop_idx].vertex_index].groups if groups[group.group] in bones] 
                    tri.append((co, norm, col))
                tris.append(tuple(tri))
            polys.append(len(mesh_tris))
        
        for i in range(len(tris)):
            tris[i] = (tris[i][1],tris[i][0],tris[i][2])
        
        f.write( len(verts).to_bytes(4, 'little') ) #number of unique points
        for v in verts:
            f.write( struct.pack('<fff', *v[:]) ) #point data

        if has_weights:
            f.write(len(armature_id).to_bytes(1, 'little')) #length of armature id
            f.write(armature_id.encode('utf-8')) #armature id (unique name)
            for i, v in enumerate(verts):
                v_weights = weights[i]
                f.write( len(v_weights).to_bytes(1, 'little') ) #number of armature weights on this vertex
                for weight in v_weights:
                    f.write( weight[0].to_bytes(2, 'little') ) #bone id of the weight
                    f.write( struct.pack( '<f', weight[1] ) ) #normalized weight
        if write_normals:
            f.write( len(norms).to_bytes(4, 'little') ) #number of normals
            for n in norms:
                f.write( struct.pack('<fff', *n[:])) #normal data
        f.write(len(tris).to_bytes(4, 'little')) #number of triangles
        for t in tris:
            for corner in t:
                f.write(corner[0].to_bytes(4, 'little')) #point index
                if write_normals:
                    f.write(corner[1].to_bytes(4, 'little')) #normal index
                if has_colors:
                    col_dex = 0 if (corner[2] == None) else corner[2]
                    f.write(col_dex.to_bytes(2, 'little')) #color index
        f.write(len(polys).to_bytes(4, 'little')) #number of polygons
        for p in polys:
            f.write(p.to_bytes(2, 'little')) #triangle ids
        f.write( struct.pack('<fff', *bbox_min[:]) ) #bounding box min
        f.write( struct.pack('<fff', *bbox_max[:]) ) #bounding box max
    mesh.free_normals_split()
    
    return full_path

def convert_armature(armature: Armature):
    coord_tf = Matrix([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)]).to_4x4()
    num_roots = 0
    bones = []
    for bone in armature.bones:
        if bone.parent == None:
            num_roots += 1
            arma_loc, arma_rot, arma_scale = (bone.matrix_local).decompose()
            arma_axis, arma_angle = arma_rot.to_axis_angle()
            bones.append((bone.name, coord_tf @ arma_loc, Quaternion(coord_tf @ arma_axis, arma_angle), None))
        else:
            parent_index = next(idx for (idx, p_bone) in enumerate(armature.bones) if p_bone == bone.parent)
            arma_loc, arma_rot, arma_scale = (bone.parent.matrix_local.inverted() @ bone.matrix_local).decompose()
            arma_axis, arma_angle = arma_rot.to_axis_angle()
            bones.append( ( bone.name, coord_tf @ arma_loc, Quaternion(coord_tf @ arma_axis, arma_angle), parent_index  ) )
    if num_roots > 1:
        bones.insert(0, ('root', Vector(), Quaternion(), None))
        for bone in bones:
            bone = (bone[0], bone[1], bone[2], 0 if bone[3] == None else bone[3] + 1)
    return bones

'''
Exports an armature and associated actions in the folowwing byte formats

pibs (pibald skeleton) format
<number_of_bones-u16>
[
    <length_of_bone_name-u8>
    <name_character-utf8> * length_of_bone_name
    <location-f32, f32, f32>
    <rotation-f32, f32, f32, f32>
    <parent_index-u16>
] * number_of_bones
'''

'''
piba (pibald animation) format
<num_bones-u16>
<start_frame-u16>
<end_frame-u16>
[
    <bone_id-u16>
    <number_of_tracks-u8>
    [
        <track_index-u8>
        <number_of_keyframes-u32>
        <frame-u32, f32> * number_of_keyframes
    ] * number_of_tracks
] * num_bones
'''
def pibald_anim_export(armature: Armature, name : str, filepath : str,):
    coord_tf = Matrix([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)])
    full_path = os_path.join(filepath, name.replace('.', '') + '.pibs')
    num_roots = 0
    bones = convert_armature(armature)
    with open(full_path, 'wb') as f:
        
        f.write(len(bones).to_bytes(2, 'little')) #number of bones
        for bone in bones:
            f.write(len(bone[0]).to_bytes(1, 'little')) #length of armature id
            f.write(bone[0].encode('utf-8')) #armature id
            f.write( struct.pack('<fff', *bone[1][:])) #location
            #this is a cruel prank, blender
            f.write( struct.pack('<ffff', *Vector(bone[2]).yzwx[:])) #rotation
            if bone[3] != None:
                f.write(bone[3].to_bytes(2, 'little')) #parent index
            else:
                f.write(b'\xff\xff') #root indicator - you only get one
    
    anim_tracks = {}
    for action in bpy.data.actions:
        if action_in_armature(action, armature):
            clip = []
            for bone in armature.bones:
                bone_tracks = {}
                min_frame = 65535
                max_frame = 0
                for channel in action.groups[bone.name].channels:
                    track = [(int(keyframe.co[0]), keyframe.co[1]) for keyframe in channel.keyframe_points]
                    max_frame = max([max_frame] + [f[0] for f in track] )
                    min_frame = min([min_frame] + [f[0] for f in track] )
                    key_list = KEY_TABLE.get(channel.data_path.split('.')[-1])
                    key = None if key_list == None or channel.array_index > len(key_list) else key_list[channel.array_index]
                    if key != None:
                        bone_tracks[key] = track
                clip.append(bone_tracks)
            anim_tracks[action.name] = (clip, int(action.frame_start) if action.use_frame_range else min_frame, int(action.frame_end) if action.use_frame_range else max_frame)
    
    tf_map = {}
    for r, row in enumerate(coord_tf.row):
        for c, v in enumerate(row):
            if v != 0.0:
                tf_map[LOC_INDEX_TABLE[r]] = (LOC_INDEX_TABLE[c], -1.0 if v < 0.0 else 1.0)
                tf_map[ROT_INDEX_TABLE[r]] = (ROT_INDEX_TABLE[c], -1.0 if v < 0.0 else 1.0)
                tf_map[SCALE_INDEX_TABLE[r]] = (SCALE_INDEX_TABLE[c], 1.0)
        
    for clip_name in anim_tracks:
        clip, start, end = anim_tracks[clip_name]
        full_path = os_path.join(filepath, clip_name.replace('.', '') + '.piba')
        with open(full_path, 'wb') as f:
            f.write( len( clip ).to_bytes(2, 'little') )
            f.write( int(start).to_bytes(2, 'little') )
            f.write( int(end).to_bytes(2, 'little') )
            for i, bone in enumerate(clip):
                f.write(i.to_bytes(2, 'little')) #bone index
                f.write(len(bone).to_bytes(1, 'little')) #number of tracks for this bone
                for track_key in bone:
                    #remap for basis change
                    adj_key = tf_map[track_key][0] if track_key in tf_map else track_key
                    adj_scale = tf_map[track_key][1] if track_key in tf_map else 1.0
                    f.write(WRITE_KEY_TABLE.index(track_key).to_bytes(1, 'little')) #track_index
                    f.write(len(bone[adj_key]).to_bytes(2, 'little')) #number of keyframes
                    for keyframe in bone[adj_key]:
                        f.write(struct.pack('<Hf', *(keyframe[0], adj_scale * keyframe[1]))) #keyframe data

'''
Exports a single mesh with its associated skeleton and animations if applicable
'''
def export_pb(context, path: str, mesh_obj, export_anim: bool, export_colors: bool, export_normals: bool):
    colors = []
    if mesh_obj.data.get(RECOLOR_ATTR_NAME) != None:
        for palette in mesh_obj.data.recolor_props.recolors:
            pal = []
            for color in palette.colors:
                pal.append(mathutils.Vector((color.color.r, color.color.g, color.color.b)))
            colors.append(pal)
    
    min_bound = Vector(mesh_obj.bound_box[0])
    max_bound = Vector(mesh_obj.bound_box[6])
    
    bbox_list = [child for child in mesh_obj.children if child.name == 'bbox']
    if len(bbox_list) > 0:
        min_bound = Vector(bbox_list[0].bound_box[0])
        max_bound = Vector(bbox_list[0].bound_box[6])
    
    #check for armature
    if export_anim and any(mod.type == 'ARMATURE' for mod in mesh_obj.modifiers):
        armature = next(mod for mod in mesh_obj.modifiers if mod.type == 'ARMATURE').object.data
        bone_names = [bone[0] for bone in convert_armature(armature)]
        groups = [group.name for group in mesh_obj.vertex_groups]
        pibald_model_export( mesh_obj.data, colors, min_bound, max_bound, armature.name, bone_names, groups, mesh_obj.name, path, export_normals, export_colors, export_anim )
        pibald_anim_export( armature, mesh_obj.name, path )
    else:
        pibald_model_export(mesh_obj.data, colors, min_bound, max_bound, None, None, None, mesh_obj.name, path, export_normals, export_colors, False)

class PIBALD_OT_ExportPibald(Operator):
    bl_idname = "pibald.export"
    bl_label = "Export Pibald"
    
    directory: StringProperty(subtype='DIR_PATH')

    export_animations: BoolProperty(
        name="Export Animations",
        description="Export skeleton and actions associated with this mesh.",
        default=True,
    )
    
    export_colors: BoolProperty(
        name="Export Palette Indices",
        description="Toggle exporting color data. If color data is not exported, all vertices will have a pallette id of zero.",
        default=True,
    )

    export_normals: BoolProperty(
        name="Export Normals",
        description="Toggle exporting vertex normals.",
        default=True,
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                export_pb(context, self.directory, obj, self.export_animations, self.export_colors, self.export_normals)
        
        return {'FINISHED'}

def menu_func_export(self, context):
    self.layout.operator(PIBALD_OT_ExportPibald.bl_idname, text="Pibald (.pibm/piba/pibs)")

def register():
    bpy.utils.register_class(PIBALD_OT_ExportPibald)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(PIBALD_OT_ExportPibald)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
