import collada
import collada.polygons
import collada.polylist
import collada.triangleset
import numpy as np
import math
import time
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade,UsdLux

from collada import Collada

class DaeToUsdConverter:
    def __init__(self,dae_file):
        if not dae_file.endswith('.dae'):
            raise ValueError("Input must be a COLLADA (.dae) file")
        # 解析 DAE 文件
        self.dae_file = dae_file
        try:
            self.mesh_dae = collada.Collada(dae_file)
        except collada.DaeError as e:
            raise RuntimeError(f"Failed to parse DAE file: {str(e)}")
        
    def convert_visual_scene(self):
        visual_scene = self.mesh_dae.scene
        nodes = visual_scene.nodes
        # print(nodes)
        def process_node(parent_prim_path, node):   # transform=None
            """递归处理场景节点"""
            # 创建当前节点的prim
            #print(node.id)
            node_name = node.id.replace("-", "_").replace(" ", "_").replace(".","_")
            #print(node_name)
            node_prim_path = f"{parent_prim_path}/{node_name}"
            #print(node_prim_path)
            node_prim = self.stage.DefinePrim(node_prim_path, "Xform")

            # 设置变换
            xform = UsdGeom.Xformable(node_prim)
            # xform_op = xform.AddTransformOp()
            for transform in node.transforms:
                node_matrix = np.array(transform.matrix, dtype=np.float64)# 显式转换为float64
                node_matrix = node_matrix.reshape(4, 4).T  # 转置矩阵（处理行列顺序差异）
                node_transform = Gf.Matrix4d(*node_matrix.flatten().tolist())  # 展开为列表
                    
                if isinstance(transform,collada.scene.MatrixTransform):
                    xform.AddTransformOp().Set(node_transform)
                elif isinstance(transform,collada.scene.TranslateTransform):
                    xform.AddTranslateOp().Set(node_transform)
                elif isinstance(transform,collada.scene.RotateTransform):
                    xform.AddRotateOp().Set(node_transform)
                elif isinstance(transform,collada.scene.ScaleTransform):
                    xform.AddScaleOp().Set(node_transform)

                
            # 将几何实例映射到相应路径中
            instance_types = ['geometry', 'camera', 'light']
            for tipo in instance_types:
                for instance in node.objects(tipo=tipo):  # 传递 tipo 参数
                    instance_url = instance.original.id
                    if(tipo == 'geometry'):
                        self.mat_dict[instance_url]=instance.materialnodebysymbol    # {'door-material': <MaterialNode symbol=door-material targetid=door-material>, 'door_glass-material': <MaterialNode symbol=door_glass-material targetid=door_glass-material>}
                    #print(f"Found {tipo} instance: {instance_url}")
                    self.geometry_map[instance_url]=node_prim_path  # 存储当前Xform的路径
                # 递归处理子节点
            if hasattr(node, 'children'):
                for child_node in node.children:
                    if hasattr(child_node, 'matrix') or hasattr(child_node, 'rotate') or hasattr(child_node, 'scale') or hasattr(child_node, 'translate'):
                        process_node(node_prim_path, child_node)  # 子节点变换是相对于父节点的

        # 从场景根节点开始处理
        for node in nodes:
            process_node("/Root",node)
    
    def convert_materials(self):
        material_scope = self.stage.DefinePrim("/Root/Materials", "Scope")
        for mat in self.mesh_dae.materials:
            # 创建材质
            mat_name = mat.id.replace("-","_").replace(" ","_").replace(".","_")
            usd_mat = UsdShade.Material.Define(self.stage,f"{material_scope.GetPath()}/{mat_name}")
            self.mat_map[mat.id] = usd_mat

            # 创建着色器节点
            mat_shader = UsdShade.Shader.Define(self.stage,f"{usd_mat.GetPath()}/Shader")
            mat_shader.CreateIdAttr("UsdPreviewSurface")

            # 连接着色器节点与材质
            usd_mat.CreateSurfaceOutput().ConnectToSource(mat_shader.ConnectableAPI(),"surface")
            Texcoord = "st"
            # 创建几何体与材质的输入接口  
            stInput = usd_mat.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
            #stInput.Set(Texcoord)                                                                                                                                           # 修改
           
            # 获取<library_effect> 
            effect = mat.effect

            # 处理emission  与USD中的emissionColor对应
            if effect.emission is not None:
                if isinstance(effect.emission, tuple):
                    emission_color = Gf.Vec3f(effect.emission[0], effect.emission[1], effect.emission[2])
                    if emission_color!=(0.11764706, 0.11764706, 0.11764706):
                        mat_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emission_color)
                elif isinstance(effect.emission, collada.material.Map):
                    sampler_surface = effect.emission.sampler.surface
                    img = sampler_surface.image
                    imgpath = img.path

                    # 创建着色器
                    texture_shader = UsdShade.Shader.Define(self.stage,f"{usd_mat.GetPath()}/EmissionTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)
                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")  # 假设纹理为sRGB颜色空间

                    # 设置UV坐标输入
                    texcoord = effect.emission.texcoord  # 通常为 "UVMap0" 或其他名称
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Emission_st_reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)                                                                # 指定使用哪个uv坐标

                    # 连接 UV 到纹理的 st 输入
                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                    
                    # 将纹理的 RGB 输出连接到材质的 emissiveColor
                    mat_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        texture_shader.ConnectableAPI(), "rgb"
                    )
                    
                    # 设置 emissiveFactor（如果需要调整亮度）
                    # mat_shader.CreateInput("emissiveFactor", Sdf.ValueTypeNames.Float).Set(1.0)

            # 处理ambient   与USD中无对应属性
            if effect.ambient is not None:
                if isinstance(effect.ambient, tuple):
                    ambient_color = Gf.Vec3f(effect.ambient[0], effect.ambient[1], effect.ambient[2])
                    mat_shader.CreateInput("ambientColor", Sdf.ValueTypeNames.Color3f).Set(ambient_color)
                elif isinstance(effect.ambient, collada.material.Map):                                                                                                              # 待完成
                    print("ambient与USD中无对应属性")
            
            # 处理diffuse
            if effect.diffuse is not None:
                if isinstance(effect.diffuse, tuple):
                    diffuse_color = Gf.Vec3f(effect.diffuse[0], effect.diffuse[1], effect.diffuse[2])
                    mat_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
                elif isinstance(effect.diffuse, collada.material.Map):
                    sampler_surface = effect.diffuse.sampler.surface
                    # print(effect.diffuse.sampler.surface)
                    Texcoord = effect.diffuse.texcoord
                    img = sampler_surface.image
                    imgpath = img.path
                    # print(imgpath)
                    # 创建纹理坐标读取器 并设置其类型为 UsdPrimvarReader_float2
                    stReader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/stReader")
                    stReader.CreateIdAttr('UsdPrimvarReader_float2')
                    # 将读取器的 varname 输入连接到材质的 frame:stPrimvarName 输入
                    stReader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(Texcoord)                                                        
                    #stReader.CreateInput('varname',Sdf.ValueTypeNames.String).ConnectToSource(stInput)
                    # 创建漫反射纹理采样器 并设置其类型为 UsdUVTexture。
                    diffuse_texture = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/DiffuseTexture")
                    diffuse_texture.CreateIdAttr("UsdUVTexture")
                    # 设置纹理文件路径
                    diffuse_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)
                    # 连接纹理采样器的纹理坐标输入
                    diffuse_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
                    # 设置纹理在 S 和 T 方向上的包裹模式为 repeat。
                    diffuse_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    diffuse_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                    # 创建纹理采样器的 RGB 输出
                    diffuse_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                    # 连接 PBR 着色器的漫反射颜色输入
                    mat_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse_texture.ConnectableAPI(), "rgb")
            
            # 处理specular 与 reflective    与USD中的specularColor对应
            if effect.specular is not None:
                if isinstance(effect.specular, tuple):
                    specular_color = Gf.Vec3f(effect.specular[0], effect.specular[1], effect.specular[2])
                    mat_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(specular_color)
                elif isinstance(effect.specular, collada.material.Map):                                                                                               
                    sampler_surface = effect.specular.sampler.surface
                    texcoord = effect.specular.texcoord  # 通常为 "UVMap0" 或其他名称
                    img = sampler_surface.image
                    imgpath = img.path

                    # 设置UV坐标输入
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Specular_st_reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord) 

                    # 创建反射纹理采样器
                    texture_shader = UsdShade.Shader.Define(self.stage,f"{usd_mat.GetPath()}/SpecularTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)
                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")

                    # 连接 UV 到纹理的 st 输入
                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                    
                    # 将纹理的 RGB 输出连接到材质的 specularColor
                    mat_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        texture_shader.ConnectableAPI(), "rgb"
                    )
                    
                    # mat_shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(1.0)
            elif effect.reflective is not None:
                if isinstance(effect.reflective, tuple):
                    reflective_color = Gf.Vec3f(effect.reflective[0], effect.reflective[1], effect.reflective[2])
                    mat_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(reflective_color)
                elif isinstance(effect.reflective, collada.material.Map):                                                                                                      
                    sampler_surface = effect.reflective.sampler.surface
                    img = sampler_surface.image
                    imgpath = img.path

                    # 创建着色器
                    texture_shader = UsdShade.Shader.Define(self.stage,f"{usd_mat.GetPath()}/ReflectiveTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)
                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # 使用线性空间

                    # 设置UV坐标输入
                    texcoord = effect.reflective.texcoord  # 通常为 "UVMap0" 或其他名称
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Reflective_st_reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)                                                                # 指定使用哪个uv坐标

                    # 连接 UV 到纹理的 st 输入
                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                    
                    # 将纹理的 RGB 输出连接到材质的 specularColor
                    mat_shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        texture_shader.ConnectableAPI(), "rgb"
                    )
                    
                    # mat_shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(1.0)

            # 处理shininess 与USD中roughness对应
            if effect.shininess is not None:
                if isinstance(effect.shininess, float):
                    roughness = math.sqrt(2/(effect.shininess+2))                                                          
                    mat_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
                elif isinstance(effect.shininess, collada.material.Map):
                    sampler_surface = effect.reflective.sampler.surface
                    img = sampler_surface.image
                    imgpath = img.path
                    # 设置UV坐标输入
                    texcoord = effect.shininess.texcoord
                    
                    
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Roughness_ST_Reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)
                    
                    
                    # 创建粗糙度纹理着色器
                    texture_shader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/RoughnessTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)
                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # 粗糙度使用线性空间

                    # 连接 UV 到纹理的 st 输入
                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")

                    # 创建中间数学计算节点（用于精确转换）
                    # 将0~1的纹理值转换为COLLADA的shininess范围（假设原始范围为0~100）
                    multiply_node = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/RoughnessMultiply")
                    multiply_node.CreateIdAttr("UsdTransform2d")
                    multiply_node.CreateInput("scale", Sdf.ValueTypeNames.Float).Set(100.0)  # 扩展纹理值到0~100范围
                    texture_shader.ConnectableAPI().ConnectToSource(multiply_node.ConnectableAPI(), "r")

                    # 简化方案：使用近似转换 roughness = 1.0 - (texture.r / 100.0)
                    texture_shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-0.01, 0, 0, 0))  # 将0~100映射到1~0
                    texture_shader.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.0, 0, 0, 0))



                    # 连接到roughness属性
                    mat_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                        texture_shader.ConnectableAPI(), "r"
                    )

            daeReflectivity = 0.0
            # 处理reflectivity  与USD中metallic对应
            if effect.reflectivity is not None:
                if isinstance(effect.reflectivity, float):
                    daeReflectivity = effect.reflectivity
                    mat_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(effect.reflectivity)
                elif isinstance(effect.reflectivity, collada.material.Map):                                                                                                        
                    sampler_surface = effect.reflective.sampler.surface
                    img = sampler_surface.image
                    imgpath = img.path

                    # 创建金属度纹理着色器（唯一路径）
                    texture_shader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/MetallicTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)

                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")

                    texcoord = effect.reflective.texcoord
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Metallic_ST_Reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)

                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                    mat_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(texture_shader.ConnectableAPI(), "r")

            # 设置USD中useSpecularColor属性
            if daeReflectivity>0.5:
                mat_shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            else:
                mat_shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

            # 处理transparent 与 transparency   与USD中opacity对应
            opacity = 1.0
            transparency = 0
            alpha = 0
            if effect.transparency is not None:
                if isinstance(effect.transparency, float):
                    transparency = effect.transparency
                    opacity = transparency
                    if opacity==0:
                        opacity=1
                    mat_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
                elif isinstance(effect.transparency, collada.material.Map):
                    sampler_surface = effect.transparency.sampler.surface
                    img = sampler_surface.image
                    imgpath = img.path

                    # 创建透明度纹理着色器
                    texture_shader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/OpacityTexture")
                    texture_shader.CreateIdAttr("UsdUVTexture")
                    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)

                    texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")

                    texcoord = effect.transparency.texcoord
                    st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Opacity_ST_Reader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)

                    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                    mat_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(texture_shader.ConnectableAPI(), "r")
            if effect.transparent is not None:
                # print("alpha")
                if isinstance(effect.transparent, tuple):
                    alpha = effect.transparent[3]
                    opacity = 1.0 - transparency*(1-alpha)
                    if opacity==0:
                        opacity=1
                    mat_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
                elif isinstance(effect.transparent, collada.material.Map):
                    # 检查透明度纹理着色器是否存在
                    if not self.stage.GetPrimAtPath(f"{usd_mat.GetPath()}/OpacityTexture").IsValid():
                        sampler_surface = effect.transparent.sampler.surface
                        img = sampler_surface.image
                        imgpath = img.path

                        # 创建透明度纹理着色器
                        texture_shader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/OpacityTexture")
                        texture_shader.CreateIdAttr("UsdUVTexture")
                        texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(imgpath)

                        texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")

                        texcoord = effect.transparent.texcoord
                        st_reader = UsdShade.Shader.Define(self.stage, f"{usd_mat.GetPath()}/Opacity_ST_Reader")
                        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(texcoord)

                        texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
                        mat_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(texture_shader.ConnectableAPI(), "r")
           
            # print(transparency)
            # print(alpha)
            # print(opacity)     

            # 处理index_of_refraction   与usd中ior对应
            if effect.index_of_refraction is not None:
                ior = effect.index_of_refraction
                mat_shader.CreateInput("ior",Sdf.ValueTypeNames.Float).Set(ior)

            # 处理double_sided
            if effect.double_sided is not None:
                if effect.double_sided:                                                                                                                                             
                    usd_mat.CreateInput("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                else:
                    usd_mat.CreateInput("doubleSided", Sdf.ValueTypeNames.Bool).Set(False)

            # 将接口绑定
            stInput.Set(Texcoord) 

    def convert_geometry(self):
        for i, geometry in enumerate(self.mesh_dae.geometries):
            # 创建 Mesh Prim
            original_name = geometry.id
            original_name = original_name.replace("-","_").replace(" ","_").replace(".","_")
            mesh_parent_path = self.geometry_map[geometry.id]
            # 遍历每个 TriangleSet
            
            for idx, primitive in enumerate(geometry.primitives):
                # 如果几何体包含多个集合，为每个集合创建单独的 Mesh
                if isinstance(primitive,collada.triangleset.TriangleSet):
                    mesh_prim_name = f"{original_name}_TriSet_{idx}" if len(geometry.primitives) > 1 else original_name
                elif isinstance(primitive,collada.polylist.Polylist):
                    mesh_prim_name = f"{original_name}_Polylist_{idx}" if len(geometry.primitives) > 1 else original_name
                mesh_prim_path = f"{mesh_parent_path}/{mesh_prim_name}"
                mesh_prim = self.stage.DefinePrim(mesh_prim_path, "Mesh")
                mesh_t = UsdGeom.Mesh(mesh_prim)
                #print(mesh_prim_path)

                vertices = primitive.vertex
                indices = primitive.vertex_index
                normals = primitive.normal  # 每条法线的坐标
                normals_index = primitive.normal_index  # 每条法线的面索引
                texcoords = primitive.texcoordset
                texcoords_index = primitive.texcoord_indexset
                materialNode = self.mat_dict[geometry.id][primitive.material]    # 通过symbol来确定一系列材质对象中的具体材质
                
                # 写入顶点数据  vertex、vertex_index
                if vertices is not None:
                    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
                    mesh_t.CreatePointsAttr(points)
                # print(vertices)

                if isinstance(primitive,collada.triangleset.TriangleSet):
                # 写入面数量与顶点索引
                    if indices is not None:
                        face_counts = []
                        for point in indices:
                            face_counts.append(len(point))                                 
                        # face_counts = [len(point) for point in indices]
                        mesh_t.CreateFaceVertexCountsAttr(face_counts)
                        mesh_t.CreateFaceVertexIndicesAttr(indices)
                    
                # 写入法线数据  normal、normal_index
                    if normals is not None and normals.any():
                        mesh_normals=[]
                        for n in normals_index:
                            for m in n:
                                # count = count+3
                                nx = float(normals[m][0])
                                ny = float(normals[m][1])
                                nz = float(normals[m][2])
                                mesh_normals.append(Gf.Vec3f(nx, ny, nz))
                        mesh_t.CreateNormalsAttr(mesh_normals)
                    # print(count)

                # 提取 UV 坐标  (名称为UVMap)   texcoordset、texcoord_indexset
                    for idx, texcoord in enumerate(texcoords):
                        txset = []
                        for ti in texcoords_index[idx]:
                            for m in ti:
                                tx = float(texcoord[m][0])
                                ty = float(texcoord[m][1])
                                txset.append(Gf.Vec2f(tx,ty))
                        primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
                        uv_primvar = primvar_api.CreatePrimvar(materialNode.inputs[idx][0], Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                        uv_primvar.Set(txset)
                elif isinstance(primitive,collada.polylist.Polylist):
                # 写入面数量与顶点索引
                    # print(indices)
                    if indices is not None:                                
                        mesh_t.CreateFaceVertexCountsAttr(primitive.vcounts)
                        mesh_t.CreateFaceVertexIndicesAttr(indices)

                # 写入法线数据  normal、normal_index
                    if normals is not None:
                        mesh_normals=[]
                        for n in normals_index:
                            nx = float(normals[n][0])
                            ny = float(normals[n][1])
                            nz = float(normals[n][2])
                            mesh_normals.append(Gf.Vec3f(nx, ny, nz))
                        mesh_t.CreateNormalsAttr(mesh_normals)
                    # print(count)

                # 提取 UV 坐标  (名称为UVMap)   texcoordset、texcoord_indexset
                    for idx,texcoord in enumerate(texcoords):
                        txset = []
                        for ti in texcoords_index[idx]:
                            tx = float(texcoord[ti][0])
                            ty = float(texcoord[ti][1])
                            txset.append(Gf.Vec2f(tx,ty))
                        primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
                        # print(materialNode.inputs)    # [('CHANNEL0', 'TEXCOORD', '0')]
                        uv_primvar = primvar_api.CreatePrimvar(materialNode.inputs[idx][0], Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                        uv_primvar.Set(txset)

                # 绑定几何纹理          应该使用之前获得的材质字典  与多组纹理搭配使用
                mesh_prim.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
                UsdShade.MaterialBindingAPI(mesh_prim).Bind(self.mat_map[materialNode.target.id])
    
    def convert_cameras(self):
        for camera in self.mesh_dae.cameras:
            original_name = camera.id
            # print(original_name)
            original_name = original_name.replace("-","_").replace(" ","_").replace(".","_")
            camera_parent_path = self.geometry_map[camera.id]
            camera_prim_path = f"{camera_parent_path}/{original_name}"
            camera_prim = UsdGeom.Camera.Define(self.stage,camera_prim_path)
            # print(type(camera))
            # 处理不同相机类型
            if isinstance(camera, collada.camera.PerspectiveCamera):
                #print("PerspectiveCamera")
                camera_prim.CreateProjectionAttr().Set("perspective")

                if camera.xfov is not None:
                    xfov_rad = math.radians(camera.xfov)
                    horizontal_aperture_mm = 36.0  # 标准35mm胶片水平孔径（毫米）
                    if xfov_rad==0:
                        focal_length_mm = 0
                    else:
                        focal_length_mm = (horizontal_aperture_mm / 2) / math.tan(xfov_rad / 2)
                    camera_prim.CreateFocalLengthAttr().Set(focal_length_mm / 10)
                    camera_prim.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm / 10)
                    if camera.aspect_ratio is not None and camera.aspect_ratio!=0:
                        camera_prim.CreateVerticalApertureAttr().Set((horizontal_aperture_mm / 10) / camera.aspect_ratio)

                elif camera.yfov is not None:
                    yfov_rad = math.radians(camera.yfov)
                    vertical_aperture_mm = 24.0  # 标准24mm胶片垂直孔径（毫米）
                    focal_length_mm = (vertical_aperture_mm / 2) / math.tan(yfov_rad / 2)
                    camera_prim.CreateFocalLengthAttr().Set(focal_length_mm / 10)
                    camera_prim.CreateVerticalApertureAttr().Set(vertical_aperture_mm / 10)
                    if camera.aspect_ratio is not None:
                        camera_prim.CreateHorizontalApertureAttr().Set((vertical_aperture_mm / 10) * camera.aspect_ratio)
                
                if camera.znear is not None and camera.zfar is not None:
                    camera_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(camera.znear, camera.zfar))
                
                camera_prim.CreateHorizontalApertureOffsetAttr().Set(0.0)
                camera_prim.CreateVerticalApertureOffsetAttr().Set(0.0)

            elif isinstance(camera, collada.camera.OrthographicCamera):
                #print("OrthographicCamera")
                camera_prim.CreateProjectionAttr().Set("orthographic")

                if camera.xmag is not None:
                    camera_prim.CreateHorizontalApertureAttr().Set(camera.xmag*2*100)
                if camera.ymag is not None:
                    camera_prim.CreateVerticalApertureAttr().Set(camera.ymag*2*100)

                if camera.aspect_ratio is not None:
                    if camera.xmag is not None:
                        camera.CreateVerticalApertureAttr().Set(camera.xmag*2*100/camera.aspect_ratio)
                    elif camera.ymag is not None:
                        camera_prim.CreateHorizontalApertureAtt().Set(camera.ymag*2*100*camera.aspect_ratio)

                camera_prim.CreateHorizontalApertureOffsetAttr().Set(0.0)
                camera_prim.CreateVerticalApertureOffsetAttr().Set(0.0)

                if camera.znear is not None and camera.zfar is not None:
                    camera_prim.CreateClippingRangeAttr().Set(Gf.Vec2f(camera.znear, camera.zfar))
            else:
                print("Camera Type Error!")
    
    def convert_lights(self):
        for light in self.mesh_dae.lights:
            original_name = light.id.replace("-","_").replace(" ","_").replace(".","_")
            light_parent_path = self.geometry_map[light.id]
            light_prim_path = f"{light_parent_path}/{original_name}"
            # 类型决策树
            if isinstance(light, collada.light.PointLight):
                # for attr in dir(light):
                #     if not attr.startswith("__"):
                #         print(attr)
                usd_light = UsdLux.SphereLight.Define(self.stage, light_prim_path)
                if light.color is not None:
                    color = Gf.Vec3f(light.color[:3])
                    usd_light.CreateColorAttr().Set(color)
                else:
                    print("Error: Light Attr is Error")
                    return
                if light.constant_att is not None:
                    const_atten = light.constant_att
                else:
                    const_atten = 1.0
                if light.linear_att is not None:
                    linear_atten = light.linear_att
                else:
                    linear_atten = 0.0
                if light.quad_att is not None:
                    quad_atten = light.quad_att
                else:
                    quad_atten = 0.0
                # 基于物理模型计算光照强度
                max_component = max(color[:3])
                attenuation = (const_atten + linear_atten + quad_atten)
                intensity = (max_component * 683.0) / (attenuation + 1e-5)
                usd_light.CreateIntensityAttr().Set(intensity)
                # 有效半径
                radius = 1.0 / math.sqrt(const_atten + 1e-5)
                usd_light.CreateRadiusAttr().Set(radius)
                # 衰减模型标记
                if quad_atten > 0:
                    usd_light.CreateTreatAsPointAttr().Set(True)
            elif isinstance(light, collada.light.SpotLight):
                usd_light = UsdLux.SphereLight.Define(self.stage, light_prim_path)
                shaping_api = UsdLux.ShapingAPI.Apply(usd_light.GetPrim())
                if light.color is not None:
                    color = Gf.Vec3f(light.color[:3])
                    usd_light.CreateColorAttr().Set(color)
                else:
                    print("Error: Light Attr is Error")
                    return
                if light.constant_att is not None:
                    const_atten = light.constant_att
                else:
                    const_atten = 1.0
                if light.linear_att is not None:
                    linear_atten = light.linear_att
                else:
                    linear_atten = 0.0
                if light.quad_att is not None:
                    quad_atten = light.quad_att
                else:
                    quad_atten = 0.0
                if light.falloff_ang is not None:
                    cone_angle = math.radians(light.falloff_ang)
                else:
                    cone_angle = math.radians(180)
                if light.falloff_exp is not None:
                    cone_exponent = light.falloff_exp
                else:
                    cone_exponent = 0.0
                # 有效半径
                radius = 1.0 / math.sqrt(const_atten + 1e-5)
                usd_light.CreateRadiusAttr().Set(radius)
                # 基于物理模型计算光照强度
                max_component = max(color[:3])
                attenuation = (const_atten + linear_atten + quad_atten)
                intensity = (max_component * 683.0) / (attenuation + 1e-5)
                usd_light.CreateIntensityAttr().Set(intensity)
                # 聚光锥角
                shaping_api.CreateShapingConeAngleAttr(cone_angle)
                # 边缘软化计算
                softness = cone_exponent / (cone_exponent + 5.0)
                shaping_api.CreateShapingConeSoftnessAttr(softness)
                # 衰减曲线混合控制
                if linear_atten > 0:
                    focus = 1.0 - linear_atten/(quad_atten + 1e-5)
                    shaping_api.CreateShapingFocusAttr(focus)
            elif isinstance(light, collada.light.DirectionalLight):
                usd_light = UsdLux.DistantLight.Define(self.stage, light_prim_path)
                if light.color is not None:
                    color = Gf.Vec3f(light.color[:3])
                    usd_light.CreateColorAttr().Set(color)
                else:
                    print("Error: Light Attr is Error")
                    return
            elif isinstance(light, collada.light.AmbientLight):
                usd_light = UsdLux.DomeLight.Define(self.stage, light_prim_path)
                if light.color is not None:
                    color = Gf.Vec3f(light.color[:3])
                    usd_light.CreateColorAttr().Set(color)
                else:
                    print("Error: Light Attr is Error")
                    return
            else:
                print(f"Unsupported light type: {light.type}")
                continue

    def convert_dae_to_usd(self,usd_file=None):
        # 创建usd保存路径
        self.usd_file = usd_file
        if self.usd_file is None:
            self.usd_file = self.dae_file[:-15]+".usd"
        # 创建 USD 场景
        self.stage = Usd.Stage.CreateNew(self.usd_file)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)  # 设置 Y 轴为上方向
        # 创建根节点
        root_prim = self.stage.DefinePrim("/Root", "Xform")
        self.stage.SetDefaultPrim(root_prim)
        self.geometry_map = {}  # 存储几何体ID到USD路径的映射
        self.mat_map = {}       # 存储材质ID到material的映射
        self.mat_dict={}        # 存储几何体ID与材质字典的映射

        # 处理视觉场景
        if self.mesh_dae.scene:
            vistime = time.time()
            self.convert_visual_scene()
            vistime = time.time()-vistime
            print(f"处理视觉场景用时{vistime:.4f}")
        

        # 处理材质 
        if self.mesh_dae.materials:
            mattime = time.time()
            self.convert_materials()
            mattime=time.time()-mattime
            print(f"处理材质用时{mattime:.4f}")

        # 遍历几何体并转换为 USD 
        if self.mesh_dae.geometries:
            geomtime=time.time()
            self.convert_geometry()
            geomtime=time.time()-geomtime
            print(f"处理几何体用时{geomtime:.4f}")

        # 处理相机参数
        if self.mesh_dae.cameras:
            cametime=time.time()
            self.convert_cameras()
            cametime=time.time()-cametime
            print(f"处理相机用时{cametime:.4f}")

        # 处理灯光参数
        if self.mesh_dae.lights:
            ligtime=time.time()
            self.convert_lights()
            ligtime=time.time()-ligtime
            print(f"处理灯光用时{ligtime:.4f}")

        # 保存 USD 文件
        self.stage.GetRootLayer().Save()
        return self.usd_file