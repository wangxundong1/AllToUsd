import pyassimp
from pyassimp import postprocess
import os

class AllToDae:
    def __init__(self,input_file):
        self.input_file = input_file
        self.output_file = input_file[:-3]+"dae"

    def AllToDaeConvert(self):
        try:
            with pyassimp.load(self.input_file,
                processing=(
                    postprocess.aiProcess_CalcTangentSpace |
                    postprocess.aiProcess_Triangulate |
                    postprocess.aiProcess_JoinIdenticalVertices |
                    postprocess.aiProcess_SortByPType
                )
            ) as scene:
                # 导出为 DAE 格式
                pyassimp.export(
                    scene,
                    self.output_file,
                    file_type="collada"
                )
                print(f"成功导出到: {self.output_file}")
                return True

        except pyassimp.AssimpError as e:
            print(f"操作失败: {str(e)}")
            return False

    def deleteDae(self):
        os.remove(self.output_file)