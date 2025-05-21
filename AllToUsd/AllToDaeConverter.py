import pyassimp
from pyassimp import postprocess
import os
import random
import string

class AllToDae:
    def __init__(self,input_file):
        self.input_file = input_file
        prefix = "_TEST_"
        random_part = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        self.output_file = input_file[:-4]+prefix+random_part+".dae"

    def AllToDaeConvert(self):
        try:
            with pyassimp.load(self.input_file,
                processing=(
                    postprocess.aiProcess_CalcTangentSpace |
                    postprocess.aiProcess_Triangulate |
                    postprocess.aiProcess_JoinIdenticalVertices|
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

                # 防止存在非UTF-8字符
                self._convert_to_utf8()

                return True

        except pyassimp.AssimpError as e:
            print(f"操作失败: {str(e)}")
            return False
    def _convert_to_utf8(self):
        # 尝试以默认编码读取再保存为 UTF-8（忽略非法字符）
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 如果不是 UTF-8，就尝试用系统默认编码读，并替换非法字符
            with open(self.output_file, 'r', encoding='latin1', errors='replace') as f:
                content = f.read()

        # 保存为 UTF-8 编码
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def deleteDae(self):
        os.remove(self.output_file)