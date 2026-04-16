from PySide6.QtGui import QUndoCommand
from core.structures import Keyframe
import copy


class MoveKeyframeCommand(QUndoCommand):
    def __init__(self, model, index, new_time, old_time):
        # super().__init__(f"Move Keyframe {index}")
        super().__init__('')
        self.model = model
        self.index = index
        self.new_time = new_time
        self.old_time = old_time

    def redo(self):
        # 直接修改内部数据，避免调用会产生 Command 的方法
        # 但要触发信号通知 View
        self.model.keyframes[self.index].output_time = self.new_time
        self.model.keyframes_changed.emit()

    def undo(self):
        self.model.keyframes[self.index].output_time = self.old_time
        self.model.keyframes_changed.emit()


class UpdatePropertyCommand(QUndoCommand):
    """通用属性修改 (Ease, VFX, Params)"""

    def __init__(self, model, property_name, new_value, description="Update Property"):
        super().__init__('')
        self.model = model
        self.property_name = property_name
        self.new_value = new_value

        # 记录旧值 (只记录选中的)
        # key: keyframe_index, value: old_property_value
        self.old_values = {}

        for i, kf in enumerate(self.model.keyframes):
            if kf.selected:
                if property_name == 'effect_params':
                    # 字典需要深拷贝
                    self.old_values[i] = copy.deepcopy(kf.effect_params)
                else:
                    self.old_values[i] = getattr(kf, property_name)

    def redo(self):
        for i in self.old_values.keys():
            kf = self.model.keyframes[i]
            if self.property_name == 'effect_params':
                # 合并更新 params
                # 注意：new_value 是一个部分字典 {'scale': 1.5}
                kf.effect_params.update(self.new_value)
            else:
                setattr(kf, self.property_name, self.new_value)
        self.model.keyframes_changed.emit()

    def undo(self):
        for i, old_val in self.old_values.items():
            kf = self.model.keyframes[i]
            if self.property_name == 'effect_params':
                # 恢复旧字典
                kf.effect_params = copy.deepcopy(old_val)
            else:
                setattr(kf, self.property_name, old_val)
        self.model.keyframes_changed.emit()
