from Orange.widgets.utils.owtencentAiApi import TencentAPI, ImageMixin


class ObjectRecgnition(ImageMixin, TencentAPI):
    name = "物体识别 (Scene Recognition)"
    description = "对图片进行物体识别，快速找出图片中包含的物体信息"
    # icon = "icons/gridworld.png"

    URL = 'https://api.ai.qq.com/fcgi-bin/vision/vision_objectr'

    def __init__(self):
        super().__init__()

    def get_results(self, res):
        items = res.get('scene_list')
        return items

    def setup_params(self):
        self.params['image'] = self.image_str
        self.params['topk'] = 5
        self.params['format'] = 1
