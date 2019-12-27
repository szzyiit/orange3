from Orange.widgets.utils.owtencentAiApi import TencentAPI, ImageMixin


class FaceRecgnition(ImageMixin, TencentAPI):
    name = "人脸识别 (Scene Recognition)"
    description = "对图片进行物体识别，快速找出图片中包含的物体信息"
    # icon = "icons/gridworld.png"

    URL = 'https://api.ai.qq.com/fcgi-bin/face/face_newperson'

    def __init__(self):
        super().__init__()

    def get_results(self, res):
        items = res.get('group_ids')
        return items

    def setup_params(self):
        self.params['image'] = self.image_str
        self.params['group_ids'] = 'group0'
        self.params['person_id'] = 'person0'
        self.params['person_name'] = 'name0'
        self.params['tag'] = 'tag0'
