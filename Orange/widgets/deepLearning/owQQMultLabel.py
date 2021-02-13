from Orange.widgets.utils.owtencentAiApi import TencentAPI, ImageMixin


class MultiImageLabel(ImageMixin, TencentAPI):
    name = "多标签识别 (Multi Label Recognition)"
    description = "识别一个图像的标签信息,对图像分类。"
    icon = "icons/multi.png"
    keywords = ['qq', 'tengxun', 'duobianqian', 'tuxiang', 'shibie']
    category = 'deeplearning'

    URL = 'https://api.ai.qq.com/fcgi-bin/image/image_tag'

    def __init__(self):
        super().__init__()

    def get_results(self, res):
        items = res.get('tag_list')
        return items

    def setup_params(self):
        self.params['image'] = self.image_str
