import colorsys
import warnings
from typing import Sequence

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QPixmap, QColor, QIcon

from Orange.util import Enum, hex_to_color, color_to_hex

NAN_COLOR = (128, 128, 128)

__all__ = ["Palette", "IndexedPalette",
           "DiscretePalette", "LimitedDiscretePalette", "DiscretePalettes",
           "DefaultDiscretePalette", "DefaultDiscretePaletteName",
           "DefaultRGBColors", "Dark2Colors",
           "ContinuousPalette", "ContinuousPalettes", "BinnedContinuousPalette",
           "DefaultContinuousPalette", "DefaultContinuousPaletteName",
           "ColorIcon", "get_default_curve_colors", "patch_variable_colors",
           "NAN_COLOR"]


class Palette:
    """
    Base class for enumerable named categorized palettes used for visualization
    of discrete and numeric data

    Attributes:
        name (str): unique name
        friendly_name (str): name to be shown in user interfaces
        category (str): category for user interfaces
        palette (np.ndarray): palette; an array of shape (n, 3)
        nan_color (np.ndarray): an array of shape (1, 3) storing the colors used
            for missing values
        flags (Palette.Flags): flags describing palettes properties
            - ColorBlindSafe: palette is color-blind safe
            - Diverging: palette passes through some neutral color (white,
              black) which appears in the middle. For binned continuous
              palettes the pure neutral color does not need to appear in a bin
            - Discrete: palette contains a small number of colors, like
              palettes for discrete values and binned palettes
    """
    Flags = Enum("PaletteFlags",
                 dict(NoFlags=0,
                      ColorBlindSafe=1,
                      Diverging=2,
                      Discrete=4),
                 type=int,
                 qualname="Palette.Flags")
    NoFlags, ColorBlindSafe, Diverging, Discrete = Flags

    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=0):
        self.name = name
        self.friendly_name = friendly_name
        self.category = category or name.split("_")[0].title()
        self.palette = np.array(palette).astype(np.ubyte)
        self.nan_color = nan_color
        self.flags = flags

    # qcolors and qcolor_w_nan must not be cached because QColor is mutable
    # and may be modified by the caller (and is, in many widgets)
    @property
    def qcolors(self):
        """An array of QColors in the palette"""
        return np.array([QColor(*col) for col in self.palette])

    @property
    def qcolors_w_nan(self):
        """An array of QColors in the palette + the color for nan values"""
        return np.array([QColor(*col) for col in self.palette]
                        + [QColor(*self.nan_color)])

    def copy(self):
        return type(self)(self.friendly_name, self.name, self.palette.copy(),
                          self.nan_color,
                          category=self.category, flags=self.flags)


class IndexedPalette(Palette):
    def __len__(self):
        return len(self.palette)

    def __getitem__(self, x):
        if isinstance(x, (Sequence, np.ndarray)):
            return self.values_to_qcolors(x)
        elif isinstance(x, slice):
            return [QColor(*col) for col in self.palette.__getitem__(x)]
        else:
            return self.value_to_qcolor(x)


class DiscretePalette(IndexedPalette):
    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete):
        super().__init__(friendly_name, name, palette, nan_color,
                         category=category, flags=flags)

    @classmethod
    def from_colors(cls, palette):
        """
        Create a palette from an (n x 3) array of ints in range (0, 255)
        """
        return cls("Custom", "Custom", palette)

    @staticmethod
    def _color_indices(x):
        x = np.asanyarray(x)
        nans = np.isnan(x)
        if np.any(nans):
            x = x.copy()
            x[nans] = -1
        return x.astype(int), nans

    def values_to_colors(self, x):
        """
        Map values x to colors; values may include nan's

        Args:
            x (np.ndarray): an array of values between 0 and len(palette) - 1

        Returns:
            An array of ubytes of shape (len(x), 3), representing RGB triplets
        """
        x, nans = self._color_indices(x)
        colors = self.palette[x]
        colors[nans, :] = self.nan_color
        return colors

    def values_to_qcolors(self, x):
        """
        Map values x to QColors; values may include nan's

        Args:
            x (np.ndarray): an array of values between 0 and len(palette) - 1

        Returns:
            An array of len(x) QColors
        """
        x, _ = self._color_indices(x)
        return self.qcolors_w_nan[x]

    def value_to_color(self, x):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        if np.isnan(x):
            return self.nan_color
        return self.palette[int(x)]

    def value_to_qcolor(self, x):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        color = self.nan_color if np.isnan(x) else self.palette[int(x)]
        return QColor(*color)


class LimitedDiscretePalette(DiscretePalette):
    """
    A palette derived from DiscretePalette that has the prescribed number of
    colors.

    Colors are taken from DefaultRGBColors (the default discrete palette),
    unless the desired number of colors is too large. In this case, colors
    are generated by making a circle around the HSV space.
    """
    def __init__(self, number_of_colors, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete, force_hsv=False):
        if number_of_colors <= len(DefaultRGBColors) and not force_hsv:
            palette = DefaultRGBColors.palette[:number_of_colors]
        else:
            hues = np.linspace(0, 1, number_of_colors, endpoint=False)
            palette = 255 * np.array(
                [colorsys.hsv_to_rgb(h, 1, 1) for h in hues])
        super().__init__("custom", "custom", palette, nan_color,
                         category=category, flags=flags)


class ContinuousPalette(Palette):
    """
    Palette for continuous values
    """
    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.NoFlags):
        super().__init__(
            friendly_name, name,
            np.asarray(palette, dtype=np.ubyte), nan_color,
            category=category, flags=flags)

    @staticmethod
    def _color_indices(x, low=None, high=None):
        x = np.asarray(x)
        nans = np.isnan(x)
        if np.all(nans):
            return np.full(len(x), -1), nans

        if low is None:
            low = np.nanmin(x)
        if high is None:
            high = np.nanmax(x)
        diff = high - low
        if diff == 0:
            x = np.full(len(x), 128)
        else:
            x = 255 * (x - low) / diff
            x = np.clip(x, 0, 255)
        x[nans] = -1
        return np.round(x).astype(int), nans

    def values_to_colors(self, x, low=None, high=None):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped
            low (float or None): minimal value; if None, determined from data
            high (float or None): maximal value; if None, determined from data

        Returns:
            an array of shape (len(x), 3) with RGB values for each point
        """
        x, nans = self._color_indices(x, low, high)
        colors = self.palette[x]
        colors[nans, :] = self.nan_color
        return colors

    def values_to_qcolors(self, x, low=None, high=None):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped
            low (float or None): minimal value; if None, determined from data
            high (float or None): maximal value; if None, determined from data

        Returns:
            an array of shape (len(x), ) with QColors for each point
        """
        x, _ = self._color_indices(x, low, high)
        return self.qcolors_w_nan[x]

    @staticmethod
    def _color_index(x, low=0, high=1):
        if np.isnan(x):
            return -1
        diff = high - low
        if diff == 0:
            return 128
        return int(np.clip(np.round(255 * (x - low) / diff), 0, 255))

    def value_to_color(self, x, low=0, high=1):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        x = self._color_index(x, low, high)
        if x == -1:
            return NAN_COLOR
        return self.palette[x]

    def value_to_qcolor(self, x, low=0, high=1):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        if np.isnan(x):
            color = self.nan_color
        else:
            x = self._color_index(x, low, high)
            color = self.palette[x]
        return QColor(*color)

    __getitem__ = value_to_qcolor

    def lookup_table(self, low=None, high=None):
        """
        A lookup table for this pallette.

        Arguments `low` and `high` (between 0 and 255) can be used to use
        just a part of palette.

        Args:
            low (float or None): minimal value
            high (float or None): maximal value

        Returns:
            an array of shape (255, 3) with RGB values
        """
        return self.values_to_colors(np.arange(256) / 256, low, high)

    def color_strip(self, strip_length, strip_width, orientation=Qt.Horizontal):
        """
        Return a pixmap of given dimensions to be used for legends.

        Args:
            strip_length (int): length of the strip; may be horizontal or vertical
            strip_width (int): width of the strip
            orientation: strip orientation

        Returns:
            QPixmap with a strip
        """
        points = np.linspace(0, 255, strip_length, dtype=np.uint8)
        section = self.palette[np.newaxis, points, :].astype(np.ubyte)
        img = np.vstack((section,) * strip_width)
        if orientation == Qt.Horizontal:
            width, height = strip_length, strip_width
        else:
            width, height = strip_width, strip_length
            img = img.swapaxes(0, 1)[::-1].copy()
        pad_width = (-img.strides[1]) % 4
        if pad_width:
            padding = np.zeros((img.shape[0], pad_width, 3), np.ubyte)
            img = np.hstack((img, padding))
        img = QImage(img, width, height, img.strides[0], QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        return img

    @classmethod
    def from_colors(cls, color1, color2, pass_through=False):
        """
        Deprecated constructor for tests and easier migration from
        Variable.color. Constructs a palette that goes from color1 to color2.

        pass_throug can be a color through which the palette will pass,
        or `True` to pass through black. Default is `False`.
        """
        if pass_through is True:
            colors = (color1, (0, 0, 0), color2)
            xf = [0, 127, 255]
        elif pass_through:
            assert isinstance(pass_through, tuple)
            colors = (color1, pass_through, color2)
            xf = [0, 127, 255]
        else:
            colors = (color1, color2)
            xf = [0, 255]
        name = repr(colors)
        friendly_name = "Custom"
        x = np.arange(256)
        r = np.interp(x, xf, np.array([color[0] for color in colors]))
        g = np.interp(x, xf, np.array([color[1] for color in colors]))
        b = np.interp(x, xf, np.array([color[2] for color in colors]))
        palette = np.vstack((r, g, b)).T
        return cls(friendly_name, name, palette,
                   flags=Palette.Diverging if pass_through else Palette.NoFlags)


class BinnedContinuousPalette(IndexedPalette):
    """
    Continuous palettes that bins values.

    Besides the derived attributes, it has an attribute `bins` (np.ndarray),
    which contains bin boundaries, including the lower and the higher
    boundary, which are essentially ignored.
    """
    def __init__(self, friendly_name, name, bin_colors, bins,
                 nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete):
        super().__init__(friendly_name, name, bin_colors, nan_color,
                         category=category, flags=flags)
        self.bins = bins

    @classmethod
    def from_palette(cls, palette, bins):
        """
        Construct a `BinnedPalette` from `ContinuousPalette` by picking the
        colors representing the centers of the bins.

        If given a `BinnedPalette`, the constructor returns a copy.

        Args:
            palette (ContinuousPalette or BinnedPalette): original palette
            bins (np.ndarray): bin boundaries
        """
        if isinstance(palette, cls):
            # Note that this silently ignores bins. This is done on purpose
            # to let predefined binned palettes override bins. Plus, it is
            # generally impossible to compute a binned palette with different
            # bins.
            return palette.copy()
        if isinstance(palette, ContinuousPalette):
            assert len(bins) >= 2
            mids = (bins[:-1] + bins[1:]) / 2
            bin_colors = palette.values_to_colors(mids, bins[0], bins[-1])
            return cls(
                palette.friendly_name, palette.name, bin_colors, bins,
                palette.nan_color, category=palette.category,
                flags=palette.flags | Palette.Discrete)
        raise TypeError(f"can't create palette from '{type(palette).__name__}'")

    def _bin_indices(self, x):
        nans = np.isnan(x)
        binx = np.digitize(x, self.bins[1:-1], right=True)
        binx.clip(0, len(self.bins) - 1)
        binx[nans] = -1
        return binx, nans

    def values_to_colors(self, x):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped

        Returns:
            an array of shape (len(x), 3) with RGB values for each point
        """

        binx, nans = self._bin_indices(x)
        colors = self.palette[binx]
        colors[nans] = self.nan_color
        return colors

    def values_to_qcolors(self, x):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped

        Returns:
            an array of shape (len(x), ) with QColors for each point
        """
        binx, _ = self._bin_indices(x)
        return self.qcolors_w_nan[binx]

    def value_to_color(self, x):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        return self.values_to_colors([x])[0]

    def value_to_qcolor(self, x):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        if np.isnan(x):
            color = self.nan_color
        else:
            binx, _ = self._bin_indices([x])
            color = self.palette[binx[0]]
        return QColor(*color)

    def copy(self):
        return type(self)(self.friendly_name, self.name, self.palette.copy(),
                          self.bins.copy(), self.nan_color,
                          category=self.category, flags=self.flags)


DefaultRGBColors = DiscretePalette("Default", "Default", [
    [70, 190, 250], [237, 70, 47], [170, 242, 43], [245, 174, 50],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 255],
    [0, 128, 255], [255, 223, 128], [127, 111, 64], [92, 46, 0],
    [0, 84, 0], [192, 192, 0], [0, 127, 127], [128, 0, 0],
    [127, 0, 127]])

Dark2Colors = DiscretePalette("Dark", "Dark", [
    [27, 158, 119], [217, 95, 2], [117, 112, 179], [231, 41, 138],
    [102, 166, 30], [230, 171, 2], [166, 118, 29], [102, 102, 102]])

DiscretePalettes = {
    "default": DefaultRGBColors,
    "dark": Dark2Colors
}

DefaultDiscretePaletteName = "default"
DefaultDiscretePalette = DiscretePalettes[DefaultDiscretePaletteName]


# pylint: disable=line-too-long
ContinuousPalettes = {
    'diverging_bwr_40_95_c42': ContinuousPalette(
        'Coolwarm', 'diverging_bwr_40_95_c42',
        [[33, 81, 219], [37, 82, 219], [42, 83, 219], [46, 84, 220], [49, 85, 220], [53, 86, 220], [56, 87, 220], [59, 88, 220], [62, 89, 221], [65, 91, 221], [67, 92, 221], [70, 93, 221], [72, 94, 221], [75, 95, 222], [77, 96, 222], [80, 97, 222], [82, 99, 222], [84, 100, 223], [86, 101, 223], [88, 102, 223], [90, 103, 223], [92, 104, 223], [94, 105, 224], [96, 107, 224], [98, 108, 224], [100, 109, 224], [102, 110, 224], [104, 111, 225], [105, 112, 225], [107, 114, 225], [109, 115, 225], [111, 116, 225], [112, 117, 226], [114, 118, 226], [116, 120, 226], [117, 121, 226], [119, 122, 226], [121, 123, 227], [122, 124, 227], [124, 126, 227], [125, 127, 227], [127, 128, 227], [129, 129, 228], [130, 130, 228], [132, 132, 228], [133, 133, 228], [135, 134, 228], [136, 135, 229], [138, 136, 229], [139, 138, 229], [141, 139, 229], [142, 140, 229], [143, 141, 229], [145, 143, 230], [146, 144, 230], [148, 145, 230], [149, 146, 230], [151, 148, 230], [152, 149, 231], [153, 150, 231], [155, 151, 231], [156, 153, 231], [158, 154, 231], [159, 155, 231], [160, 156, 232], [162, 158, 232], [163, 159, 232], [164, 160, 232], [166, 161, 232], [167, 163, 232], [168, 164, 233], [170, 165, 233], [171, 166, 233], [172, 168, 233], [174, 169, 233], [175, 170, 233], [176, 172, 234], [178, 173, 234], [179, 174, 234], [180, 175, 234], [181, 177, 234], [183, 178, 234], [184, 179, 235], [185, 180, 235], [187, 182, 235], [188, 183, 235], [189, 184, 235], [190, 186, 235], [192, 187, 235], [193, 188, 236], [194, 190, 236], [195, 191, 236], [197, 192, 236], [198, 193, 236], [199, 195, 236], [200, 196, 236], [202, 197, 237], [203, 199, 237], [204, 200, 237], [205, 201, 237], [207, 203, 237], [208, 204, 237], [209, 205, 237], [210, 207, 238], [211, 208, 238], [213, 209, 238], [214, 211, 238], [215, 212, 238], [216, 213, 238], [218, 215, 238], [219, 216, 238], [220, 217, 239], [221, 218, 239], [222, 220, 239], [224, 221, 239], [225, 222, 239], [226, 224, 239], [227, 225, 239], [228, 226, 239], [229, 227, 239], [230, 228, 238], [231, 229, 238], [232, 229, 238], [233, 230, 237], [234, 231, 237], [235, 231, 236], [236, 231, 235], [237, 231, 234], [238, 231, 233], [238, 231, 232], [239, 230, 231], [239, 230, 230], [240, 229, 228], [240, 228, 227], [241, 227, 225], [241, 226, 223], [241, 225, 221], [241, 224, 220], [241, 222, 218], [242, 221, 216], [242, 219, 214], [242, 218, 212], [242, 216, 210], [242, 215, 208], [242, 213, 206], [242, 212, 204], [242, 210, 203], [242, 209, 201], [242, 207, 199], [242, 206, 197], [242, 204, 195], [241, 203, 193], [241, 201, 191], [241, 199, 189], [241, 198, 187], [241, 196, 185], [241, 195, 183], [241, 193, 181], [241, 192, 180], [240, 190, 178], [240, 189, 176], [240, 187, 174], [240, 185, 172], [240, 184, 170], [240, 182, 168], [239, 181, 166], [239, 179, 165], [239, 178, 163], [239, 176, 161], [238, 175, 159], [238, 173, 157], [238, 172, 155], [238, 170, 153], [237, 168, 152], [237, 167, 150], [237, 165, 148], [237, 164, 146], [236, 162, 144], [236, 161, 142], [236, 159, 140], [235, 157, 139], [235, 156, 137], [235, 154, 135], [234, 153, 133], [234, 151, 131], [234, 150, 130], [233, 148, 128], [233, 147, 126], [232, 145, 124], [232, 143, 122], [232, 142, 121], [231, 140, 119], [231, 139, 117], [230, 137, 115], [230, 136, 113], [230, 134, 112], [229, 132, 110], [229, 131, 108], [228, 129, 106], [228, 128, 105], [227, 126, 103], [227, 124, 101], [226, 123, 99], [226, 121, 98], [225, 120, 96], [225, 118, 94], [224, 116, 92], [224, 115, 91], [223, 113, 89], [223, 111, 87], [222, 110, 85], [222, 108, 84], [221, 106, 82], [221, 105, 80], [220, 103, 79], [219, 102, 77], [219, 100, 75], [218, 98, 73], [218, 96, 72], [217, 95, 70], [216, 93, 68], [216, 91, 67], [215, 90, 65], [215, 88, 63], [214, 86, 62], [213, 84, 60], [213, 82, 58], [212, 81, 56], [211, 79, 55], [211, 77, 53], [210, 75, 51], [209, 73, 50], [209, 71, 48], [208, 69, 46], [207, 68, 45], [207, 66, 43], [206, 64, 41], [205, 61, 40], [205, 59, 38], [204, 57, 36], [203, 55, 34], [202, 53, 33], [202, 51, 31], [201, 48, 29], [200, 46, 27], [200, 43, 26], [199, 41, 24], [198, 38, 22], [197, 35, 20], [197, 32, 18], [196, 28, 16], [195, 25, 14], [194, 20, 12], [193, 15, 10], [193, 9, 8], [192, 2, 6]],
        flags=Palette.Diverging
    ),
    'diverging_gkr_60_10_c40': ContinuousPalette(
        'Green-Red', 'diverging_gkr_60_10_c40',
        [[54, 166, 22], [54, 165, 23], [54, 164, 23], [54, 162, 24], [54, 161, 24], [54, 160, 24], [54, 159, 25], [54, 158, 25], [54, 157, 26], [54, 155, 26], [54, 154, 26], [54, 153, 27], [54, 152, 27], [54, 151, 27], [54, 149, 27], [54, 148, 28], [54, 147, 28], [54, 146, 28], [54, 145, 28], [54, 144, 29], [54, 142, 29], [54, 141, 29], [54, 140, 29], [54, 139, 30], [54, 138, 30], [54, 137, 30], [54, 135, 30], [54, 134, 30], [53, 133, 30], [53, 132, 31], [53, 131, 31], [53, 130, 31], [53, 129, 31], [53, 127, 31], [53, 126, 31], [53, 125, 32], [53, 124, 32], [53, 123, 32], [53, 122, 32], [52, 121, 32], [52, 119, 32], [52, 118, 32], [52, 117, 32], [52, 116, 32], [52, 115, 32], [52, 114, 33], [51, 113, 33], [51, 112, 33], [51, 110, 33], [51, 109, 33], [51, 108, 33], [51, 107, 33], [51, 106, 33], [50, 105, 33], [50, 104, 33], [50, 103, 33], [50, 102, 33], [50, 100, 33], [49, 99, 33], [49, 98, 33], [49, 97, 33], [49, 96, 33], [49, 95, 33], [48, 94, 33], [48, 93, 33], [48, 92, 33], [48, 91, 33], [48, 90, 33], [47, 88, 33], [47, 87, 33], [47, 86, 33], [47, 85, 33], [46, 84, 33], [46, 83, 33], [46, 82, 33], [46, 81, 33], [45, 80, 33], [45, 79, 33], [45, 78, 33], [45, 77, 33], [44, 76, 33], [44, 75, 33], [44, 74, 33], [44, 73, 33], [43, 71, 33], [43, 70, 33], [43, 69, 33], [42, 68, 33], [42, 67, 32], [42, 66, 32], [42, 65, 32], [41, 64, 32], [41, 63, 32], [41, 62, 32], [40, 61, 32], [40, 60, 32], [40, 59, 32], [39, 58, 32], [39, 57, 32], [39, 56, 31], [38, 55, 31], [38, 54, 31], [38, 53, 31], [37, 52, 31], [37, 51, 31], [37, 50, 31], [36, 49, 31], [36, 48, 31], [36, 47, 30], [35, 46, 30], [35, 45, 30], [35, 44, 30], [34, 43, 30], [34, 42, 30], [34, 41, 30], [33, 40, 30], [33, 39, 29], [33, 39, 29], [32, 38, 29], [32, 37, 29], [32, 36, 29], [32, 35, 29], [32, 35, 29], [32, 34, 29], [33, 34, 29], [33, 33, 29], [33, 33, 29], [34, 32, 29], [35, 32, 29], [35, 32, 29], [36, 32, 29], [37, 32, 29], [38, 32, 29], [40, 32, 29], [41, 33, 30], [42, 33, 30], [44, 33, 30], [45, 34, 30], [47, 34, 30], [48, 34, 31], [50, 35, 31], [51, 35, 31], [53, 36, 31], [54, 36, 32], [56, 37, 32], [58, 37, 32], [59, 38, 32], [61, 38, 33], [62, 39, 33], [64, 39, 33], [66, 40, 33], [67, 40, 34], [69, 41, 34], [71, 41, 34], [72, 42, 34], [74, 42, 34], [75, 43, 35], [77, 43, 35], [79, 44, 35], [80, 45, 35], [82, 45, 36], [84, 46, 36], [85, 46, 36], [87, 46, 36], [89, 47, 37], [90, 47, 37], [92, 48, 37], [94, 48, 37], [95, 49, 38], [97, 49, 38], [99, 50, 38], [100, 50, 38], [102, 51, 39], [104, 51, 39], [106, 52, 39], [107, 52, 39], [109, 53, 39], [111, 53, 40], [112, 54, 40], [114, 54, 40], [116, 55, 40], [117, 55, 41], [119, 56, 41], [121, 56, 41], [123, 57, 41], [124, 57, 42], [126, 58, 42], [128, 58, 42], [130, 59, 42], [131, 59, 42], [133, 59, 43], [135, 60, 43], [137, 60, 43], [138, 61, 43], [140, 61, 44], [142, 62, 44], [144, 62, 44], [145, 63, 44], [147, 63, 44], [149, 64, 45], [151, 64, 45], [152, 65, 45], [154, 65, 45], [156, 65, 46], [158, 66, 46], [160, 66, 46], [161, 67, 46], [163, 67, 46], [165, 68, 47], [167, 68, 47], [169, 69, 47], [170, 69, 47], [172, 70, 47], [174, 70, 48], [176, 70, 48], [178, 71, 48], [180, 71, 48], [181, 72, 49], [183, 72, 49], [185, 73, 49], [187, 73, 49], [189, 73, 49], [191, 74, 50], [192, 74, 50], [194, 75, 50], [196, 75, 50], [198, 76, 50], [200, 76, 51], [202, 76, 51], [204, 77, 51], [205, 77, 51], [207, 78, 52], [209, 78, 52], [211, 79, 52], [213, 79, 52], [215, 79, 52], [217, 80, 53], [219, 80, 53], [220, 81, 53], [222, 81, 53], [224, 82, 53], [226, 82, 54], [228, 82, 54], [230, 83, 54], [232, 83, 54], [234, 84, 54], [236, 84, 55], [238, 85, 55], [240, 85, 55], [241, 85, 55], [243, 86, 55], [245, 86, 56], [247, 87, 56], [249, 87, 56], [251, 87, 56], [253, 88, 56]],
        flags=Palette.Diverging
    ),
    'linear_bgyw_15_100_c68': ContinuousPalette(
        'Blue-Green-Yellow', 'linear_bgyw_15_100_c68',
        [[26, 0, 134], [26, 0, 136], [26, 0, 137], [26, 0, 139], [26, 1, 141], [26, 1, 143], [26, 2, 145], [26, 3, 146], [26, 4, 148], [26, 4, 150], [26, 5, 152], [27, 6, 153], [27, 7, 155], [27, 8, 157], [27, 9, 158], [27, 10, 160], [27, 11, 162], [27, 12, 164], [27, 13, 165], [27, 14, 167], [27, 15, 168], [27, 16, 170], [27, 17, 172], [27, 19, 173], [27, 20, 175], [27, 21, 176], [27, 22, 178], [27, 23, 180], [27, 24, 181], [27, 25, 183], [27, 26, 184], [27, 27, 186], [27, 28, 187], [27, 29, 188], [27, 30, 190], [27, 31, 191], [27, 32, 193], [27, 33, 194], [27, 34, 195], [27, 35, 197], [27, 36, 198], [27, 37, 199], [27, 39, 200], [27, 40, 201], [28, 41, 203], [28, 42, 204], [28, 43, 205], [28, 44, 206], [28, 45, 207], [28, 47, 208], [28, 48, 209], [29, 49, 210], [29, 50, 210], [29, 52, 211], [29, 53, 212], [29, 54, 213], [30, 55, 213], [30, 57, 214], [30, 58, 214], [30, 59, 215], [31, 61, 215], [31, 62, 215], [31, 64, 215], [31, 65, 215], [32, 67, 215], [32, 68, 215], [32, 70, 215], [32, 71, 214], [33, 73, 214], [33, 75, 213], [33, 76, 212], [33, 78, 211], [33, 80, 209], [33, 82, 207], [33, 84, 205], [32, 86, 202], [32, 88, 199], [32, 90, 196], [32, 92, 193], [32, 94, 190], [32, 96, 187], [33, 98, 184], [33, 100, 181], [34, 101, 178], [35, 103, 175], [36, 105, 173], [37, 106, 170], [38, 108, 167], [39, 109, 164], [40, 111, 161], [41, 112, 159], [42, 113, 156], [43, 115, 153], [44, 116, 150], [45, 118, 148], [46, 119, 145], [47, 120, 142], [47, 121, 140], [48, 123, 137], [49, 124, 134], [50, 125, 132], [51, 126, 129], [52, 128, 126], [53, 129, 124], [54, 130, 121], [55, 131, 118], [55, 132, 116], [56, 134, 113], [57, 135, 110], [58, 136, 108], [59, 137, 105], [59, 138, 102], [60, 139, 100], [61, 140, 97], [62, 142, 94], [62, 143, 92], [63, 144, 89], [64, 145, 87], [65, 146, 85], [66, 147, 83], [67, 148, 80], [68, 149, 78], [68, 150, 76], [69, 151, 74], [70, 152, 72], [72, 153, 70], [73, 154, 68], [74, 155, 66], [75, 156, 65], [76, 157, 63], [77, 158, 61], [79, 159, 60], [80, 160, 58], [81, 161, 56], [83, 161, 55], [84, 162, 53], [86, 163, 52], [87, 164, 50], [89, 165, 49], [90, 166, 47], [92, 167, 46], [93, 168, 44], [95, 168, 43], [97, 169, 42], [98, 170, 41], [100, 171, 39], [102, 172, 38], [103, 173, 37], [105, 173, 36], [107, 174, 34], [109, 175, 33], [111, 176, 32], [113, 177, 31], [115, 177, 30], [116, 178, 29], [118, 179, 28], [120, 180, 28], [122, 180, 27], [124, 181, 26], [126, 182, 25], [128, 183, 25], [130, 183, 24], [132, 184, 23], [134, 185, 23], [136, 186, 23], [139, 186, 22], [141, 187, 22], [143, 188, 22], [145, 188, 22], [147, 189, 22], [149, 190, 22], [151, 190, 22], [154, 191, 23], [156, 192, 23], [158, 192, 23], [160, 193, 24], [162, 194, 24], [164, 194, 25], [166, 195, 25], [169, 196, 26], [171, 196, 26], [173, 197, 27], [175, 198, 28], [177, 198, 28], [179, 199, 29], [181, 200, 30], [183, 200, 31], [185, 201, 32], [187, 202, 33], [189, 202, 33], [191, 203, 34], [192, 204, 35], [194, 204, 36], [196, 205, 37], [198, 206, 38], [200, 206, 40], [202, 207, 41], [203, 208, 42], [205, 208, 43], [207, 209, 44], [209, 210, 46], [210, 210, 47], [212, 211, 48], [214, 212, 50], [215, 213, 51], [217, 213, 53], [218, 214, 54], [220, 215, 56], [222, 216, 57], [223, 216, 59], [224, 217, 61], [226, 218, 63], [227, 219, 64], [229, 220, 66], [230, 220, 68], [231, 221, 70], [232, 222, 72], [233, 223, 74], [234, 224, 77], [235, 225, 79], [236, 226, 82], [237, 227, 84], [238, 228, 87], [238, 229, 90], [239, 230, 93], [239, 231, 96], [239, 232, 99], [240, 233, 103], [240, 234, 106], [240, 235, 110], [241, 236, 114], [241, 237, 118], [242, 238, 122], [242, 239, 126], [243, 240, 131], [243, 241, 135], [244, 242, 140], [245, 243, 145], [245, 244, 150], [246, 244, 155], [247, 245, 160], [247, 246, 165], [248, 247, 171], [249, 247, 176], [249, 248, 182], [250, 249, 188], [250, 250, 194], [251, 250, 200], [252, 251, 207], [252, 252, 213], [253, 252, 220], [253, 253, 226], [254, 253, 233], [254, 254, 240], [255, 254, 248], [255, 255, 255]],
    ),
    'linear_bmy_10_95_c78': ContinuousPalette(
        'Blue-Magenta-Yellow', 'linear_bmy_10_95_c78',
        [[0, 12, 125], [0, 13, 126], [0, 13, 128], [0, 14, 130], [0, 14, 132], [0, 14, 134], [0, 15, 135], [0, 15, 137], [0, 16, 139], [0, 16, 140], [0, 17, 142], [0, 17, 144], [0, 17, 145], [0, 18, 147], [0, 18, 148], [0, 18, 150], [0, 19, 151], [0, 19, 153], [0, 19, 154], [0, 20, 155], [0, 20, 157], [0, 20, 158], [0, 20, 159], [0, 21, 160], [0, 21, 161], [0, 21, 162], [0, 21, 163], [0, 21, 164], [0, 21, 165], [0, 22, 166], [0, 22, 167], [0, 22, 167], [0, 22, 168], [0, 22, 169], [0, 22, 169], [0, 22, 169], [10, 22, 170], [21, 22, 170], [29, 21, 170], [35, 21, 170], [41, 21, 170], [47, 21, 169], [52, 20, 169], [57, 20, 168], [62, 19, 167], [67, 19, 166], [71, 18, 165], [76, 18, 164], [80, 17, 163], [83, 17, 162], [87, 16, 161], [90, 15, 160], [94, 15, 159], [97, 14, 159], [100, 14, 158], [103, 13, 157], [106, 13, 156], [108, 12, 155], [111, 11, 154], [114, 11, 153], [116, 10, 153], [119, 10, 152], [121, 9, 151], [124, 8, 151], [126, 8, 150], [128, 7, 149], [130, 7, 149], [133, 6, 148], [135, 6, 147], [137, 6, 147], [139, 5, 146], [141, 5, 146], [143, 4, 145], [145, 4, 145], [147, 4, 144], [149, 3, 144], [151, 3, 143], [153, 3, 143], [155, 2, 142], [157, 2, 142], [159, 2, 142], [161, 1, 141], [163, 1, 141], [165, 1, 140], [167, 1, 140], [169, 0, 139], [171, 0, 139], [172, 0, 138], [174, 0, 138], [176, 0, 137], [178, 0, 137], [180, 0, 136], [182, 0, 136], [184, 0, 135], [185, 0, 135], [187, 0, 135], [189, 0, 134], [191, 0, 134], [193, 0, 133], [195, 0, 133], [196, 0, 132], [198, 0, 132], [200, 0, 131], [202, 0, 131], [204, 0, 130], [206, 0, 130], [207, 0, 129], [209, 0, 129], [211, 0, 128], [213, 0, 128], [214, 0, 127], [216, 0, 127], [218, 0, 126], [219, 0, 126], [221, 0, 125], [222, 0, 124], [224, 1, 124], [226, 2, 123], [227, 4, 123], [229, 6, 122], [230, 8, 122], [232, 11, 121], [233, 13, 120], [234, 16, 120], [236, 18, 119], [237, 20, 119], [238, 22, 118], [240, 24, 117], [241, 26, 117], [242, 28, 116], [244, 30, 115], [245, 32, 115], [246, 34, 114], [247, 36, 113], [248, 38, 113], [249, 40, 112], [251, 42, 111], [252, 44, 111], [253, 46, 110], [254, 48, 109], [255, 50, 108], [255, 52, 108], [255, 54, 107], [255, 56, 106], [255, 58, 106], [255, 61, 105], [255, 63, 104], [255, 65, 103], [255, 67, 102], [255, 69, 102], [255, 71, 101], [255, 73, 100], [255, 75, 99], [255, 77, 98], [255, 80, 98], [255, 82, 97], [255, 84, 96], [255, 86, 95], [255, 88, 94], [255, 90, 93], [255, 93, 92], [255, 95, 91], [255, 97, 91], [255, 99, 90], [255, 101, 89], [255, 103, 88], [255, 106, 87], [255, 108, 86], [255, 110, 85], [255, 112, 84], [255, 114, 83], [255, 116, 82], [255, 118, 81], [255, 120, 80], [255, 122, 78], [255, 124, 77], [255, 126, 76], [255, 127, 75], [255, 129, 74], [255, 131, 73], [255, 133, 72], [255, 135, 71], [255, 136, 69], [255, 138, 68], [255, 140, 67], [255, 142, 66], [255, 143, 64], [255, 145, 63], [255, 147, 62], [255, 149, 60], [255, 150, 59], [255, 152, 57], [255, 154, 56], [255, 155, 54], [255, 157, 53], [255, 159, 51], [255, 160, 50], [255, 162, 48], [255, 163, 47], [255, 165, 46], [255, 166, 44], [255, 168, 43], [255, 169, 42], [255, 171, 41], [255, 172, 40], [255, 174, 39], [255, 175, 38], [255, 177, 38], [255, 178, 37], [255, 180, 36], [255, 181, 35], [255, 182, 35], [255, 184, 34], [255, 185, 34], [255, 187, 33], [255, 188, 32], [255, 189, 32], [255, 191, 31], [255, 192, 31], [255, 194, 31], [255, 195, 30], [255, 196, 30], [255, 198, 30], [255, 199, 29], [255, 200, 29], [255, 202, 29], [255, 203, 29], [255, 204, 29], [255, 206, 29], [255, 207, 28], [255, 208, 28], [255, 210, 28], [255, 211, 28], [255, 212, 28], [255, 214, 28], [255, 215, 29], [255, 216, 29], [255, 218, 29], [255, 219, 29], [255, 220, 29], [255, 222, 29], [255, 223, 30], [255, 224, 30], [255, 226, 30], [255, 227, 30], [255, 228, 31], [255, 230, 31], [255, 231, 31], [255, 232, 32], [255, 233, 32], [255, 235, 33], [255, 236, 33], [255, 237, 34], [255, 239, 34], [255, 240, 35], [255, 241, 35]],
    ),
    'linear_grey_10_95_c0': ContinuousPalette(
        'Dim gray', 'linear_grey_10_95_c0',
        [[27, 27, 27], [28, 28, 28], [29, 29, 29], [29, 29, 29], [30, 30, 30], [31, 31, 31], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [34, 34, 34], [35, 35, 35], [36, 36, 36], [36, 36, 36], [37, 37, 37], [38, 38, 38], [38, 38, 38], [39, 39, 39], [40, 40, 40], [40, 40, 40], [41, 41, 41], [42, 42, 42], [43, 43, 43], [43, 43, 43], [44, 44, 44], [45, 45, 45], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [48, 48, 48], [49, 49, 49], [50, 50, 50], [50, 50, 50], [51, 51, 51], [52, 52, 52], [53, 53, 53], [53, 53, 53], [54, 54, 54], [55, 55, 55], [56, 56, 56], [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [59, 59, 59], [60, 60, 60], [61, 61, 61], [62, 62, 62], [62, 62, 62], [63, 63, 63], [64, 64, 64], [65, 65, 65], [65, 65, 65], [66, 66, 66], [67, 67, 67], [68, 68, 68], [68, 68, 68], [69, 69, 69], [70, 70, 70], [71, 71, 71], [71, 71, 71], [72, 72, 72], [73, 73, 73], [74, 74, 74], [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77], [78, 78, 78], [78, 78, 78], [79, 79, 79], [80, 80, 80], [81, 81, 81], [81, 82, 82], [82, 82, 82], [83, 83, 83], [84, 84, 84], [85, 85, 85], [85, 85, 85], [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [89, 89, 89], [90, 90, 90], [91, 91, 91], [92, 92, 92], [93, 93, 93], [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [97, 97, 97], [98, 98, 98], [99, 99, 99], [100, 100, 100], [101, 101, 101], [102, 102, 102], [102, 102, 102], [103, 103, 103], [104, 104, 104], [105, 105, 105], [106, 106, 106], [106, 106, 106], [107, 107, 107], [108, 108, 108], [109, 109, 109], [110, 110, 110], [111, 111, 111], [111, 111, 111], [112, 112, 112], [113, 113, 113], [114, 114, 114], [115, 115, 115], [116, 116, 116], [116, 116, 116], [117, 117, 117], [118, 118, 118], [119, 119, 119], [120, 120, 120], [121, 121, 121], [121, 121, 121], [122, 122, 122], [123, 123, 123], [124, 124, 124], [125, 125, 125], [126, 126, 126], [126, 127, 127], [127, 127, 127], [128, 128, 128], [129, 129, 129], [130, 130, 130], [131, 131, 131], [132, 132, 132], [132, 132, 132], [133, 133, 133], [134, 134, 134], [135, 135, 135], [136, 136, 136], [137, 137, 137], [138, 138, 138], [138, 138, 138], [139, 139, 139], [140, 140, 140], [141, 141, 141], [142, 142, 142], [143, 143, 143], [144, 144, 144], [145, 145, 145], [145, 145, 145], [146, 146, 146], [147, 147, 147], [148, 148, 148], [149, 149, 149], [150, 150, 150], [151, 151, 151], [152, 152, 152], [152, 152, 152], [153, 153, 153], [154, 154, 154], [155, 155, 155], [156, 156, 156], [157, 157, 157], [158, 158, 158], [159, 159, 159], [159, 159, 159], [160, 160, 160], [161, 161, 161], [162, 162, 162], [163, 163, 163], [164, 164, 164], [165, 165, 165], [166, 166, 166], [167, 167, 167], [167, 167, 167], [168, 168, 168], [169, 169, 169], [170, 170, 170], [171, 171, 171], [172, 172, 172], [173, 173, 173], [174, 174, 174], [175, 175, 175], [176, 176, 176], [176, 176, 176], [177, 177, 177], [178, 178, 178], [179, 179, 179], [180, 180, 180], [181, 181, 181], [182, 182, 182], [183, 183, 183], [184, 184, 184], [185, 185, 185], [185, 186, 186], [186, 186, 186], [187, 187, 187], [188, 188, 188], [189, 189, 189], [190, 190, 190], [191, 191, 191], [192, 192, 192], [193, 193, 193], [194, 194, 194], [195, 195, 195], [196, 196, 196], [196, 196, 196], [197, 197, 197], [198, 198, 198], [199, 199, 199], [200, 200, 200], [201, 201, 201], [202, 202, 202], [203, 203, 203], [204, 204, 204], [205, 205, 205], [206, 206, 206], [207, 207, 207], [208, 208, 208], [208, 209, 209], [209, 209, 209], [210, 210, 210], [211, 211, 211], [212, 212, 212], [213, 213, 213], [214, 214, 214], [215, 215, 215], [216, 216, 216], [217, 217, 217], [218, 218, 218], [219, 219, 219], [220, 220, 220], [221, 221, 221], [222, 222, 222], [223, 223, 223], [223, 224, 223], [224, 224, 224], [225, 225, 225], [226, 226, 226], [227, 227, 227], [228, 228, 228], [229, 229, 229], [230, 230, 230], [231, 231, 231], [232, 232, 232], [233, 233, 233], [234, 234, 234], [235, 235, 235], [236, 236, 236], [237, 237, 237], [238, 238, 238], [239, 239, 239], [240, 240, 240], [241, 241, 241]],
    ),
    'linear_kryw_0_100_c71': ContinuousPalette(
        'Fire', 'linear_kryw_0_100_c71',
        [[0, 0, 0], [7, 0, 0], [13, 0, 0], [18, 0, 0], [22, 0, 0], [26, 0, 0], [29, 0, 0], [32, 0, 0], [34, 0, 0], [37, 0, 0], [39, 0, 0], [41, 0, 0], [43, 0, 0], [45, 0, 0], [47, 0, 0], [49, 0, 0], [50, 0, 0], [52, 0, 0], [54, 1, 0], [55, 1, 0], [57, 0, 0], [58, 0, 0], [60, 0, 0], [61, 0, 0], [63, 0, 0], [64, 0, 0], [66, 1, 0], [67, 1, 0], [69, 1, 0], [70, 1, 0], [72, 1, 0], [73, 1, 0], [75, 1, 0], [76, 1, 0], [78, 1, 0], [79, 1, 0], [81, 1, 0], [82, 1, 0], [84, 1, 0], [85, 1, 0], [87, 1, 0], [88, 1, 0], [90, 1, 0], [92, 1, 0], [93, 1, 0], [95, 1, 0], [96, 1, 0], [98, 1, 0], [99, 1, 0], [101, 1, 0], [103, 1, 0], [104, 1, 0], [106, 1, 0], [107, 2, 0], [109, 2, 0], [111, 2, 0], [112, 2, 0], [114, 2, 0], [116, 2, 0], [117, 2, 0], [119, 2, 0], [121, 2, 0], [122, 2, 0], [124, 2, 0], [126, 2, 0], [127, 2, 0], [129, 2, 0], [131, 2, 0], [132, 3, 0], [134, 3, 0], [136, 3, 0], [137, 3, 0], [139, 3, 0], [141, 3, 0], [142, 3, 0], [144, 3, 0], [146, 3, 0], [147, 3, 0], [149, 3, 0], [151, 4, 0], [153, 4, 0], [154, 4, 0], [156, 4, 0], [158, 4, 0], [159, 4, 0], [161, 4, 0], [163, 4, 0], [165, 5, 0], [166, 5, 0], [168, 5, 0], [170, 5, 0], [172, 5, 0], [173, 5, 0], [175, 5, 0], [177, 6, 0], [179, 6, 0], [180, 6, 0], [182, 6, 0], [184, 6, 0], [186, 7, 0], [188, 7, 0], [189, 7, 0], [191, 7, 0], [193, 7, 0], [195, 8, 0], [197, 8, 0], [198, 8, 0], [200, 8, 0], [202, 9, 0], [204, 9, 0], [206, 9, 0], [207, 9, 0], [209, 10, 0], [211, 10, 0], [213, 11, 0], [215, 11, 0], [216, 11, 0], [218, 12, 0], [220, 12, 0], [222, 13, 0], [224, 13, 0], [225, 14, 0], [227, 14, 0], [229, 15, 0], [231, 16, 0], [232, 17, 0], [234, 18, 0], [236, 19, 0], [237, 21, 0], [239, 23, 0], [240, 25, 0], [242, 27, 0], [243, 30, 0], [244, 32, 0], [245, 35, 0], [246, 38, 0], [247, 41, 0], [248, 44, 0], [248, 47, 0], [249, 50, 0], [250, 53, 0], [250, 56, 0], [251, 59, 0], [251, 62, 0], [252, 65, 0], [252, 68, 0], [252, 70, 0], [253, 73, 0], [253, 76, 0], [253, 78, 0], [253, 81, 0], [254, 84, 0], [254, 86, 0], [254, 89, 0], [254, 91, 0], [254, 93, 0], [254, 96, 0], [254, 98, 0], [254, 100, 0], [255, 103, 0], [255, 105, 0], [255, 107, 0], [255, 109, 0], [255, 111, 0], [255, 114, 0], [255, 116, 0], [255, 118, 0], [255, 120, 0], [255, 122, 0], [255, 124, 0], [255, 126, 0], [255, 128, 0], [255, 130, 0], [255, 132, 0], [255, 134, 0], [255, 136, 0], [255, 138, 0], [255, 140, 0], [255, 142, 0], [255, 143, 0], [255, 145, 0], [255, 147, 0], [255, 149, 0], [255, 151, 0], [255, 153, 0], [255, 154, 0], [255, 156, 1], [255, 158, 1], [255, 160, 1], [255, 161, 1], [255, 163, 1], [255, 165, 1], [255, 167, 1], [255, 168, 1], [255, 170, 1], [255, 172, 1], [255, 173, 2], [255, 175, 2], [255, 177, 2], [255, 178, 2], [255, 180, 2], [255, 182, 2], [255, 183, 3], [255, 185, 3], [255, 187, 3], [255, 188, 3], [255, 190, 3], [255, 192, 4], [255, 193, 4], [255, 195, 4], [255, 196, 4], [255, 198, 5], [255, 200, 5], [255, 201, 5], [255, 203, 5], [255, 204, 6], [255, 206, 6], [255, 208, 6], [255, 209, 7], [255, 211, 7], [255, 212, 8], [255, 214, 8], [255, 215, 9], [255, 217, 9], [255, 219, 10], [255, 220, 10], [255, 222, 11], [255, 223, 11], [255, 225, 12], [255, 226, 13], [255, 228, 13], [255, 229, 14], [255, 231, 15], [255, 233, 16], [255, 234, 17], [255, 236, 19], [255, 237, 21], [255, 239, 23], [255, 240, 26], [255, 242, 31], [255, 243, 36], [255, 245, 43], [255, 246, 51], [255, 247, 60], [255, 249, 71], [255, 250, 84], [255, 251, 98], [255, 252, 114], [255, 253, 132], [255, 254, 150], [255, 254, 168], [255, 254, 187], [255, 255, 205], [255, 255, 222], [255, 255, 239], [255, 255, 255]],
    ),
    'diverging_protanopic_deuteranopic_bwy_60_95_c32': ContinuousPalette(
        'Diverging protanopic', 'diverging_protanopic_deuteranopic_bwy_60_95_c32',
        [[58, 144, 254], [62, 145, 254], [65, 146, 254], [68, 146, 254], [70, 147, 254], [73, 148, 254], [76, 148, 254], [78, 149, 254], [80, 150, 254], [83, 151, 253], [85, 151, 253], [87, 152, 253], [89, 153, 253], [91, 153, 253], [94, 154, 253], [96, 155, 253], [97, 155, 253], [99, 156, 253], [101, 157, 253], [103, 158, 253], [105, 158, 253], [107, 159, 253], [109, 160, 252], [110, 160, 252], [112, 161, 252], [114, 162, 252], [115, 163, 252], [117, 163, 252], [119, 164, 252], [120, 165, 252], [122, 165, 252], [124, 166, 252], [125, 167, 252], [127, 168, 251], [128, 168, 251], [130, 169, 251], [131, 170, 251], [133, 171, 251], [134, 171, 251], [136, 172, 251], [137, 173, 251], [139, 173, 251], [140, 174, 251], [141, 175, 251], [143, 176, 250], [144, 176, 250], [146, 177, 250], [147, 178, 250], [148, 179, 250], [150, 179, 250], [151, 180, 250], [152, 181, 250], [154, 182, 250], [155, 182, 250], [156, 183, 250], [158, 184, 249], [159, 185, 249], [160, 185, 249], [161, 186, 249], [163, 187, 249], [164, 187, 249], [165, 188, 249], [166, 189, 249], [168, 190, 249], [169, 191, 249], [170, 191, 248], [171, 192, 248], [173, 193, 248], [174, 194, 248], [175, 194, 248], [176, 195, 248], [177, 196, 248], [179, 197, 248], [180, 197, 248], [181, 198, 247], [182, 199, 247], [183, 200, 247], [185, 200, 247], [186, 201, 247], [187, 202, 247], [188, 203, 247], [189, 203, 247], [190, 204, 247], [192, 205, 246], [193, 206, 246], [194, 206, 246], [195, 207, 246], [196, 208, 246], [197, 209, 246], [198, 210, 246], [199, 210, 246], [201, 211, 246], [202, 212, 245], [203, 213, 245], [204, 213, 245], [205, 214, 245], [206, 215, 245], [207, 216, 245], [208, 217, 245], [209, 217, 245], [211, 218, 244], [212, 219, 244], [213, 220, 244], [214, 220, 244], [215, 221, 244], [216, 222, 244], [217, 223, 244], [218, 224, 244], [219, 224, 243], [220, 225, 243], [221, 226, 243], [222, 227, 243], [223, 228, 243], [224, 228, 243], [226, 229, 243], [227, 230, 242], [228, 231, 242], [229, 231, 242], [230, 232, 242], [231, 233, 242], [232, 234, 241], [233, 234, 241], [234, 235, 241], [234, 236, 240], [235, 236, 240], [236, 236, 239], [236, 237, 238], [237, 237, 237], [237, 237, 236], [238, 237, 235], [238, 236, 234], [238, 236, 232], [238, 236, 231], [238, 235, 229], [237, 234, 228], [237, 234, 226], [237, 233, 224], [236, 232, 223], [236, 231, 221], [236, 231, 219], [235, 230, 218], [235, 229, 216], [234, 228, 214], [234, 228, 213], [233, 227, 211], [233, 226, 209], [233, 225, 208], [232, 224, 206], [232, 224, 204], [231, 223, 202], [231, 222, 201], [230, 221, 199], [230, 220, 197], [229, 220, 196], [229, 219, 194], [228, 218, 192], [228, 217, 191], [227, 216, 189], [227, 216, 187], [226, 215, 186], [226, 214, 184], [226, 213, 182], [225, 213, 181], [225, 212, 179], [224, 211, 177], [224, 210, 176], [223, 209, 174], [223, 209, 172], [222, 208, 171], [222, 207, 169], [221, 206, 167], [220, 206, 166], [220, 205, 164], [219, 204, 162], [219, 203, 161], [218, 203, 159], [218, 202, 157], [217, 201, 156], [217, 200, 154], [216, 199, 152], [216, 199, 151], [215, 198, 149], [215, 197, 148], [214, 196, 146], [214, 196, 144], [213, 195, 143], [212, 194, 141], [212, 193, 139], [211, 193, 138], [211, 192, 136], [210, 191, 134], [210, 190, 133], [209, 190, 131], [208, 189, 129], [208, 188, 128], [207, 187, 126], [207, 187, 125], [206, 186, 123], [206, 185, 121], [205, 184, 120], [204, 184, 118], [204, 183, 116], [203, 182, 115], [203, 181, 113], [202, 181, 111], [201, 180, 110], [201, 179, 108], [200, 178, 106], [199, 178, 105], [199, 177, 103], [198, 176, 102], [198, 175, 100], [197, 175, 98], [196, 174, 97], [196, 173, 95], [195, 172, 93], [194, 172, 92], [194, 171, 90], [193, 170, 88], [193, 169, 87], [192, 169, 85], [191, 168, 83], [191, 167, 81], [190, 166, 80], [189, 166, 78], [189, 165, 76], [188, 164, 75], [187, 164, 73], [187, 163, 71], [186, 162, 69], [185, 161, 68], [185, 161, 66], [184, 160, 64], [183, 159, 62], [183, 159, 60], [182, 158, 59], [181, 157, 57], [180, 156, 55], [180, 156, 53], [179, 155, 51], [178, 154, 49], [178, 153, 47], [177, 153, 45], [176, 152, 43], [176, 151, 41], [175, 151, 39], [174, 150, 36], [173, 149, 34], [173, 149, 32], [172, 148, 29], [171, 147, 26], [171, 146, 23], [170, 146, 20], [169, 145, 17], [168, 144, 13], [168, 144, 8]],
        category="Color blind", flags=Palette.ColorBlindSafe | Palette.Diverging
    ),
    'diverging_tritanopic_cwr_75_98_c20': ContinuousPalette(
        'Diverging tritanopic', 'diverging_tritanopic_cwr_75_98_c20',
        [[41, 202, 231], [46, 202, 231], [50, 202, 231], [54, 203, 231], [57, 203, 231], [60, 203, 232], [64, 204, 232], [67, 204, 232], [70, 205, 232], [72, 205, 232], [75, 205, 232], [78, 206, 232], [80, 206, 233], [83, 207, 233], [85, 207, 233], [87, 207, 233], [89, 208, 233], [92, 208, 233], [94, 208, 234], [96, 209, 234], [98, 209, 234], [100, 210, 234], [102, 210, 234], [104, 210, 234], [106, 211, 234], [108, 211, 235], [110, 211, 235], [111, 212, 235], [113, 212, 235], [115, 213, 235], [117, 213, 235], [119, 213, 235], [120, 214, 236], [122, 214, 236], [124, 214, 236], [125, 215, 236], [127, 215, 236], [129, 216, 236], [130, 216, 236], [132, 216, 237], [134, 217, 237], [135, 217, 237], [137, 217, 237], [138, 218, 237], [140, 218, 237], [141, 219, 237], [143, 219, 238], [144, 219, 238], [146, 220, 238], [147, 220, 238], [149, 220, 238], [150, 221, 238], [152, 221, 238], [153, 222, 239], [155, 222, 239], [156, 222, 239], [158, 223, 239], [159, 223, 239], [160, 223, 239], [162, 224, 239], [163, 224, 240], [165, 225, 240], [166, 225, 240], [167, 225, 240], [169, 226, 240], [170, 226, 240], [172, 226, 240], [173, 227, 241], [174, 227, 241], [176, 228, 241], [177, 228, 241], [178, 228, 241], [180, 229, 241], [181, 229, 241], [182, 229, 242], [184, 230, 242], [185, 230, 242], [186, 230, 242], [188, 231, 242], [189, 231, 242], [190, 232, 242], [191, 232, 243], [193, 232, 243], [194, 233, 243], [195, 233, 243], [197, 233, 243], [198, 234, 243], [199, 234, 243], [200, 235, 244], [202, 235, 244], [203, 235, 244], [204, 236, 244], [205, 236, 244], [207, 236, 244], [208, 237, 244], [209, 237, 245], [210, 237, 245], [212, 238, 245], [213, 238, 245], [214, 239, 245], [215, 239, 245], [216, 239, 245], [218, 240, 246], [219, 240, 246], [220, 240, 246], [221, 241, 246], [223, 241, 246], [224, 241, 246], [225, 242, 246], [226, 242, 247], [227, 243, 247], [229, 243, 247], [230, 243, 247], [231, 244, 247], [232, 244, 247], [233, 244, 247], [235, 245, 247], [236, 245, 248], [237, 245, 248], [238, 246, 248], [239, 246, 248], [240, 246, 248], [242, 247, 248], [243, 247, 248], [244, 247, 248], [245, 247, 248], [246, 247, 247], [246, 247, 247], [247, 247, 247], [248, 246, 246], [248, 246, 246], [249, 246, 245], [249, 245, 244], [250, 245, 244], [250, 244, 243], [250, 243, 242], [250, 243, 241], [251, 242, 241], [251, 241, 240], [251, 241, 239], [251, 240, 238], [251, 239, 237], [251, 239, 237], [251, 238, 236], [252, 237, 235], [252, 237, 234], [252, 236, 233], [252, 235, 233], [252, 235, 232], [252, 234, 231], [252, 233, 230], [252, 232, 229], [252, 232, 229], [253, 231, 228], [253, 230, 227], [253, 230, 226], [253, 229, 225], [253, 228, 224], [253, 228, 224], [253, 227, 223], [253, 226, 222], [253, 226, 221], [253, 225, 220], [253, 224, 220], [254, 224, 219], [254, 223, 218], [254, 222, 217], [254, 221, 216], [254, 221, 216], [254, 220, 215], [254, 219, 214], [254, 219, 213], [254, 218, 212], [254, 217, 212], [254, 217, 211], [254, 216, 210], [254, 215, 209], [254, 215, 208], [254, 214, 208], [254, 213, 207], [255, 213, 206], [255, 212, 205], [255, 211, 204], [255, 210, 204], [255, 210, 203], [255, 209, 202], [255, 208, 201], [255, 208, 201], [255, 207, 200], [255, 206, 199], [255, 206, 198], [255, 205, 197], [255, 204, 197], [255, 204, 196], [255, 203, 195], [255, 202, 194], [255, 202, 193], [255, 201, 193], [255, 200, 192], [255, 199, 191], [255, 199, 190], [255, 198, 190], [255, 197, 189], [255, 197, 188], [255, 196, 187], [255, 195, 186], [255, 195, 186], [255, 194, 185], [255, 193, 184], [255, 193, 183], [255, 192, 182], [255, 191, 182], [255, 190, 181], [255, 190, 180], [255, 189, 179], [255, 188, 179], [255, 188, 178], [255, 187, 177], [255, 186, 176], [255, 186, 175], [255, 185, 175], [255, 184, 174], [255, 184, 173], [255, 183, 172], [255, 182, 172], [255, 181, 171], [255, 181, 170], [255, 180, 169], [255, 179, 169], [254, 179, 168], [254, 178, 167], [254, 177, 166], [254, 177, 165], [254, 176, 165], [254, 175, 164], [254, 174, 163], [254, 174, 162], [254, 173, 162], [254, 172, 161], [254, 172, 160], [254, 171, 159], [254, 170, 159], [254, 170, 158], [254, 169, 157], [254, 168, 156], [254, 167, 156], [254, 167, 155], [253, 166, 154], [253, 165, 153], [253, 165, 153], [253, 164, 152], [253, 163, 151], [253, 163, 150], [253, 162, 150], [253, 161, 149], [253, 160, 148]],
        category="Color blind", flags=Palette.ColorBlindSafe | Palette.Diverging
    ),
    'linear_protanopic_deuteranopic_kbw_5_98_c40': ContinuousPalette(
        'Linear protanopic', 'linear_protanopic_deuteranopic_kbw_5_98_c40',
        [[17, 17, 17], [17, 18, 19], [18, 19, 21], [19, 19, 23], [19, 20, 24], [20, 21, 26], [20, 22, 28], [20, 23, 29], [21, 23, 31], [21, 24, 33], [21, 25, 34], [22, 25, 36], [22, 26, 38], [22, 27, 39], [22, 27, 41], [22, 28, 43], [22, 29, 45], [22, 30, 46], [23, 30, 48], [23, 31, 50], [23, 32, 52], [23, 33, 54], [23, 33, 55], [23, 34, 57], [23, 35, 59], [22, 36, 61], [22, 36, 63], [22, 37, 64], [22, 38, 66], [22, 39, 68], [22, 39, 70], [21, 40, 72], [21, 41, 74], [21, 42, 75], [20, 43, 77], [20, 43, 79], [20, 44, 81], [19, 45, 83], [19, 46, 84], [19, 46, 86], [18, 47, 88], [18, 48, 90], [18, 49, 91], [17, 50, 93], [17, 50, 95], [17, 51, 96], [16, 52, 98], [16, 53, 99], [16, 54, 101], [16, 54, 103], [16, 55, 104], [16, 56, 106], [16, 57, 107], [15, 58, 109], [15, 59, 110], [15, 59, 112], [15, 60, 113], [15, 61, 115], [15, 62, 116], [15, 63, 118], [15, 63, 119], [15, 64, 121], [15, 65, 122], [15, 66, 124], [15, 67, 126], [15, 68, 127], [15, 68, 129], [15, 69, 130], [15, 70, 132], [15, 71, 133], [15, 72, 135], [15, 73, 136], [15, 73, 138], [15, 74, 139], [15, 75, 141], [15, 76, 142], [15, 77, 144], [15, 78, 146], [15, 79, 147], [15, 79, 149], [14, 80, 150], [14, 81, 152], [14, 82, 153], [14, 83, 155], [14, 84, 157], [14, 85, 158], [14, 86, 160], [14, 86, 161], [13, 87, 163], [13, 88, 165], [13, 89, 166], [13, 90, 168], [13, 91, 169], [13, 92, 171], [12, 92, 173], [12, 93, 174], [12, 94, 176], [12, 95, 178], [11, 96, 179], [11, 97, 181], [11, 98, 182], [10, 99, 184], [10, 100, 186], [10, 100, 187], [9, 101, 189], [9, 102, 191], [9, 103, 192], [8, 104, 194], [8, 105, 196], [7, 106, 197], [7, 107, 199], [7, 108, 201], [6, 109, 202], [6, 109, 204], [5, 110, 206], [5, 111, 207], [4, 112, 209], [4, 113, 211], [3, 114, 212], [3, 115, 214], [2, 116, 216], [2, 117, 217], [1, 118, 219], [1, 119, 221], [0, 120, 222], [0, 120, 224], [0, 121, 226], [1, 122, 227], [1, 123, 229], [2, 124, 230], [3, 125, 232], [5, 126, 233], [7, 127, 235], [11, 128, 236], [14, 129, 237], [18, 130, 238], [22, 131, 239], [26, 132, 240], [31, 132, 241], [35, 133, 242], [40, 134, 242], [45, 135, 242], [49, 136, 242], [54, 137, 242], [59, 138, 242], [64, 139, 241], [69, 140, 240], [74, 141, 239], [79, 141, 238], [84, 142, 236], [89, 143, 234], [93, 144, 232], [98, 145, 230], [103, 146, 228], [107, 147, 226], [111, 148, 223], [115, 149, 220], [119, 150, 217], [123, 151, 214], [127, 152, 212], [131, 153, 208], [134, 154, 205], [137, 155, 202], [141, 155, 199], [144, 156, 196], [147, 157, 193], [150, 158, 189], [152, 159, 186], [155, 160, 183], [158, 161, 180], [160, 162, 177], [163, 163, 174], [165, 164, 170], [168, 165, 167], [170, 166, 164], [172, 167, 161], [175, 168, 158], [177, 169, 155], [179, 170, 152], [181, 171, 149], [183, 172, 146], [185, 173, 143], [187, 174, 140], [189, 175, 137], [191, 176, 134], [193, 177, 131], [195, 178, 128], [197, 179, 125], [199, 180, 122], [201, 181, 119], [202, 182, 116], [204, 183, 113], [206, 184, 109], [208, 184, 106], [209, 185, 103], [211, 186, 100], [213, 187, 96], [214, 188, 93], [216, 189, 90], [218, 190, 86], [219, 191, 82], [221, 192, 79], [222, 193, 75], [224, 194, 72], [225, 195, 68], [227, 196, 64], [228, 197, 60], [229, 198, 57], [231, 199, 53], [232, 200, 49], [233, 201, 46], [235, 202, 42], [236, 203, 39], [237, 204, 37], [239, 205, 35], [240, 206, 34], [241, 207, 34], [242, 208, 35], [243, 209, 37], [244, 210, 39], [245, 211, 43], [246, 212, 47], [247, 214, 52], [248, 215, 57], [249, 216, 62], [250, 217, 68], [251, 218, 74], [251, 219, 80], [252, 220, 86], [253, 221, 93], [253, 222, 99], [254, 223, 106], [254, 224, 112], [255, 225, 119], [255, 226, 126], [255, 227, 132], [255, 228, 139], [255, 229, 146], [255, 230, 152], [255, 231, 159], [255, 232, 165], [255, 233, 171], [255, 235, 177], [255, 236, 183], [254, 237, 189], [254, 238, 195], [254, 239, 201], [253, 240, 206], [253, 241, 211], [253, 242, 216], [253, 243, 221], [252, 245, 226], [252, 246, 231], [252, 247, 235], [252, 248, 239], [252, 249, 243]],
        category="Color blind", flags=Palette.ColorBlindSafe
    ),
    'linear_tritanopic_krjcw_5_95_c24': ContinuousPalette(
        'Linear tritanopic', 'linear_tritanopic_krjcw_5_95_c24',
        [[17, 17, 17], [20, 17, 17], [22, 18, 17], [24, 18, 17], [26, 18, 17], [28, 19, 17], [30, 19, 17], [32, 19, 17], [34, 20, 17], [35, 20, 17], [37, 20, 17], [39, 20, 17], [41, 21, 17], [42, 21, 18], [44, 21, 18], [46, 21, 18], [47, 22, 18], [49, 22, 18], [51, 22, 18], [52, 22, 18], [54, 22, 19], [56, 22, 19], [57, 23, 19], [59, 23, 19], [61, 23, 19], [62, 23, 19], [64, 23, 20], [65, 24, 20], [67, 24, 20], [68, 24, 20], [70, 24, 21], [71, 25, 21], [73, 25, 21], [75, 25, 21], [76, 25, 22], [78, 26, 22], [79, 26, 22], [81, 26, 22], [82, 26, 23], [83, 27, 23], [85, 27, 23], [86, 27, 24], [88, 28, 24], [89, 28, 24], [91, 28, 25], [92, 29, 25], [93, 29, 25], [95, 30, 26], [96, 30, 26], [97, 31, 27], [99, 31, 27], [100, 32, 28], [101, 32, 28], [102, 33, 29], [104, 33, 29], [105, 34, 30], [106, 35, 30], [107, 35, 31], [108, 36, 31], [110, 36, 32], [111, 37, 33], [112, 38, 33], [113, 39, 34], [114, 39, 35], [115, 40, 35], [116, 41, 36], [117, 42, 37], [118, 43, 37], [119, 44, 38], [120, 45, 39], [121, 46, 40], [122, 46, 41], [122, 47, 41], [123, 48, 42], [124, 49, 43], [125, 51, 44], [125, 52, 45], [126, 53, 46], [127, 54, 47], [127, 55, 48], [128, 56, 49], [128, 57, 50], [129, 58, 51], [129, 60, 53], [130, 61, 54], [130, 62, 55], [130, 63, 56], [131, 65, 57], [131, 66, 59], [131, 67, 60], [131, 69, 61], [131, 70, 63], [131, 72, 64], [131, 73, 66], [131, 74, 67], [131, 76, 69], [131, 77, 70], [131, 79, 72], [131, 80, 73], [130, 81, 75], [130, 83, 76], [130, 84, 78], [130, 85, 79], [130, 87, 81], [129, 88, 82], [129, 89, 84], [129, 91, 86], [129, 92, 87], [128, 93, 89], [128, 95, 90], [128, 96, 92], [128, 97, 93], [127, 99, 95], [127, 100, 97], [126, 101, 98], [126, 103, 100], [126, 104, 101], [125, 105, 103], [125, 106, 105], [124, 108, 106], [124, 109, 108], [123, 110, 110], [123, 112, 111], [122, 113, 113], [122, 114, 115], [121, 115, 116], [120, 117, 118], [120, 118, 120], [119, 119, 121], [118, 121, 123], [118, 122, 125], [117, 123, 126], [116, 124, 128], [115, 126, 130], [114, 127, 132], [114, 128, 133], [113, 129, 135], [112, 131, 137], [111, 132, 139], [110, 133, 140], [109, 134, 142], [107, 136, 144], [106, 137, 146], [105, 138, 147], [104, 140, 149], [103, 141, 151], [101, 142, 153], [100, 143, 155], [98, 145, 156], [97, 146, 158], [95, 147, 160], [94, 148, 162], [92, 150, 164], [90, 151, 166], [88, 152, 167], [86, 153, 169], [84, 155, 171], [82, 156, 173], [80, 157, 175], [77, 158, 177], [75, 160, 179], [72, 161, 181], [69, 162, 182], [66, 164, 184], [63, 165, 186], [60, 166, 188], [56, 167, 190], [53, 168, 192], [50, 170, 193], [46, 171, 195], [43, 172, 197], [40, 173, 198], [36, 174, 200], [33, 175, 201], [29, 177, 203], [26, 178, 204], [23, 179, 206], [19, 180, 207], [16, 181, 208], [13, 182, 210], [11, 183, 211], [9, 184, 212], [8, 185, 213], [8, 186, 215], [9, 187, 216], [11, 188, 217], [13, 189, 218], [16, 190, 219], [19, 191, 220], [22, 192, 221], [26, 193, 222], [29, 194, 223], [33, 195, 224], [36, 196, 225], [40, 197, 226], [43, 198, 227], [47, 199, 227], [50, 200, 228], [54, 201, 229], [57, 202, 230], [61, 203, 230], [64, 203, 231], [68, 204, 232], [71, 205, 233], [75, 206, 233], [78, 207, 234], [81, 208, 234], [85, 209, 235], [88, 209, 235], [92, 210, 236], [95, 211, 236], [98, 212, 237], [102, 213, 237], [105, 213, 238], [108, 214, 238], [112, 215, 239], [115, 216, 239], [118, 217, 239], [121, 217, 240], [125, 218, 240], [128, 219, 240], [131, 220, 241], [135, 220, 241], [138, 221, 241], [141, 222, 241], [144, 223, 242], [148, 223, 242], [151, 224, 242], [154, 225, 242], [157, 225, 242], [160, 226, 242], [164, 227, 243], [167, 227, 243], [170, 228, 243], [173, 229, 243], [177, 229, 243], [180, 230, 243], [183, 231, 243], [186, 231, 243], [189, 232, 243], [193, 232, 243], [196, 233, 243], [199, 234, 243], [202, 234, 243], [205, 235, 243], [209, 235, 242], [212, 236, 242], [215, 236, 242], [218, 237, 242], [221, 238, 242], [225, 238, 242], [228, 239, 241], [231, 239, 241], [234, 240, 241], [237, 240, 241], [241, 241, 241]],
        category="Color blind", flags=Palette.ColorBlindSafe
    ),
    'isoluminant_cgo_80_c38': ContinuousPalette(
        'Isoluminant', 'isoluminant_cgo_80_c38',
        [[112, 209, 255], [112, 210, 255], [112, 210, 255], [112, 210, 255], [112, 210, 255], [112, 210, 254], [112, 210, 254], [112, 210, 253], [112, 210, 252], [112, 210, 251], [112, 210, 250], [112, 210, 250], [113, 211, 249], [113, 211, 248], [113, 211, 247], [113, 211, 247], [113, 211, 246], [113, 211, 245], [113, 211, 244], [113, 211, 243], [113, 211, 243], [113, 211, 242], [113, 211, 241], [114, 212, 240], [114, 212, 239], [114, 212, 238], [114, 212, 238], [114, 212, 237], [114, 212, 236], [114, 212, 235], [114, 212, 234], [115, 212, 234], [115, 212, 233], [115, 212, 232], [115, 212, 231], [115, 212, 230], [115, 213, 229], [115, 213, 229], [116, 213, 228], [116, 213, 227], [116, 213, 226], [116, 213, 225], [116, 213, 225], [116, 213, 224], [116, 213, 223], [117, 213, 222], [117, 213, 221], [117, 213, 220], [117, 213, 219], [117, 213, 219], [118, 213, 218], [118, 214, 217], [118, 214, 216], [118, 214, 215], [118, 214, 214], [119, 214, 214], [119, 214, 213], [119, 214, 212], [119, 214, 211], [119, 214, 210], [120, 214, 209], [120, 214, 208], [120, 214, 208], [120, 214, 207], [121, 214, 206], [121, 214, 205], [121, 214, 204], [122, 214, 203], [122, 214, 202], [122, 214, 201], [122, 214, 201], [123, 214, 200], [123, 215, 199], [123, 215, 198], [124, 215, 197], [124, 215, 196], [124, 215, 195], [125, 215, 194], [125, 215, 193], [125, 215, 193], [126, 215, 192], [126, 215, 191], [126, 215, 190], [127, 215, 189], [127, 215, 188], [128, 215, 187], [128, 215, 186], [129, 215, 185], [129, 215, 184], [129, 215, 184], [130, 215, 183], [130, 215, 182], [131, 215, 181], [131, 215, 180], [132, 215, 179], [132, 215, 178], [133, 215, 177], [133, 215, 176], [134, 215, 175], [134, 215, 174], [135, 215, 173], [136, 215, 172], [136, 215, 172], [137, 215, 171], [137, 215, 170], [138, 215, 169], [139, 215, 168], [139, 215, 167], [140, 215, 166], [141, 214, 165], [141, 214, 164], [142, 214, 163], [143, 214, 162], [144, 214, 161], [144, 214, 160], [145, 214, 160], [146, 214, 159], [147, 214, 158], [147, 214, 157], [148, 214, 156], [149, 214, 155], [150, 214, 154], [151, 213, 153], [152, 213, 153], [153, 213, 152], [154, 213, 151], [154, 213, 150], [155, 213, 149], [156, 213, 148], [157, 213, 148], [158, 212, 147], [159, 212, 146], [160, 212, 145], [161, 212, 144], [162, 212, 144], [163, 212, 143], [164, 211, 142], [165, 211, 142], [166, 211, 141], [167, 211, 140], [168, 211, 140], [169, 211, 139], [170, 210, 138], [171, 210, 138], [172, 210, 137], [173, 210, 136], [174, 210, 136], [175, 209, 135], [176, 209, 135], [177, 209, 134], [178, 209, 134], [179, 209, 133], [180, 208, 133], [181, 208, 132], [182, 208, 132], [183, 208, 131], [184, 207, 131], [185, 207, 130], [186, 207, 130], [187, 207, 129], [188, 207, 129], [189, 206, 128], [190, 206, 128], [191, 206, 127], [192, 206, 127], [193, 205, 127], [194, 205, 126], [195, 205, 126], [196, 205, 125], [197, 204, 125], [197, 204, 125], [198, 204, 124], [199, 204, 124], [200, 203, 124], [201, 203, 123], [202, 203, 123], [203, 203, 123], [204, 202, 122], [205, 202, 122], [206, 202, 122], [207, 202, 121], [208, 201, 121], [209, 201, 121], [209, 201, 121], [210, 201, 120], [211, 200, 120], [212, 200, 120], [213, 200, 120], [214, 199, 119], [215, 199, 119], [216, 199, 119], [217, 199, 119], [217, 198, 119], [218, 198, 119], [219, 198, 118], [220, 197, 118], [221, 197, 118], [222, 197, 118], [223, 197, 118], [224, 196, 118], [224, 196, 118], [225, 196, 118], [226, 195, 118], [227, 195, 118], [228, 195, 118], [229, 194, 118], [229, 194, 118], [230, 194, 118], [231, 194, 118], [232, 193, 118], [233, 193, 118], [233, 193, 118], [234, 192, 118], [235, 192, 118], [236, 192, 118], [237, 191, 118], [237, 191, 118], [238, 191, 118], [239, 190, 118], [240, 190, 119], [241, 190, 119], [241, 189, 119], [242, 189, 119], [243, 189, 119], [244, 188, 119], [244, 188, 120], [245, 188, 120], [246, 188, 120], [247, 187, 120], [247, 187, 121], [248, 187, 121], [249, 186, 121], [249, 186, 121], [250, 186, 122], [251, 185, 122], [252, 185, 122], [252, 185, 122], [253, 184, 123], [254, 184, 123], [254, 184, 123], [255, 183, 124], [255, 183, 124], [255, 183, 124], [255, 182, 125], [255, 182, 125], [255, 182, 125], [255, 181, 126], [255, 181, 126], [255, 181, 127], [255, 180, 127], [255, 180, 127], [255, 180, 128], [255, 179, 128], [255, 179, 129], [255, 179, 129], [255, 178, 129]],
        category="Other"
    ),
    'rainbow_bgyr_35_85_c73': ContinuousPalette(
        'Rainbow', 'rainbow_bgyr_35_85_c73',
        [[0, 53, 249], [0, 56, 246], [0, 58, 243], [0, 61, 240], [0, 63, 237], [0, 66, 234], [0, 68, 231], [0, 71, 228], [0, 73, 225], [0, 75, 223], [0, 77, 220], [0, 79, 217], [0, 81, 214], [0, 83, 211], [0, 85, 208], [0, 87, 205], [0, 89, 202], [0, 91, 199], [0, 92, 196], [0, 94, 194], [0, 96, 191], [0, 98, 188], [0, 99, 185], [0, 101, 182], [0, 103, 179], [0, 104, 176], [0, 106, 174], [0, 108, 171], [0, 109, 168], [0, 111, 165], [0, 112, 163], [0, 113, 160], [0, 115, 157], [0, 116, 155], [0, 117, 152], [0, 118, 150], [7, 119, 147], [14, 120, 145], [20, 122, 142], [24, 123, 140], [28, 124, 137], [32, 125, 135], [35, 126, 133], [38, 127, 130], [41, 128, 128], [43, 129, 126], [45, 130, 123], [47, 131, 121], [49, 132, 118], [51, 133, 116], [52, 134, 114], [53, 135, 111], [55, 136, 109], [56, 137, 106], [57, 138, 104], [58, 139, 101], [59, 140, 99], [59, 141, 96], [60, 142, 94], [61, 143, 91], [61, 144, 88], [62, 145, 86], [62, 146, 83], [62, 147, 80], [63, 148, 78], [63, 149, 75], [63, 150, 72], [63, 152, 69], [63, 153, 66], [63, 154, 63], [63, 155, 60], [63, 156, 57], [63, 157, 53], [63, 158, 50], [63, 159, 47], [63, 160, 43], [63, 161, 40], [64, 162, 36], [64, 163, 33], [65, 164, 30], [66, 165, 27], [68, 166, 24], [70, 166, 21], [72, 167, 19], [74, 168, 17], [77, 169, 16], [79, 169, 15], [82, 170, 14], [85, 171, 13], [88, 171, 13], [90, 172, 13], [93, 172, 14], [96, 173, 14], [99, 174, 14], [101, 174, 14], [104, 175, 15], [106, 175, 15], [109, 176, 16], [112, 177, 16], [114, 177, 16], [117, 178, 17], [119, 178, 17], [122, 179, 17], [124, 180, 18], [126, 180, 18], [129, 181, 19], [131, 181, 19], [134, 182, 19], [136, 182, 20], [138, 183, 20], [141, 183, 20], [143, 184, 21], [145, 185, 21], [148, 185, 21], [150, 186, 22], [152, 186, 22], [154, 187, 23], [157, 187, 23], [159, 188, 23], [161, 188, 24], [163, 189, 24], [166, 189, 24], [168, 190, 25], [170, 191, 25], [172, 191, 26], [175, 192, 26], [177, 192, 26], [179, 193, 27], [181, 193, 27], [183, 194, 27], [186, 194, 28], [188, 195, 28], [190, 195, 28], [192, 196, 29], [194, 196, 29], [196, 197, 30], [199, 197, 30], [201, 198, 30], [203, 198, 31], [205, 199, 31], [207, 199, 31], [209, 200, 32], [211, 200, 32], [214, 201, 33], [216, 201, 33], [218, 202, 33], [220, 202, 34], [222, 203, 34], [224, 203, 34], [226, 203, 35], [229, 204, 35], [231, 204, 35], [233, 205, 36], [235, 205, 36], [237, 205, 36], [239, 205, 37], [241, 205, 37], [242, 205, 37], [244, 205, 37], [245, 205, 37], [247, 204, 37], [248, 204, 36], [249, 203, 36], [250, 202, 36], [251, 201, 35], [251, 200, 35], [252, 199, 34], [252, 197, 34], [253, 196, 33], [253, 195, 33], [253, 193, 32], [253, 192, 32], [254, 191, 31], [254, 189, 30], [254, 188, 30], [254, 187, 29], [254, 185, 29], [255, 184, 28], [255, 182, 27], [255, 181, 27], [255, 180, 26], [255, 178, 25], [255, 177, 25], [255, 176, 24], [255, 174, 24], [255, 173, 23], [255, 171, 22], [255, 170, 22], [255, 168, 21], [255, 167, 20], [255, 166, 20], [255, 164, 19], [255, 163, 18], [255, 161, 18], [255, 160, 17], [255, 158, 16], [255, 157, 16], [255, 156, 15], [255, 154, 14], [255, 153, 13], [255, 151, 13], [255, 150, 12], [255, 148, 11], [255, 147, 10], [255, 145, 10], [255, 144, 9], [255, 142, 8], [255, 141, 7], [255, 139, 7], [255, 138, 6], [255, 136, 5], [255, 134, 5], [255, 133, 4], [255, 131, 3], [255, 130, 3], [255, 128, 2], [255, 127, 2], [255, 125, 1], [255, 123, 1], [255, 122, 0], [255, 120, 0], [255, 118, 0], [255, 117, 0], [255, 115, 0], [255, 113, 0], [255, 112, 0], [255, 110, 0], [255, 108, 0], [255, 106, 0], [255, 104, 0], [255, 103, 0], [255, 101, 0], [255, 99, 0], [255, 97, 0], [255, 95, 0], [255, 93, 0], [255, 91, 0], [255, 89, 0], [255, 87, 0], [255, 85, 0], [255, 83, 0], [255, 81, 0], [255, 79, 0], [255, 76, 0], [255, 74, 0], [255, 72, 0], [255, 69, 0], [255, 67, 0], [255, 64, 0], [255, 61, 0], [255, 59, 0], [255, 56, 0], [255, 53, 0], [255, 49, 0], [255, 46, 0], [255, 42, 0]],
        category="Other"
    ),
}


DefaultContinuousPaletteName = "linear_bgyw_15_100_c68"
DefaultContinuousPalette = ContinuousPalettes[DefaultContinuousPaletteName]


class ColorIcon(QIcon):
    def __init__(self, color, size=12):
        p = QPixmap(size, size)
        p.fill(color)
        super().__init__(p)


def get_default_curve_colors(n):
    if n <= len(Dark2Colors):
        return list(Dark2Colors)[:n]
    if n <= len(DefaultRGBColors):
        return list(DefaultRGBColors)[:n]
    else:
        return list(LimitedDiscretePalette(n))


def patch_variable_colors():
    # This function patches Variable with properties and private attributes:
    # pylint: disable=protected-access
    from Orange.data import Variable, DiscreteVariable, ContinuousVariable

    def get_colors(var):
        return var._colors

    def set_colors(var, colors):
        var._colors = colors
        var._palette = None
        var.attributes["colors"] = [
            color_to_hex(color) if isinstance(color, (Sequence, np.ndarray))
            else color
            for color in colors]
        if "palette" in var.attributes:
            del var.attributes["palette"]

    def get_palette(var):
        return var._palette

    def set_palette(var, palette):
        var._palette = palette
        var.attributes["palette"] = palette.name
        var._colors = None
        if "colors" in var.attributes:
            del var.attributes["colors"]

    def continuous_get_colors(var):
        warnings.warn("ContinuousVariable.color is deprecated; "
                      "use ContinuousVariable.palette",
                      DeprecationWarning, stacklevel=2)
        if var._colors is None:
            try:
                col1, col2, black = var.attributes["colors"]
                var._colors = (hex_to_color(col1), hex_to_color(col2), black)
            except (KeyError, ValueError):  # unavailable or invalid
                if var._palette or "palette" in var.attributes:
                    palette = var.palette
                    col1 = tuple(palette.palette[0])
                    col2 = tuple(palette.palette[-1])
                    black = bool(palette.flags & palette.Diverging)
                    var._colors = col1, col2, black
                else:
                    var._colors = ((0, 0, 255), (255, 255, 0), False)
        return var._colors

    def continuous_get_palette(var):
        if var._palette is None:
            if "palette" in var.attributes:
                var._palette = ContinuousPalettes.get(var.attributes["palette"],
                                                      DefaultContinuousPalette)
            elif var._colors is not None or "colors" in var.attributes:
                col1, col2, black = var.colors
                var._palette = ContinuousPalette.from_colors(col1, col2, black)
            else:
                var._palette = DefaultContinuousPalette
        return var._palette

    def discrete_get_colors(var):
        if var._colors is None or len(var._colors) < len(var.values):
            if var._palette is not None or "palette" in var.attributes:
                var._colors = var.palette.palette[:len(var.values)]
            else:
                var._colors = np.empty((0, 3), dtype=object)
            colors = var.attributes.get("colors")
            if colors:
                try:
                    var._colors = np.vstack(
                        ([hex_to_color(color) for color in colors],
                         var._colors[len(colors):]))
                except ValueError:
                    pass
            if len(var._colors) < len(var.values):
                var._colors = LimitedDiscretePalette(len(var.values)).palette
            var._colors.flags.writeable = False
        return var._colors

    def discrete_set_colors(var, colors):
        colors = colors.copy()
        colors.flags.writeable = False
        set_colors(var, colors)

    def discrete_get_palette(var):
        if var._palette is None:
            if "palette" in var.attributes:
                var._palette = DiscretePalettes.get(var.attributes["palette"],
                                                    DefaultDiscretePalette)
            elif var._colors is not None or "colors" in var.attributes:
                var._palette = DiscretePalette.from_colors(var.colors)
            else:
                var._palette = LimitedDiscretePalette(len(var.values))
        return var._palette

    Variable._colors = None
    Variable._palette = None
    Variable.colors = property(get_colors, set_colors)
    Variable.palette = property(get_palette, set_palette)

    DiscreteVariable.colors = property(discrete_get_colors, discrete_set_colors)
    DiscreteVariable.palette = property(discrete_get_palette, set_palette)

    ContinuousVariable.colors = property(continuous_get_colors, set_colors)
    ContinuousVariable.palette = property(continuous_get_palette, set_palette)
