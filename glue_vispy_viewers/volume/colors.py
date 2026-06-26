from glue.config import AsinhStretch, LinearStretch, LogStretch, SqrtStretch, DictRegistry
from matplotlib.colors import ListedColormap
from vispy.color import BaseColormap, Colormap
from uuid import uuid4


class GLSLStretchRegistry(DictRegistry):

    def add(self, stretch_cls, glsl):
        self.members[stretch_cls] = glsl


stretch_glsl = GLSLStretchRegistry()
stretch_glsl.add(LogStretch, "log({stretch.a} * {parameter} + 1.0) / log({stretch.a} + 1.0)")
stretch_glsl.add(SqrtStretch, "sqrt({parameter})")
stretch_glsl.add(AsinhStretch,
                 "log({parameter} / {stretch.a} + sqrt(pow({parameter} / {stretch.a}, 2) + 1)) / "
                 "log(1.0 / {stretch.a} + sqrt(pow(1.0 / {stretch.a}, 2) + 1))")
stretch_glsl.add(LinearStretch, "{parameter}")


def glsl_for_stretch(stretch, parameter="t"):
    template = stretch_glsl.members.get(type(stretch), "{param}")
    return template.format(stretch=stretch, parameter=parameter)


class CustomColormap(Colormap):

    def __init__(self, colors, controls=None, interpolation="linear"):
        super().__init__(colors, controls=controls, interpolation=interpolation)

        self.uuid = uuid4().hex
        self.texture_name = f"texture2D_LUT_{self.uuid}"
        self.glsl_map = self.glsl_map.replace("texture2D_LUT", self.texture_name)


def get_translucent_cmap(r, g, b, stretch):

    func = glsl_for_stretch(stretch)

    class TranslucentCmap(BaseColormap):
        glsl_map = """
        vec4 translucent_fire(float t) {{
            return vec4({0}, {1}, {2}, {3});
        }}
        """.format(r, g, b, func)

    return TranslucentCmap()


def get_mpl_cmap(cmap, stretch):

    if isinstance(cmap, ListedColormap):
        colors = cmap.colors
        n_colors = len(colors)
        ts = stretch([index / (n_colors - 1) for index in range(n_colors)])
        colors = [[*color, t] for t, color in zip(ts, colors)]
    else:
        n_colors = 256
        ts = stretch([index / (n_colors - 1) for index in range(n_colors)])
        colors = [[*cmap(t)[:3], t] for t in ts]

    return CustomColormap(colors=colors, controls=ts, interpolation="linear")
