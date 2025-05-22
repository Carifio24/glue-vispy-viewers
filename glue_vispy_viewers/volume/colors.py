from glue.config import AsinhStretch, LinearStretch, LogStretch, SqrtStretch
from matplotlib.colors import ListedColormap
from vispy.color import BaseColormap


STRETCHES_GLSL = {
    LogStretch: "log({stretch.a} * {parameter} + 1.0) / log({stretch.a} + 1.0)",
    SqrtStretch: "sqrt({parameter})",
    AsinhStretch: "log({parameter} / {stretch.a} + sqrt(pow({parameter} / {stretch.a}, 2) + 1)) / "
                  "log(1.0 / {stretch.a} + sqrt(pow(1.0 / {stretch.a}, 2) + 1))",
    LinearStretch: "{parameter}",
}


def create_cmap_template(n, stretch_glsl):
    ts = tuple(i / n for i in range(n-1))
    lines = [
        "vec4 translucent_colormap(float t) {",
        f"    float s = {stretch_glsl};"
    ]
    for i, t in enumerate(ts):
        lines.append(f"    if (s <= {t})")
        lines.append(f"        {{ return $color_{i}; }}")
    lines.append(f"    return $color_{n-1};")
    lines.append("}")

    return "\n".join(lines)


def glsl_for_stretch(stretch, parameter="t"):
    template = STRETCHES_GLSL.get(type(stretch), "{param}")
    return template.format(stretch=stretch, parameter=parameter)


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
        ts = stretch([index / n_colors for index in range(len(colors))])
        colors = [color + [t] for t, color in zip(ts, colors)]
    else:
        n_colors = 256
        ts = stretch([index / n_colors for index in range(n_colors)])
        colors = [cmap(t)[:3] + (t,) for t in ts]

    stretch_glsl = glsl_for_stretch(stretch)
    template = create_cmap_template(n_colors, stretch_glsl)

    class MatplotlibCmap(BaseColormap):
        glsl_map = template

    return MatplotlibCmap(colors=colors)
