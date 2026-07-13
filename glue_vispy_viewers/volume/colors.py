from math import floor
from glue.config import AsinhStretch, LinearStretch, LogStretch, SqrtStretch, DictRegistry
from matplotlib.colors import ListedColormap
from vispy.color import BaseColormap


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


def create_cmap_template(n):
    ts = tuple((i + 1) / n for i in range(n-1))
    lines = [
        "vec4 translucent_colormap(float t) {",
    ]
    for i, t in enumerate(ts):
        lines.append(f"    if (t <= {t})")
        lines.append(f"        {{ return $color_{i}; }}")
    lines.append(f"    return $color_{n-1};")
    lines.append("}")

    return "\n".join(lines)


def glsl_for_stretch(stretch, parameter="t"):
    template = stretch_glsl.members.get(type(stretch), "{param}")
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


def get_mpl_cmap(cmap, alpha_stretch, color_stretch):

    # Mesa llvmpipe (Linux/CI software OpenGL) miscompiles the long
    # chained-if function emitted by ``create_cmap_template`` past some
    # length. Uniform-data probes suggested the chain is fine up to ~100
    # ifs, but real-data tests show n=100 produces dark-speckle artifacts
    # on a small fraction of t values (~1000 broken pixels in an L1448
    # cube render at n=100, vs 0 at n=80 and n=64). 64 is a comfortable
    # margin that's been visually verified to match Apple's GL output on
    # real data. Apple's GL handles long chains fine; this cap is purely
    # to keep CI/headless renders matching real-GPU output.
    #
    # Most matplotlib cmaps (including plasma/viridis/inferno) are
    # ``ListedColormap`` with 256 entries -- without this cap the bug
    # triggers on every standard mpl cmap.
    n_colors = 64

    if isinstance(cmap, ListedColormap):
        all_colors = cmap.colors
        # Subsample if the cmap has more entries than n_colors
        step = max(1, len(all_colors) // n_colors)
        colors = list(all_colors[::step])[:n_colors]
        n_colors = len(colors)
        base_ts = [index / n_colors for index in range(n_colors)]
        alpha_ts = alpha_stretch(base_ts)
        color_ts = color_stretch(base_ts)
        color_indices = [floor(t * n_colors) for t in color_ts]
        colors = [[*colors[cidx][:3], t] for t, cidx in zip(alpha_ts, color_indices)]
    else:
        base_ts = [index / n_colors for index in range(n_colors)]
        alpha_ts = alpha_stretch(base_ts)
        color_ts = color_stretch(base_ts)
        colors = [[*cmap(color_t)[:3], alpha_t] for color_t, alpha_t in zip(alpha_ts, color_ts)]

    template = create_cmap_template(n_colors)

    class MatplotlibCmap(BaseColormap):
        glsl_map = template

    return MatplotlibCmap(colors=colors)
