# This file implements a MultiIsoVisual class that can be used to show
# multiple layers of isosurface simultaneously. It is derived from the original
# VolumeVisual class in vispy.visuals.volume, which is releaed under a BSD license
# included here:
#
# ===========================================================================
# Vispy is licensed under the terms of the (new) BSD license:
#
# Copyright (c) 2015, authors of Vispy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of Vispy Development Team nor the names of its
#   contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================
#
# This modified version is released under the BSD license given in the LICENSE
# file in this repository.


from vispy.gloo import Texture3D, TextureEmulated3D, VertexBuffer, IndexBuffer
from vispy.visuals.volume import VolumeVisual, Visual
from vispy.scene.visuals import create_visual_node
from vispy.color import get_colormap, Color

import numpy as np

from ..volume.shaders import FRAG_SHADER, VERT_SHADER


def get_frag_shader():
    declarations = """
        uniform $sampler_type u_volumetex;
        uniform int u_level;
        uniform float u_relative_step_size;
        uniform float u_threshold;
    """
    before_loop = """
        vec4 total_color = vec4(0.0);  // final color
        vec4 src = vec4(0.0);
        vec4 dst = vec4(0.0);
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative
        gl_FragColor = vec4(0.0);
        float val_prev = 0;
        float outa = 0;
        vec3 loc_prev = vec3(0.0);
        vec3 loc_mid = vec3(0.0);
    """
    in_loop="""
        for (int i=0; i<u_level; i++){

        // render from outside to inside
        if (val < u_threshold*(1.0-i/float(u_level)) && val_prev > u_threshold*(1.0-i/float(u_level))){
            // Use bisection to find correct position of contour
            for (int i=0; i<20; i++) {
                loc_mid = 0.5 * (loc_prev + loc);
                val = $sample(u_volumetex, loc_mid).g;
                if (val < u_threshold) {
                    loc = loc_mid;
                } else {
                    loc_prev = loc_mid;
                }
            }

            //dst = $cmap(val);  // this will call colormap function if have
            dst = $cmap(i);
            dst = calculateColor(dst, loc, dstep);
            dst.a = 1. * (1.0 - i/float(level)); // transparency

            src = total_color;

            outa = src.a + dst.a * (1 - src.a);
            total_color = (src * src.a + dst * dst.a * (1 - src.a)) / outa;
            total_color.a = outa;
        }
        }
        val_prev = val;
        loc_prev = loc;
    """

    after_loop="""
        gl_FragColor = total_color;
    """

    functions = """
    float colorToVal(vec4 color1)
    {{
        return color1.g; // todo: why did I have this abstraction in visvis?
    }}
    vec4 calculateColor(vec4 betterColor, vec3 loc, vec3 step)
    {{
        // Calculate color by incorporating lighting
        vec4 color1;
        vec4 color2;

        // View direction
        vec3 V = normalize(view_ray);

        // calculate normal vector from gradient
        vec3 N; // normal
        color1 = $sample( u_volumetex, loc+vec3(-step[0],0.0,0.0) );
        color2 = $sample( u_volumetex, loc+vec3(step[0],0.0,0.0) );
        N[0] = colorToVal(color1) - colorToVal(color2);
        betterColor = max(max(color1, color2),betterColor);
        color1 = $sample( u_volumetex, loc+vec3(0.0,-step[1],0.0) );
        color2 = $sample( u_volumetex, loc+vec3(0.0,step[1],0.0) );
        N[1] = colorToVal(color1) - colorToVal(color2);
        betterColor = max(max(color1, color2),betterColor);
        color1 = $sample( u_volumetex, loc+vec3(0.0,0.0,-step[2]) );
        color2 = $sample( u_volumetex, loc+vec3(0.0,0.0,step[2]) );
        N[2] = colorToVal(color1) - colorToVal(color2);
        betterColor = max(max(color1, color2),betterColor);
        float gm = length(N); // gradient magnitude
        N = normalize(N);

        // Flip normal so it points towards viewer
        float Nselect = float(dot(N,V) > 0.0);
        N = (2.0*Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

        // Get color of the texture (albeido)
        color1 = betterColor;
        color2 = color1;
        // todo: parametrise color1_to_color2

        // Init colors
        vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
        vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
        vec4 specular_color = vec4(0.0, 0.0, 0.0, 0.0);
        vec4 final_color;

        // todo: allow multiple light, define lights on viewvox or subscene
        int nlights = 1;
        for (int i=0; i<nlights; i++)
        {{
            // Get light direction (make sure to prevent zero devision)
            vec3 L = normalize(view_ray);  //lightDirs[i];
            float lightEnabled = float( length(L) > 0.0 );
            L = normalize(L+(1.0-lightEnabled));

            // Calculate lighting properties
            float lambertTerm = clamp( dot(N,L), 0.0, 1.0 );
            vec3 H = normalize(L+V); // Halfway vector
            float specularTerm = pow( max(dot(H,N),0.0), u_shininess);

            // Calculate mask
            float mask1 = lightEnabled;

            // Calculate colors
            ambient_color +=  mask1 * u_ambient;  // * gl_LightSource[i].ambient;
            diffuse_color +=  mask1 * lambertTerm;
            specular_color += mask1 * specularTerm * u_specular;
        }}

        // Calculate final color by componing different components
        final_color = color2 * ( ambient_color + diffuse_color) + specular_color;
        final_color.a = color2.a;

        // Done
        return final_color;
    }}"""

    return FRAG_SHADER.format(declarations=declarations,
                              before_loop=before_loop,
                              in_loop=in_loop,
                              after_loop=after_loop,
                              functions=functions)

class MultiIsoVisual(VolumeVisual):

    def __init__(self, emulate_texture=False, relative_step_size=0.8, threshold=0.8, step=4, bgcolor='white', cmap='hot'):

        tex_cls = TextureEmulated3D if emulate_texture else Texture3D

        # Storage of information of volume
        self._vol_shape = ()
        self._clim = None
        self._need_vertex_update = True

        self._cmap = get_colormap(cmap)

        # We deliberately don't use super here because we don't want to call
        # VolumeVisual.__init__
        Visual.__init__(self, vcode=VERT_SHADER, fcode="")

        self._update_shader()

        self._vertices = VertexBuffer()
        self._texcoord = VertexBuffer(
            np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ], dtype=np.float32))

        self._draw_mode = 'triangle_strip'
        self._index_buffer = IndexBuffer()

        self.shared_program['a_position'] = self._vertices
        self.shared_program['a_texcoord'] = self._texcoord

        # Only show back faces of cuboid. This is required because if we are
        # inside the volume, then the front faces are outside of the clipping
        # box and will not be drawn.
        self.set_gl_state('translucent', cull_face=False)

        # Don't use downsampling initially (1 means show 1:1 resolution)
        self.shared_program['u_downsample'] = 1.

        # Set initial background color
        self.shared_program['u_bgcolor'] = Color(bgcolor).rgba

        self.shared_program['u_level'] = step
        self.shared_program['u_relative_step_size'] = relative_step_size
        self.shared_program['u_threshold'] = threshold

        # Prevent additional attributes from being added
        try:
            self.freeze()
        except AttributeError:  # Older versions of VisPy
            pass

    def _update_shader(self, force=False):
        shader = get_frag_shader()
        # We only actually update the shader in OpenGL if the code has changed
        # to avoid any overheads in uploading the new shader code
        if force or getattr(self, '_shader_cache', None) != shader:
            self.shared_program.frag = shader
            self._shader_cache = shader

    def _prepare_transforms(self, view):

        # Copied from VolumeVisual in vispy v0.8.1 as this was then chnaged
        # in v0.9.0 in a way that breaks things.

        trs = view.transforms
        view.view_program.vert['transform'] = trs.get_transform()

        view_tr_f = trs.get_transform('visual', 'document')
        view_tr_i = view_tr_f.inverse
        view.view_program.vert['viewtransformf'] = view_tr_f
        view.view_program.vert['viewtransformi'] = view_tr_i


MultiIsoVisual = create_visual_node(MultiIsoVisual)
