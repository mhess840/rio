# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Draw scene layers into a WiVRn session. Display only: nothing here moves a robot.

WiVRn streams an OpenXR application from this machine. The browser page does
not appear in that session. This module draws whatever layers are in an
:class:`~rio.xr.hub.XrHub`, in front of the head until a controller has touched
three saved poses.

    uv run python -m rio.apps.kassow.xr --headset

Ctrl-C leaves the session. The headset returns to the WiVRn lobby.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import sys
import time
from ctypes import (
    POINTER,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_ulong,
    c_void_p,
)
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from rio.xr.auth import token_path, write_token
from rio.xr.capture import PoseCapture
from rio.xr.geometry import layer_vertices, layers_from_scene, pose_axes, preview_transform, rotation_matrix
from rio.xr.hub import XrHub
from rio.xr.scene import Layer, axes
from rio.xr.screen import ScreenCfg, ScreenGrabber, matrix_pose, panel_pose, pose_matrix
from rio.xr.server import XrServer

LOADER_CACHE = Path.home() / ".cache" / "rio" / "openxr" / "libopenxr_loader.so.1"
# The aim pose sits in the controller body. The recorded point is this far along
# its forward axis (-Z), so the controller never has to touch the gripper.
POINTER_M = 0.15
RUNTIME_JSON = Path.home() / ".cache" / "rio" / "openxr" / "wivrn_runtime.json"
PRIMARY_STEREO = 2
LOCAL_SPACE = 2
STAGE_SPACE = 3
# Location flags: orientation and position valid (0x1, 0x2) and actually tracked (0x4, 0x8).
LOCATION_VALID = 0x3
LOCATION_TRACKED = 0xF
# Before a fit in STAGE, the preview is lifted to roughly table height above the floor origin.
STAGE_PREVIEW_LIFT_M = 1.2
BLEND_OPAQUE = 1
BLEND_ALPHA = 3
LAYER_SOURCE_ALPHA = 0x2
PASSTHROUGH_LAYER_TYPE = 1000118003
READY = 2
STOPPING = 6
LOSS_PENDING = 7
# Controllers seen and then gone this long: rebuild the OpenXR session (WiVRn binds them at connect).
CONTROLLERS_LOST_S = 10.0
RESTART_PAUSE_S = 3.0
EXITING = 8
COLOR_ATTACHMENT = 0x1
GL_SRGB8_ALPHA8 = 0x8C43
GL_RGBA8 = 0x8058
GL_FLOAT = 0x1406
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_DYNAMIC_DRAW = 0x88E8
VERTEX_STRIDE = 28
GL_DEPTH_TEST = 0x0B71
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x100
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_DEPTH_ATTACHMENT = 0x8D00
GL_TEXTURE_2D = 0x0DE1
GL_RENDERBUFFER = 0x8D41
GL_DEPTH_COMPONENT24 = 0x81A6
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_ARRAY_BUFFER = 0x8892
GL_STATIC_DRAW = 0x88E4
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_BGRA = 0x80E1
GL_UNSIGNED_BYTE = 0x1401
GL_UNPACK_ALIGNMENT = 0x0CF5
QUAD_LAYER_TYPE = 36


def _loader() -> Path:
    """OpenXR loader shared library, copied into the user cache when missing."""
    if LOADER_CACHE.is_file():
        return LOADER_CACHE
    found = list(Path("/tmp/xrload").glob("usr/lib/*/libopenxr_loader.so.1"))
    if not found:
        raise FileNotFoundError("OpenXR loader not found. Install libopenxr-loader1 or extract it under /tmp/xrload.")
    LOADER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LOADER_CACHE.write_bytes(found[0].read_bytes())
    return LOADER_CACHE


def _runtime_json() -> Path:
    """Runtime manifest whose library path points at the installed WiVRn flatpak."""
    libs = sorted(Path("/var/lib/flatpak/app/io.github.wivrn.wivrn").glob("x86_64/stable/*/files/lib/wivrn/libopenxr_wivrn.so"))
    if not libs:
        raise FileNotFoundError("WiVRn OpenXR runtime library was not found in the flatpak install.")
    RUNTIME_JSON.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"file_format_version": "1.0.0", "runtime": {"name": "WiVRn", "library_path": str(libs[-1])}}
    RUNTIME_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    return RUNTIME_JSON


def _check(lib, instance: int, result: int, what: str) -> None:
    if result == 0:
        return
    text = ""
    if instance:
        buf = ctypes.create_string_buffer(64)
        lib.xrResultToString(instance, result, buf)
        text = buf.value.decode()
    raise RuntimeError(f"{what} failed ({result} {text})")


class _Vec3(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


class _Quat(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("w", c_float)]


class _Pose(ctypes.Structure):
    _fields_ = [("orientation", _Quat), ("position", _Vec3)]


class _Fov(ctypes.Structure):
    _fields_ = [("angle_left", c_float), ("angle_right", c_float), ("angle_up", c_float), ("angle_down", c_float)]


def _identity_pose() -> _Pose:
    return _Pose(orientation=_Quat(0, 0, 0, 1), position=_Vec3(0, 0, 0))


def _glx_context():
    """A current OpenGL 3.3 core context on a pbuffer. Returns display, config, drawable, context, visual id."""
    x11 = ctypes.CDLL("libX11.so.6")
    glx = ctypes.CDLL("libGLX.so.0")
    x11.XOpenDisplay.restype = c_void_p
    x11.XOpenDisplay.argtypes = [c_char_p]
    x11.XDefaultScreen.argtypes = [c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError(f"XOpenDisplay failed for DISPLAY={os.environ.get('DISPLAY')!r}")
    screen = x11.XDefaultScreen(display)
    attrs = (c_int * 17)(0x8012, 1, 0x8010, 0x4, 0x8011, 0x1, 8, 8, 9, 8, 10, 8, 11, 8, 12, 24, 0)
    # The pairs above are GLX_X_RENDERABLE, drawable type PBUFFER, RGBA, then red/green/blue/alpha/depth sizes.
    count = c_int()
    glx.glXChooseFBConfig.restype = POINTER(c_void_p)
    glx.glXChooseFBConfig.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_int)]
    configs = glx.glXChooseFBConfig(display, screen, attrs, byref(count))
    if not configs or count.value < 1:
        raise RuntimeError("glXChooseFBConfig returned no config with a pbuffer")
    config = configs[0]
    visual = c_int()
    glx.glXGetFBConfigAttrib.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_int)]
    glx.glXGetFBConfigAttrib(display, config, 0x800B, byref(visual))
    glx.glXGetProcAddress.restype = c_void_p
    glx.glXGetProcAddress.argtypes = [c_char_p]
    create_ptr = glx.glXGetProcAddress(b"glXCreateContextAttribsARB")
    if not create_ptr:
        raise RuntimeError("glXCreateContextAttribsARB is missing")
    create = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_void_p, c_int, POINTER(c_int))(create_ptr)
    context_attrs = (c_int * 7)(0x2091, 3, 0x2092, 3, 0x9126, 1, 0)
    context = create(display, config, None, 1, context_attrs)
    if not context:
        raise RuntimeError("OpenGL 3.3 context was not created")
    pbuffer_attrs = (c_int * 5)(0x8041, 32, 0x8042, 32, 0)
    glx.glXCreatePbuffer.restype = c_ulong
    glx.glXCreatePbuffer.argtypes = [c_void_p, c_void_p, POINTER(c_int)]
    drawable = glx.glXCreatePbuffer(display, config, pbuffer_attrs)
    glx.glXMakeContextCurrent.argtypes = [c_void_p, c_ulong, c_ulong, c_void_p]
    if not glx.glXMakeContextCurrent(display, drawable, drawable, context):
        raise RuntimeError("glXMakeContextCurrent failed")
    return display, config, drawable, context, visual.value


def _gl_fn(glx, name: str, restype, argtypes):
    ptr = glx.glXGetProcAddress(name.encode())
    if not ptr:
        raise RuntimeError(f"OpenGL entry {name} is missing")
    fn = ctypes.CFUNCTYPE(restype, *argtypes)(ptr)
    return fn


class _Gl:
    """The OpenGL entry points this view uses, loaded from the current context."""

    def __init__(self) -> None:
        glx = ctypes.CDLL("libGLX.so.0")
        glx.glXGetProcAddress.restype = c_void_p
        glx.glXGetProcAddress.argtypes = [c_char_p]
        g = lambda name, restype, argtypes: _gl_fn(glx, name, restype, argtypes)
        self.gen_buffers = g("glGenBuffers", None, [c_int, POINTER(c_uint32)])
        self.bind_buffer = g("glBindBuffer", None, [c_uint32, c_uint32])
        self.buffer_data = g("glBufferData", None, [c_uint32, c_int, c_void_p, c_uint32])
        self.gen_vertex_arrays = g("glGenVertexArrays", None, [c_int, POINTER(c_uint32)])
        self.bind_vertex_array = g("glBindVertexArray", None, [c_uint32])
        self.enable_vertex_attrib = g("glEnableVertexAttribArray", None, [c_uint32])
        self.vertex_attrib_pointer = g("glVertexAttribPointer", None, [c_uint32, c_int, c_uint32, c_uint32, c_int, c_void_p])
        self.create_shader = g("glCreateShader", c_uint32, [c_uint32])
        self.shader_source = g("glShaderSource", None, [c_uint32, c_int, POINTER(c_char_p), POINTER(c_int)])
        self.compile_shader = g("glCompileShader", None, [c_uint32])
        self.get_shader_iv = g("glGetShaderiv", None, [c_uint32, c_uint32, POINTER(c_int)])
        self.get_shader_info = g("glGetShaderInfoLog", None, [c_uint32, c_int, POINTER(c_int), c_char_p])
        self.create_program = g("glCreateProgram", c_uint32, [])
        self.attach_shader = g("glAttachShader", None, [c_uint32, c_uint32])
        self.link_program = g("glLinkProgram", None, [c_uint32])
        self.get_program_iv = g("glGetProgramiv", None, [c_uint32, c_uint32, POINTER(c_int)])
        self.get_program_info = g("glGetProgramInfoLog", None, [c_uint32, c_int, POINTER(c_int), c_char_p])
        self.use_program = g("glUseProgram", None, [c_uint32])
        self.get_uniform = g("glGetUniformLocation", c_int, [c_uint32, c_char_p])
        self.uniform_matrix = g("glUniformMatrix4fv", None, [c_int, c_int, c_uint32, POINTER(c_float)])
        self.gen_framebuffers = g("glGenFramebuffers", None, [c_int, POINTER(c_uint32)])
        self.bind_framebuffer = g("glBindFramebuffer", None, [c_uint32, c_uint32])
        self.framebuffer_texture = g("glFramebufferTexture2D", None, [c_uint32, c_uint32, c_uint32, c_uint32, c_int])
        self.gen_renderbuffers = g("glGenRenderbuffers", None, [c_int, POINTER(c_uint32)])
        self.bind_renderbuffer = g("glBindRenderbuffer", None, [c_uint32, c_uint32])
        self.renderbuffer_storage = g("glRenderbufferStorage", None, [c_uint32, c_uint32, c_int, c_int])
        self.framebuffer_renderbuffer = g("glFramebufferRenderbuffer", None, [c_uint32, c_uint32, c_uint32, c_uint32])
        self.check_framebuffer = g("glCheckFramebufferStatus", c_uint32, [c_uint32])
        self.viewport = g("glViewport", None, [c_int, c_int, c_int, c_int])
        self.clear_color = g("glClearColor", None, [c_float, c_float, c_float, c_float])
        self.clear = g("glClear", None, [c_uint32])
        self.finish = g("glFinish", None, [])
        self.enable = g("glEnable", None, [c_uint32])
        self.blend_func = g("glBlendFunc", None, [c_uint32, c_uint32])
        self.draw_arrays = g("glDrawArrays", None, [c_uint32, c_int, c_int])
        self.line_width = g("glLineWidth", None, [c_float])
        self.bind_texture = g("glBindTexture", None, [c_uint32, c_uint32])
        self.pixel_store = g("glPixelStorei", None, [c_uint32, c_int])
        self.tex_sub_image = g(
            "glTexSubImage2D", None, [c_uint32, c_int, c_int, c_int, c_int, c_int, c_uint32, c_uint32, c_void_p]
        )

    def shader(self, kind: int, source: str) -> int:
        shader = self.create_shader(kind)
        src = c_char_p(source.encode())
        self.shader_source(shader, 1, ctypes.byref(src), None)
        self.compile_shader(shader)
        status = c_int()
        self.get_shader_iv(shader, GL_COMPILE_STATUS, byref(status))
        if not status.value:
            log = ctypes.create_string_buffer(1024)
            self.get_shader_info(shader, 1024, None, log)
            raise RuntimeError(log.value.decode())
        return shader


def _program(gl: _Gl) -> int:
    vertex = gl.shader(
        GL_VERTEX_SHADER,
        "#version 330 core\nlayout(location=0) in vec3 a_pos; layout(location=1) in vec4 a_color;\n"
        "uniform mat4 u_mvp; out vec4 v_color; void main(){ gl_Position = u_mvp * vec4(a_pos, 1.0); v_color = a_color; }\n",
    )
    fragment = gl.shader(
        GL_FRAGMENT_SHADER,
        "#version 330 core\nin vec4 v_color; out vec4 color; void main(){ color = v_color; }\n",
    )
    program = gl.create_program()
    gl.attach_shader(program, vertex)
    gl.attach_shader(program, fragment)
    gl.link_program(program)
    status = c_int()
    gl.get_program_iv(program, GL_LINK_STATUS, byref(status))
    if not status.value:
        log = ctypes.create_string_buffer(1024)
        gl.get_program_info(program, 1024, None, log)
        raise RuntimeError(log.value.decode())
    gl.enable(GL_DEPTH_TEST)
    gl.enable(GL_BLEND)
    gl.blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    gl.line_width(3)
    gl.clear_color(0.02, 0.04, 0.08, 1)
    return program


def _geometry_buffer(gl: _Gl) -> tuple[int, int]:
    """A vertex array of xyz rgba, ready to receive :func:`_upload`."""
    vao, vbo = c_uint32(), c_uint32()
    gl.gen_vertex_arrays(1, byref(vao))
    gl.gen_buffers(1, byref(vbo))
    gl.bind_vertex_array(vao)
    gl.bind_buffer(GL_ARRAY_BUFFER, vbo)
    gl.enable_vertex_attrib(0)
    gl.vertex_attrib_pointer(0, 3, GL_FLOAT, 0, VERTEX_STRIDE, None)
    gl.enable_vertex_attrib(1)
    gl.vertex_attrib_pointer(1, 4, GL_FLOAT, 0, VERTEX_STRIDE, c_void_p(12))
    return vao.value, vbo.value


def _upload(gl: _Gl, vbo: int, vertices: np.ndarray) -> None:
    array = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 7)
    gl.bind_buffer(GL_ARRAY_BUFFER, vbo)
    if len(array) == 0:
        gl.buffer_data(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        return
    gl.buffer_data(GL_ARRAY_BUFFER, array.nbytes, array.ctypes.data_as(c_void_p), GL_DYNAMIC_DRAW)


def _projection(fov: _Fov, near: float = 0.05, far: float = 20.0) -> np.ndarray:
    left, right = np.tan(fov.angle_left), np.tan(fov.angle_right)
    down, up = np.tan(fov.angle_down), np.tan(fov.angle_up)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 2 / (right - left)
    matrix[1, 1] = 2 / (up - down)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (up + down) / (up - down)
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2 * far * near) / (far - near)
    matrix[3, 2] = -1
    return matrix


def _view(pose: _Pose) -> np.ndarray:
    x, y, z, w = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation.T
    position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    matrix[:3, 3] = -rotation.T @ position
    return matrix


class _Xr:
    """Thin OpenXR call table. Handles stay Python ints; structs are built per call."""

    def __getattr__(self, name: str) -> Any:
        """OpenXR entry points are bound with ``setattr``; this only tells the type checker so."""
        raise AttributeError(name)

    def __init__(self, lib) -> None:
        self.lib = lib
        for name, restype, argtypes in (
            ("xrCreateInstance", c_uint32, [c_void_p, POINTER(c_uint64)]),
            ("xrDestroyInstance", c_uint32, [c_uint64]),
            ("xrGetSystem", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrGetInstanceProcAddr", c_uint32, [c_uint64, c_char_p, POINTER(c_void_p)]),
            ("xrResultToString", c_uint32, [c_uint64, c_uint32, c_char_p]),
            ("xrCreateSession", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrDestroySession", c_uint32, [c_uint64]),
            ("xrBeginSession", c_uint32, [c_uint64, c_void_p]),
            ("xrEndSession", c_uint32, [c_uint64]),
            ("xrPollEvent", c_uint32, [c_uint64, c_void_p]),
            ("xrCreateReferenceSpace", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            (
                "xrEnumerateViewConfigurationViews",
                c_uint32,
                [c_uint64, c_uint64, c_uint32, c_uint32, POINTER(c_uint32), c_void_p],
            ),
            ("xrEnumerateSwapchainFormats", c_uint32, [c_uint64, c_uint32, POINTER(c_uint32), c_void_p]),
            ("xrCreateSwapchain", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrEnumerateSwapchainImages", c_uint32, [c_uint64, c_uint32, POINTER(c_uint32), c_void_p]),
            ("xrAcquireSwapchainImage", c_uint32, [c_uint64, c_void_p, POINTER(c_uint32)]),
            ("xrWaitSwapchainImage", c_uint32, [c_uint64, c_void_p]),
            ("xrReleaseSwapchainImage", c_uint32, [c_uint64, c_void_p]),
            ("xrWaitFrame", c_uint32, [c_uint64, c_void_p, c_void_p]),
            ("xrBeginFrame", c_uint32, [c_uint64, c_void_p]),
            ("xrEndFrame", c_uint32, [c_uint64, c_void_p]),
            ("xrLocateViews", c_uint32, [c_uint64, c_void_p, c_void_p, c_uint32, POINTER(c_uint32), c_void_p]),
            (
                "xrEnumerateEnvironmentBlendModes",
                c_uint32,
                [c_uint64, c_uint64, c_uint32, c_uint32, POINTER(c_uint32), c_void_p],
            ),
        ):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)

    def proc(self, instance: int, name: str, restype, argtypes):
        ptr = c_void_p()
        _check(self.lib, instance, self.xrGetInstanceProcAddr(instance, name.encode(), byref(ptr)), name)
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr.value)  # ty: ignore[no-matching-overload]


class _Composite:
    """How the submitted frame sits on the real room."""

    blend = BLEND_OPAQUE
    passthrough = 0


_composite = _Composite()


def _extension_names(lib) -> set[bytes]:
    """Extensions this runtime offers. Empty when the query fails."""
    fn = lib.xrEnumerateInstanceExtensionProperties
    fn.restype = c_uint32
    fn.argtypes = [c_char_p, c_uint32, POINTER(c_uint32), c_void_p]
    count = c_uint32()
    if fn(None, 0, byref(count), None) != 0 or not count.value:
        return set()

    class Ext(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("name", c_char * 128), ("version", c_uint32)]

    props = (Ext * count.value)()
    for item in props:
        item.type = 2
    if fn(None, count, byref(count), ctypes.cast(props, c_void_p)) != 0:
        return set()
    return {bytes(item.name).split(b"\0", 1)[0] for item in props}


def _bind_instance(xr: _Xr, extensions: list[bytes]) -> tuple[int, int]:
    class App(ctypes.Structure):
        _fields_ = [
            ("applicationName", c_char * 128),
            ("applicationVersion", c_uint32),
            ("engineName", c_char * 128),
            ("engineVersion", c_uint32),
            ("apiVersion", c_uint64),
        ]

    class Create(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("createFlags", c_uint64),
            ("applicationInfo", App),
            ("enabledApiLayerCount", c_uint32),
            ("enabledApiLayerNames", POINTER(c_char_p)),
            ("enabledExtensionCount", c_uint32),
            ("enabledExtensionNames", POINTER(c_char_p)),
        ]

    names = (c_char_p * len(extensions))(*extensions)
    app = App(b"rio-headset", 1, b"rio", 1, (1 << 48) | 34)
    info = Create(type=3, applicationInfo=app, enabledExtensionCount=len(extensions), enabledExtensionNames=names)
    instance = c_uint64()
    _check(xr.lib, 0, xr.xrCreateInstance(byref(info), byref(instance)), "xrCreateInstance")

    class Get(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("formFactor", c_uint32)]

    system = c_uint64()
    _check(xr.lib, instance.value, xr.xrGetSystem(instance, byref(Get(type=4, formFactor=1)), byref(system)), "xrGetSystem")
    return instance.value, system.value


def _session(xr: _Xr, instance: int, system: int, display, config, drawable: int, context, visual: int) -> int:
    class Requirements(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("min_api", c_uint64), ("max_api", c_uint64)]

    get_req = xr.proc(instance, "xrGetOpenGLGraphicsRequirementsKHR", c_uint32, [c_uint64, c_uint64, c_void_p])
    req = Requirements(type=1000023005)
    _check(xr.lib, instance, get_req(instance, system, byref(req)), "xrGetOpenGLGraphicsRequirementsKHR")

    class Binding(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("xDisplay", c_void_p),
            ("visualid", c_uint32),
            ("glxFBConfig", c_void_p),
            ("glxDrawable", c_ulong),
            ("glxContext", c_void_p),
        ]

    class Create(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("createFlags", c_uint64), ("systemId", c_uint64)]

    binding = Binding(
        type=1000023001,
        xDisplay=display,
        visualid=visual,
        glxFBConfig=config,
        glxDrawable=drawable,
        glxContext=context,
    )
    info = Create(type=8, next=ctypes.addressof(binding), systemId=system)
    session = c_uint64()
    _check(xr.lib, instance, xr.xrCreateSession(instance, byref(info), byref(session)), "xrCreateSession")
    return session.value


def _choose_blend(xr: _Xr, instance: int, system: int) -> int:
    """Alpha blend when the runtime can show the room behind a transparent frame."""
    count = c_uint32()
    query = xr.xrEnumerateEnvironmentBlendModes
    if query(instance, system, PRIMARY_STEREO, 0, byref(count), None) != 0 or not count.value:
        return BLEND_OPAQUE
    modes = (c_uint32 * count.value)()
    if query(instance, system, PRIMARY_STEREO, count, byref(count), ctypes.cast(modes, c_void_p)) != 0:
        return BLEND_OPAQUE
    chosen = BLEND_ALPHA if BLEND_ALPHA in list(modes) else BLEND_OPAQUE
    logger.info("Environment blend {}", "alpha" if chosen == BLEND_ALPHA else "opaque")
    return chosen


def _start_passthrough(xr: _Xr, instance: int, session: int) -> int:
    """Quest camera reconstruction. Zero when the runtime has no passthrough extension."""

    def proc(name: str, argtypes: list) -> Any:
        return xr.proc(instance, name, c_uint32, argtypes)

    create = proc("xrCreatePassthroughFB", [c_uint64, c_void_p, POINTER(c_uint64)])
    create_layer = proc("xrCreatePassthroughLayerFB", [c_uint64, c_void_p, POINTER(c_uint64)])
    resume = proc("xrPassthroughLayerResumeFB", [c_uint64])

    class Create(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("flags", c_uint64)]

    class LayerInfo(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("passthrough", c_uint64),
            ("flags", c_uint64),
            ("purpose", c_uint32),
        ]

    passthrough = c_uint64()
    created = create(session, byref(Create(type=1000118001, flags=1)), byref(passthrough))
    if created != 0:
        logger.warning("Passthrough was not created ({})", created)
        return 0
    layer = c_uint64()
    info = LayerInfo(type=1000118002, passthrough=passthrough.value, flags=1, purpose=0)
    created = create_layer(session, byref(info), byref(layer))
    if created != 0:
        logger.warning("Passthrough layer was not created ({})", created)
        return 0
    resumed = resume(layer.value)
    if resumed != 0:
        logger.warning("Passthrough layer did not resume ({})", resumed)
        return 0
    logger.info("Passthrough camera is behind the overlay")
    return layer.value


def run(hub: XrHub | None = None, *, space: str = "stage", screen: ScreenCfg | None = None) -> None:
    """Open the WiVRn view and draw ``hub`` until Ctrl-C, rebuilding the session after a headset drop.

    Without a hub, a local one with the robot base axes is served on port 8770.
    Layers stay in front of the head until a calibration is set in ``hub`` or
    three controller touches fit the registration. When the headset sleeps or WiVRn
    reconnects, the runtime ends the session or the controllers vanish; the view then
    starts a fresh OpenXR session, keeping ``hub`` and its server.

    Args:
        hub: Scene, anchors, and calibration to draw.
        space: ``stage`` keeps the fit valid across recentring; falls back to ``local``.
        screen: Show the X11 desktop as a panel; None leaves it off.
    """
    os.environ["XR_RUNTIME_JSON"] = str(_runtime_json())
    own_server = None
    if hub is None:
        hub = XrHub()
        hub.put(Layer("base", "Robot base axes", tuple(axes(0.25))))
        own_server = XrServer(hub, port=8770, token=write_token(token_path()))
        own_server.start()
        logger.info("Desktop preview: {}", own_server.url)
    stop = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop
        stop = True
        logger.info("Stopping headset view ({})", signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop:
            try:
                restart = _run_session(hub, space, screen, lambda: stop)
            except RuntimeError as error:
                logger.warning("Headset session failed: {}", error)
                restart = True
            if not restart or stop:
                break
            logger.warning("Headset view restarts in {:.0f} s; reconnect WiVRn in the Quest if asked", RESTART_PAUSE_S)
            time.sleep(RESTART_PAUSE_S)
    finally:
        if own_server is not None:
            own_server.stop()


def _run_session(hub: XrHub, space: str, screen: ScreenCfg | None, stopped) -> bool:
    """One OpenXR instance and session. True when the headset dropped and a new session should start."""
    _FBO.clear()
    _FBO_OK.clear()
    display, config, drawable, context, visual = _glx_context()
    gl = _Gl()
    lib = ctypes.CDLL(str(_loader()))
    xr = _Xr(lib)
    extensions = [b"XR_KHR_opengl_enable"]
    offered = _extension_names(lib)
    if b"XR_FB_passthrough" in offered:
        extensions.append(b"XR_FB_passthrough")
    # Quest 3 controllers use the Touch Plus profile; without it the actions stay inactive.
    if b"XR_META_touch_controller_plus" in offered:
        extensions.append(b"XR_META_touch_controller_plus")
    instance, system = _bind_instance(xr, extensions)
    _composite.passthrough = 0
    _composite.blend = _choose_blend(xr, instance, system)
    session = 0
    try:
        session = _session(xr, instance, system, display, config, drawable, context, visual)
        if b"XR_FB_passthrough" in extensions:
            _composite.passthrough = _start_passthrough(xr, instance, session)
        hands = _Hands.try_create(xr, instance, session)
        grabber_cm = ScreenGrabber(screen.fps, factor=screen.downscale) if screen is not None else contextlib.nullcontext()
        with grabber_cm as grabber:
            desktop = None if grabber is None or screen is None else (grabber, screen)
            return _frames(xr, gl, instance, system, session, hub, hands, stopped, space, desktop)
    finally:
        if session:
            xr.xrDestroySession(session)
        xr.xrDestroyInstance(instance)
        logger.info("Headset session closed")


def _frames(
    xr: _Xr,
    gl: _Gl,
    instance: int,
    system: int,
    session: int,
    hub: XrHub,
    hands: _Hands | None,
    stopped,
    wanted: str,
    desktop: tuple[ScreenGrabber, ScreenCfg] | None = None,
) -> bool:
    """Render until stopped; True when the headset dropped (session lost, or controllers gone after being seen)."""
    begun = False
    space, hub.space = _reference_space(xr, instance, session, wanted)
    views_info, swapchains, images = _swapchains(xr, instance, system, session)
    if hands is not None:
        hands.create_spaces(xr, session, space)
    live = _LiveView(gl, hub, hands)
    panel = None
    if desktop is not None:
        try:
            panel = _ScreenPanel(xr, gl, instance, session, *desktop)
        except RuntimeError as error:
            logger.warning("Desktop panel off: {}", error)
    if _composite.passthrough or _composite.blend == BLEND_ALPHA:
        gl.clear_color(0.0, 0.0, 0.0, 0.0)
        logger.info("The room shows through. Graphics stay in front of you until three saved poses are touched.")
    frame = 0
    controllers_seen = False
    controllers_gone_at: float | None = None
    next_check = 0.0
    while not stopped():
        state = _poll(xr, instance, session)
        if state == LOSS_PENDING:
            logger.warning("WiVRn is losing the session (headset asleep or disconnected)")
            return True
        if state == READY and not begun:

            class Begin(ctypes.Structure):
                _fields_ = [("type", c_uint32), ("next", c_void_p), ("view", c_uint32)]

            _check(xr.lib, instance, xr.xrBeginSession(session, byref(Begin(type=10, view=PRIMARY_STEREO))), "xrBeginSession")
            begun = True
            logger.info("Session begun. Trigger saves the next saved pose, grip removes the last point.")
        if state in (STOPPING, EXITING) and begun:
            xr.xrEndSession(session)
            return state == STOPPING
        if not begun:
            time.sleep(0.01)
            continue
        _one_frame(xr, gl, instance, session, space, views_info, swapchains, images, live, panel)
        frame += 1
        now = time.monotonic()
        if hands is not None and hands.ready and now >= next_check:
            next_check = now + 1.0
            if any(hands.profiles().values()):
                controllers_seen, controllers_gone_at = True, None
            elif controllers_seen:
                controllers_gone_at = controllers_gone_at or now
                if now - controllers_gone_at > CONTROLLERS_LOST_S:
                    logger.warning("Controllers gone for {:.0f} s after a headset drop", CONTROLLERS_LOST_S)
                    return True
        if frame == 1 or frame % 300 == 0:
            logger.info("Submitted frame {}", frame)
            if hands is not None and hands.ready:
                seen = {name: ("tracked" if ok else "inferred") for name, (_, ok) in hands.devices.items()}
                logger.info("Controllers: profiles {} poses {}", hands.profiles(), seen or "none")
    return False


def _reference_space(xr: _Xr, instance: int, session: int, wanted: str = "stage") -> tuple[int, str]:
    """Create the render space. STAGE is tied to the guardian and survives recentring."""

    class Create(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("referenceSpaceType", c_uint32), ("pose", _Pose)]

    space = c_uint64()
    if wanted == "stage":
        created = xr.xrCreateReferenceSpace(
            session, byref(Create(type=37, referenceSpaceType=STAGE_SPACE, pose=_identity_pose())), byref(space)
        )
        if created == 0:
            logger.info("Drawing in the STAGE space; a saved calibration survives recentring")
            return space.value, "stage"
        logger.warning("STAGE space unavailable ({}); using LOCAL, recentring breaks the calibration", created)
    _check(
        xr.lib,
        instance,
        xr.xrCreateReferenceSpace(
            session, byref(Create(type=37, referenceSpaceType=LOCAL_SPACE, pose=_identity_pose())), byref(space)
        ),
        "xrCreateReferenceSpace",
    )
    return space.value, "local"


def _swapchains(xr: _Xr, instance: int, system: int, session: int):
    class ConfigView(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("recommendedImageRectWidth", c_uint32),
            ("maxImageRectWidth", c_uint32),
            ("recommendedImageRectHeight", c_uint32),
            ("maxImageRectHeight", c_uint32),
            ("recommendedSwapchainSampleCount", c_uint32),
            ("maxSwapchainSampleCount", c_uint32),
        ]

    count = c_uint32()
    listed = xr.xrEnumerateViewConfigurationViews(instance, system, PRIMARY_STEREO, 0, byref(count), None)
    _check(xr.lib, instance, listed, "view count")
    config = (ConfigView * count.value)()
    for item in config:
        item.type = 41
    _check(
        xr.lib,
        instance,
        xr.xrEnumerateViewConfigurationViews(
            instance, system, PRIMARY_STEREO, count, byref(count), ctypes.cast(config, c_void_p)
        ),
        "views",
    )
    chosen = _swapchain_format(xr, instance, session)
    logger.info("Swapchain format {:#x}, {} views", chosen, count.value)
    swapchains = []
    images = []
    described = []
    for view in config:
        width, height = view.recommendedImageRectWidth, view.recommendedImageRectHeight
        swapchain, pics = _make_swapchain(xr, instance, session, chosen, width, height, COLOR_ATTACHMENT)
        swapchains.append(swapchain)
        images.append(pics)
        described.append((width, height))
    return described, swapchains, images


def _swapchain_format(xr: _Xr, instance: int, session: int) -> int:
    """sRGB RGBA when the runtime offers it, else plain RGBA, else its first format."""
    formats_n = c_uint32()
    _check(xr.lib, instance, xr.xrEnumerateSwapchainFormats(session, 0, byref(formats_n), None), "format count")
    formats = (c_int64 * formats_n.value)()
    listed = xr.xrEnumerateSwapchainFormats(session, formats_n, byref(formats_n), ctypes.cast(formats, c_void_p))
    _check(xr.lib, instance, listed, "formats")
    chosen = GL_SRGB8_ALPHA8 if GL_SRGB8_ALPHA8 in formats else GL_RGBA8
    return chosen if chosen in formats else int(formats[0])


def _make_swapchain(
    xr: _Xr, instance: int, session: int, fmt: int, width: int, height: int, usage: int
) -> tuple[int, list[int]]:
    """One single-sample 2D swapchain and its GL texture names."""

    class SwapCreate(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("createFlags", c_uint64),
            ("usageFlags", c_uint64),
            ("format", c_int64),
            ("sampleCount", c_uint32),
            ("width", c_uint32),
            ("height", c_uint32),
            ("faceCount", c_uint32),
            ("arraySize", c_uint32),
            ("mipCount", c_uint32),
        ]

    class Image(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("image", c_uint32)]

    info = SwapCreate(
        type=9,
        usageFlags=usage,
        format=fmt,
        sampleCount=1,
        width=width,
        height=height,
        faceCount=1,
        arraySize=1,
        mipCount=1,
    )
    swapchain = c_uint64()
    _check(xr.lib, instance, xr.xrCreateSwapchain(session, byref(info), byref(swapchain)), "xrCreateSwapchain")
    n_images = c_uint32()
    _check(xr.lib, instance, xr.xrEnumerateSwapchainImages(swapchain, 0, byref(n_images), None), "image count")
    pics = (Image * n_images.value)()
    for pic in pics:
        pic.type = 1000023004
    _check(
        xr.lib,
        instance,
        xr.xrEnumerateSwapchainImages(swapchain, n_images, byref(n_images), ctypes.cast(pics, c_void_p)),
        "images",
    )
    return swapchain.value, [pic.image for pic in pics]


class _ScreenPanel:
    """The desktop as a quad layer, first upright in front of the head.

    Holding the left grip drags it with the left controller; the left trigger hides or shows it.
    """

    def __init__(self, xr: _Xr, gl: _Gl, instance: int, session: int, grabber: ScreenGrabber, cfg: ScreenCfg) -> None:
        self.xr = xr
        self.gl = gl
        self.instance = instance
        self.grabber = grabber
        self.cfg = cfg
        self.width, self.height = grabber.size
        fmt = _swapchain_format(xr, instance, session)
        self.swapchain, self.images = _make_swapchain(xr, instance, session, fmt, self.width, self.height, COLOR_ATTACHMENT)
        self.uploaded = -1
        self.pose: tuple[np.ndarray, np.ndarray] | None = None
        self.visible = True
        self.place_next = True
        # ``controller⁻¹ · panel`` while the left grip holds the panel.
        self.grab: np.ndarray | None = None
        logger.info(
            "Desktop panel {}x{} px, {:.2f} m wide. Hold left grip to move it, left trigger hides or shows it.",
            self.width,
            self.height,
            cfg.width_m,
        )

    def update(
        self,
        head: tuple[np.ndarray, np.ndarray] | None,
        left_pressed: set[str],
        left_held: set[str],
        left_controller: np.ndarray | None,
    ) -> None:
        """Apply the left controller, place the panel if asked, and upload a new frame.

        Args:
            head: Head position and orientation in the reference space.
            left_pressed: Left buttons that went down this frame.
            left_held: Left buttons held now.
            left_controller: ``4 x 4`` left controller pose, or None when not located.
        """
        if "trigger" in left_pressed:
            self.visible = not self.visible
            logger.info("Desktop panel {}", "shown" if self.visible else "hidden")
            if self.visible and self.pose is None:
                self.place_next = True
        if self.place_next and head is not None:
            self.pose = panel_pose(head[0], head[1], self.cfg)
            self.place_next = False
        if "squeeze" in left_held and left_controller is not None and self.pose is not None and self.visible:
            if self.grab is None:
                self.grab = np.linalg.inv(left_controller) @ pose_matrix(*self.pose)
            self.pose = matrix_pose(left_controller @ self.grab)
        else:
            self.grab = None
        generation, frame = self.grabber.latest()
        if frame is None or generation == self.uploaded or not self.visible:
            return

        class Acquire(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p)]

        class WaitImage(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("timeout", c_int64)]

        xr, gl = self.xr, self.gl
        index = c_uint32()
        _check(
            xr.lib, self.instance, xr.xrAcquireSwapchainImage(self.swapchain, byref(Acquire(type=55)), byref(index)), "acquire"
        )
        waited = xr.xrWaitSwapchainImage(self.swapchain, byref(WaitImage(type=56, timeout=int(1e9))))
        _check(xr.lib, self.instance, waited, "wait image")
        gl.bind_texture(GL_TEXTURE_2D, self.images[index.value])
        gl.pixel_store(GL_UNPACK_ALIGNMENT, 4)
        gl.tex_sub_image(
            GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_BGRA, GL_UNSIGNED_BYTE, frame.ctypes.data_as(c_void_p)
        )
        gl.bind_texture(GL_TEXTURE_2D, 0)
        _check(xr.lib, self.instance, xr.xrReleaseSwapchainImage(self.swapchain, byref(Acquire(type=57))), "release")
        self.uploaded = generation

    def layer(self, space: int):
        """The quad layer to submit this frame, or None while hidden or before the first frame."""
        if not self.visible or self.pose is None or self.uploaded < 0:
            return None

        class Rect(ctypes.Structure):
            _fields_ = [("x", c_int32), ("y", c_int32), ("width", c_int32), ("height", c_int32)]

        class SubImage(ctypes.Structure):
            _fields_ = [("swapchain", c_uint64), ("imageRect", Rect), ("imageArrayIndex", c_uint32)]

        class Extent(ctypes.Structure):
            _fields_ = [("width", c_float), ("height", c_float)]

        class Quad(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("layerFlags", c_uint64),
                ("space", c_uint64),
                ("eyeVisibility", c_uint32),
                ("subImage", SubImage),
                ("pose", _Pose),
                ("size", Extent),
            ]

        centre, quat = self.pose
        pose = _Pose(_Quat(*[float(v) for v in quat]), _Vec3(*[float(v) for v in centre]))
        width_m = float(self.cfg.width_m)
        return Quad(
            type=QUAD_LAYER_TYPE,
            space=space,
            subImage=SubImage(swapchain=self.swapchain, imageRect=Rect(0, 0, self.width, self.height)),
            pose=pose,
            size=Extent(width_m, width_m * self.height / self.width),
        )


def _head(located) -> tuple[np.ndarray, np.ndarray]:
    """Mean eye position and the first eye's orientation, from ``xrLocateViews``."""
    positions = [np.array([v.pose.position.x, v.pose.position.y, v.pose.position.z]) for v in located]
    q = located[0].pose.orientation
    return np.mean(positions, axis=0), rotation_matrix(q.x, q.y, q.z, q.w)


class _LiveView:
    """Layers from ``hub``, plus controller axes, uploaded when the scene or the fit changes."""

    def __init__(self, gl: _Gl, hub: XrHub, hands: _Hands | None) -> None:
        self.gl = gl
        self.hub = hub
        self.hands = hands
        self.program = _program(gl)
        self.line_vao, self.line_vbo = _geometry_buffer(gl)
        self.tri_vao, self.tri_vbo = _geometry_buffer(gl)
        self.hand_vao, self.hand_vbo = _geometry_buffer(gl)
        self.line_count = 0
        self.tri_count = 0
        self.hand_count = 0
        self._key: tuple[int, int] | None = None
        self.capture = PoseCapture({})
        self._mode = ""

    def before_draw(self, display_time: int, local_space: int) -> None:
        """Poll the controllers, refit if a button edge asks for it, and refresh the buffers."""
        scene = self.hub.scene()
        anchors = {name: np.asarray(point, dtype=float) for name, point in scene["anchors"].items()}
        self.capture.min_spread_m = 0.012 if anchors and all(name.startswith("marker_") for name in anchors) else 0.05
        self.capture.set_anchors(anchors)
        if self.hands is not None:
            edge, markers = self.hands.poll(display_time, local_space)
            for name, (pose, tracked) in self.hands.devices.items():
                self.hub.set_tracked(name, pose, tracked=tracked)
            if self.hands.buttons:
                self.hub.set_buttons("right", self.hands.buttons)
            # With a saved calibration the right trigger and grip belong to teleop (gripper), not to touch capture.
            if edge is not None and self.hub.calibration() is None:
                kind, point = edge
                message = self.capture.undo() if kind == "undo" else self.capture.capture(point)
                logger.info("{}", message)
            hand = markers
        else:
            hand = np.zeros((0, 7), dtype=np.float32)
        calibration = self.hub.calibration()
        if calibration is not None and calibration.get("space", self.hub.space) != self.hub.space:
            mode = f"ignored: calibration is for {calibration.get('space')}, drawing in {self.hub.space}"
            calibration = None
        else:
            mode = ""
        if calibration is not None:
            matrix, mode = np.asarray(calibration["xr_from_base"], dtype=float), "gripper-controller calibration"
        elif self.capture.transform is not None:
            matrix = self.capture.transform
            mode = mode or f"touch registration, rms {(self.capture.rms or 0.0) * 1000:.0f} mm"
        else:
            matrix = preview_transform()
            if self.hub.space == "stage":
                matrix[1, 3] += STAGE_PREVIEW_LIFT_M
            mode = mode or "preview in front of the head, no calibration"
        if mode != self._mode:
            self._mode = mode
            logger.info("Headset alignment: {}", mode)
        key = (int(scene["version"]), self.capture.generation)
        if key != self._key:
            lines, triangles = layer_vertices(layers_from_scene(scene["layers"]), matrix)
            _upload(self.gl, self.line_vbo, lines)
            _upload(self.gl, self.tri_vbo, triangles)
            self.line_count = len(lines)
            self.tri_count = len(triangles)
            self._key = key
            logger.info(
                "Scene version {}: {} line vertices, {} triangle vertices",
                scene["version"],
                self.line_count,
                self.tri_count,
            )
        _upload(self.gl, self.hand_vbo, hand)
        self.hand_count = len(hand)


def _poll(xr: _Xr, instance: int, session: int) -> int:
    class Buffer(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("varying", c_char * 4000)]

    class Changed(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("session", c_uint64), ("state", c_uint32), ("time", c_int64)]

    state = 0
    while True:
        event = Buffer(type=16)
        result = xr.xrPollEvent(instance, byref(event))
        if result == 4:  # XR_EVENT_UNAVAILABLE
            return state
        _check(xr.lib, instance, result, "xrPollEvent")
        if event.type == 18:
            changed = ctypes.cast(byref(event), POINTER(Changed)).contents
            if changed.session == session:
                state = changed.state
                logger.info("Session state {}", state)
                if state == READY:
                    return state
    return state


def _one_frame(
    xr, gl, instance, session, space, described, swapchains, images, live: _LiveView, panel: _ScreenPanel | None = None
) -> None:
    class Empty(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p)]

    class FrameState(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("predictedDisplayTime", c_int64),
            ("predictedDisplayPeriod", c_int64),
            ("shouldRender", c_uint32),
        ]

    class Locate(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("viewConfigurationType", c_uint32),
            ("displayTime", c_int64),
            ("space", c_uint64),
        ]

    class ViewState(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("viewStateFlags", c_uint64)]

    class View(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("pose", _Pose), ("fov", _Fov)]

    class Acquire(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p)]

    class WaitImage(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("timeout", c_int64)]

    class Rect(ctypes.Structure):
        _fields_ = [("x", c_int32), ("y", c_int32), ("width", c_int32), ("height", c_int32)]

    class SubImage(ctypes.Structure):
        _fields_ = [("swapchain", c_uint64), ("imageRect", Rect), ("imageArrayIndex", c_uint32)]

    class ProjView(ctypes.Structure):
        _fields_ = [("type", c_uint32), ("next", c_void_p), ("pose", _Pose), ("fov", _Fov), ("subImage", SubImage)]

    class Projection(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("layerFlags", c_uint64),
            ("space", c_uint64),
            ("viewCount", c_uint32),
            ("views", c_void_p),
        ]

    class End(ctypes.Structure):
        _fields_ = [
            ("type", c_uint32),
            ("next", c_void_p),
            ("displayTime", c_int64),
            ("environmentBlendMode", c_uint32),
            ("layerCount", c_uint32),
            ("layers", c_void_p),
        ]

    state = FrameState(type=44)
    _check(xr.lib, instance, xr.xrWaitFrame(session, byref(Empty(type=33)), byref(state)), "xrWaitFrame")
    _check(xr.lib, instance, xr.xrBeginFrame(session, byref(Empty(type=46))), "xrBeginFrame")
    layer_count = 0
    projection = None
    proj_views = None
    layer_list = None
    if state.shouldRender:
        view_count = c_uint32(len(swapchains))
        located = (View * len(swapchains))()
        for view in located:
            view.type = 7
        _check(
            xr.lib,
            instance,
            xr.xrLocateViews(
                session,
                byref(
                    Locate(type=6, viewConfigurationType=PRIMARY_STEREO, displayTime=state.predictedDisplayTime, space=space)
                ),
                byref(ViewState(type=11)),
                len(swapchains),
                byref(view_count),
                ctypes.cast(located, c_void_p),
            ),
            "xrLocateViews",
        )
        live.before_draw(int(state.predictedDisplayTime), space)
        if panel is not None:
            hands = live.hands
            left = hands.devices.get("left") if hands is not None else None
            panel.update(
                _head(located),
                hands.left_pressed if hands is not None else set(),
                hands.left_held if hands is not None else set(),
                None if left is None else left[0],
            )
        proj_views = (ProjView * len(swapchains))()
        for index, (view, swapchain, pics, size) in enumerate(zip(located, swapchains, images, described, strict=True)):
            image_index = c_uint32()
            acquired = xr.xrAcquireSwapchainImage(swapchain, byref(Acquire(type=55)), byref(image_index))
            _check(xr.lib, instance, acquired, "acquire")
            waited = xr.xrWaitSwapchainImage(swapchain, byref(WaitImage(type=56, timeout=int(1e9))))
            _check(xr.lib, instance, waited, "wait image")
            _draw(gl, live, pics[image_index.value], size, view, index)
            gl.bind_framebuffer(GL_FRAMEBUFFER, 0)
            gl.finish()
            _check(xr.lib, instance, xr.xrReleaseSwapchainImage(swapchain, byref(Acquire(type=57))), "release")
            proj_views[index].type = 48
            proj_views[index].pose = view.pose
            proj_views[index].fov = view.fov
            proj_views[index].subImage.swapchain = swapchain
            proj_views[index].subImage.imageRect.width = size[0]
            proj_views[index].subImage.imageRect.height = size[1]
        see_through = bool(_composite.passthrough) or _composite.blend == BLEND_ALPHA
        projection = Projection(
            type=35,
            layerFlags=LAYER_SOURCE_ALPHA if see_through else 0,
            space=space,
            viewCount=len(swapchains),
            views=ctypes.addressof(proj_views),
        )

        class PassLayer(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("flags", c_uint64),
                ("space", c_uint64),
                ("layerHandle", c_uint64),
            ]

        pass_layer = None
        quad = panel.layer(space) if panel is not None else None
        stack = []
        if _composite.passthrough:
            pass_layer = PassLayer(type=PASSTHROUGH_LAYER_TYPE, space=space, layerHandle=_composite.passthrough)
            stack.append(ctypes.addressof(pass_layer))
        stack.append(ctypes.addressof(projection))
        if quad is not None:
            stack.append(ctypes.addressof(quad))
        layer_list = (c_void_p * len(stack))(*stack)
        layer_count = len(stack)
    end = End(
        type=12,
        displayTime=state.predictedDisplayTime,
        environmentBlendMode=_composite.blend,
        layerCount=layer_count,
        layers=ctypes.addressof(layer_list) if layer_list is not None else 0,
    )
    _check(xr.lib, instance, xr.xrEndFrame(session, byref(end)), "xrEndFrame")


_FBO: dict[int, tuple[int, int]] = {}
_FBO_OK: set[int] = set()


def _draw(gl: _Gl, live: _LiveView, texture: int, size: tuple[int, int], view, eye: int) -> None:
    if texture not in _FBO:
        fbo, depth = c_uint32(), c_uint32()
        gl.gen_framebuffers(1, byref(fbo))
        gl.gen_renderbuffers(1, byref(depth))
        gl.bind_renderbuffer(GL_RENDERBUFFER, depth)
        gl.renderbuffer_storage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size[0], size[1])
        _FBO[texture] = (fbo.value, depth.value)
    fbo, depth = _FBO[texture]
    gl.bind_framebuffer(GL_FRAMEBUFFER, fbo)
    gl.framebuffer_texture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    gl.framebuffer_renderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth)
    gl.viewport(0, 0, size[0], size[1])
    status = gl.check_framebuffer(GL_FRAMEBUFFER)
    if texture not in _FBO_OK:
        _FBO_OK.add(texture)
        logger.info("Eye {} framebuffer {:#x} (complete is {:#x}), texture {}", eye, status, GL_FRAMEBUFFER_COMPLETE, texture)
    gl.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    mvp = _projection(view.fov) @ _view(view.pose)
    gl.use_program(live.program)
    flat = np.ascontiguousarray(mvp.T, dtype=np.float32)
    gl.uniform_matrix(gl.get_uniform(live.program, b"u_mvp"), 1, 0, flat.ctypes.data_as(POINTER(c_float)))
    if live.tri_count:
        gl.bind_vertex_array(live.tri_vao)
        gl.draw_arrays(GL_TRIANGLES, 0, live.tri_count)
    if live.line_count:
        gl.bind_vertex_array(live.line_vao)
        gl.draw_arrays(GL_LINES, 0, live.line_count)
    if live.hand_count:
        gl.bind_vertex_array(live.hand_vao)
        gl.draw_arrays(GL_LINES, 0, live.hand_count)


class _Hands:
    """Trigger and grip on either Touch controller, plus an aim-pose marker per hand."""

    PRESS = 0.65

    def __getattr__(self, name: str) -> Any:
        """OpenXR entry points are bound with ``setattr``; this only tells the type checker so."""
        raise AttributeError(name)

    def __init__(self, xr: _Xr, instance: int, session: int) -> None:
        self.xr = xr
        self.instance = instance
        self.session = session
        lib = xr.lib
        for name, restype, argtypes in (
            ("xrStringToPath", c_uint32, [c_uint64, c_char_p, POINTER(c_uint64)]),
            ("xrCreateActionSet", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrCreateAction", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrSuggestInteractionProfileBindings", c_uint32, [c_uint64, c_void_p]),
            ("xrAttachSessionActionSets", c_uint32, [c_uint64, c_void_p]),
            ("xrSyncActions", c_uint32, [c_uint64, c_void_p]),
            ("xrGetActionStateFloat", c_uint32, [c_uint64, c_void_p, c_void_p]),
            ("xrGetActionStatePose", c_uint32, [c_uint64, c_void_p, c_void_p]),
            ("xrCreateActionSpace", c_uint32, [c_uint64, c_void_p, POINTER(c_uint64)]),
            ("xrLocateSpace", c_uint32, [c_uint64, c_uint64, c_int64, c_void_p]),
            ("xrGetCurrentInteractionProfile", c_uint32, [c_uint64, c_uint64, c_void_p]),
            ("xrPathToString", c_uint32, [c_uint64, c_uint64, c_uint32, POINTER(c_uint32), c_char_p]),
        ):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)
        self.left = self._path("/user/hand/left")
        self.right = self._path("/user/hand/right")
        self._subpaths = (c_uint64 * 2)(self.left, self.right)
        self.action_set = self._create_set()
        self.trigger = self._create_action("trigger", "Trigger", 2)
        self.squeeze = self._create_action("squeeze", "Grip", 2)
        self.aim = self._create_action("aim", "Aim", 4)
        self._suggest()
        self._attach()
        self._hand_space: dict[int, int] = {}
        self._down = {
            ("trigger", self.left): False,
            ("trigger", self.right): False,
            ("squeeze", self.left): False,
            ("squeeze", self.right): False,
        }
        self._pose: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.devices: dict[str, tuple[np.ndarray, bool]] = {}
        # Right controller ``trigger`` / ``squeeze`` held state; empty while the app has no input focus.
        self.buttons: dict[str, bool] = {}
        # Left controller buttons that went down this frame, and those held now (``trigger``, ``squeeze``).
        self.left_pressed: set[str] = set()
        self.left_held: set[str] = set()
        self._said: set[str] = set()
        self.ready = False

    @classmethod
    def try_create(cls, xr: _Xr, instance: int, session: int) -> _Hands | None:
        try:
            hands = cls(xr, instance, session)
        except Exception as error:
            logger.warning("Controller registration unavailable: {}", error)
            return None
        logger.info("Controllers bound. Trigger saves the next saved pose, grip removes the last point.")
        return hands

    def _path(self, text: str) -> int:
        handle = c_uint64()
        _check(self.xr.lib, self.instance, self.xrStringToPath(self.instance, text.encode(), byref(handle)), text)
        return handle.value

    def _create_set(self) -> int:
        class Info(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("name", c_char * 64),
                ("localized", c_char * 128),
                ("priority", c_uint32),
            ]

        handle = c_uint64()
        info = Info(type=28, name=b"rio", localized=b"RIO")
        created = self.xrCreateActionSet(self.instance, byref(info), byref(handle))
        _check(self.xr.lib, self.instance, created, "xrCreateActionSet")
        return handle.value

    def _create_action(self, name: str, title: str, kind: int) -> int:
        class Info(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("name", c_char * 64),
                ("actionType", c_uint32),
                ("countSubactionPaths", c_uint32),
                ("subactionPaths", c_void_p),
                ("localized", c_char * 128),
            ]

        handle = c_uint64()
        info = Info(
            type=29,
            name=name.encode(),
            actionType=kind,
            countSubactionPaths=2,
            subactionPaths=ctypes.addressof(self._subpaths),
            localized=title.encode(),
        )
        _check(self.xr.lib, self.instance, self.xrCreateAction(self.action_set, byref(info), byref(handle)), name)
        return handle.value

    def _suggest(self) -> None:
        class Binding(ctypes.Structure):
            _fields_ = [("action", c_uint64), ("path", c_uint64)]

        class Suggest(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("profile", c_uint64),
                ("count", c_uint32),
                ("bindings", c_void_p),
            ]

        pairs = (
            (self.trigger, "/user/hand/left/input/trigger/value"),
            (self.trigger, "/user/hand/right/input/trigger/value"),
            (self.squeeze, "/user/hand/left/input/squeeze/value"),
            (self.squeeze, "/user/hand/right/input/squeeze/value"),
            (self.aim, "/user/hand/left/input/aim/pose"),
            (self.aim, "/user/hand/right/input/aim/pose"),
        )
        poses_only = tuple(pair for pair in pairs if pair[0] == self.aim)
        for profile in ("/interaction_profiles/oculus/touch_controller", "/interaction_profiles/meta/touch_plus_controller"):
            # A runtime rejects the whole list for one bad path; then keep at least the aim poses.
            for chosen in (pairs, poses_only):
                bindings = (Binding * len(chosen))(*((action, self._path(path)) for action, path in chosen))
                info = Suggest(type=51, profile=self._path(profile), count=len(bindings), bindings=ctypes.addressof(bindings))
                result = self.xrSuggestInteractionProfileBindings(self.instance, byref(info))
                if result == 0:
                    logger.info("Controller profile {}: {} bindings", profile, len(chosen))
                    break
                logger.info("Controller profile {} rejected {} bindings ({})", profile, len(chosen), result)

    def _attach(self) -> None:
        class Info(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("count", c_uint32), ("actionSets", c_void_p)]

        sets = (c_uint64 * 1)(self.action_set)
        info = Info(type=60, count=1, actionSets=ctypes.addressof(sets))
        attached = self.xrAttachSessionActionSets(self.session, byref(info))
        _check(self.xr.lib, self.instance, attached, "xrAttachSessionActionSets")

    def create_spaces(self, xr: _Xr, session: int, local_space: int) -> None:
        class Info(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("action", c_uint64),
                ("subactionPath", c_uint64),
                ("pose", _Pose),
            ]

        self.local_space = local_space
        for hand in (self.left, self.right):
            handle = c_uint64()
            info = Info(type=38, action=self.aim, subactionPath=hand, pose=_identity_pose())
            _check(xr.lib, self.instance, self.xrCreateActionSpace(session, byref(info), byref(handle)), "xrCreateActionSpace")
            self._hand_space[hand] = handle.value
        self.ready = True

    def poll(self, display_time: int, local_space: int) -> tuple[tuple[str, np.ndarray] | None, np.ndarray]:
        """Return a button edge, if one rose, and the aim-pose axes to draw."""
        empty = np.zeros((0, 7), dtype=np.float32)
        if not self.ready:
            return None, empty

        class Active(ctypes.Structure):
            _fields_ = [("actionSet", c_uint64), ("subactionPath", c_uint64)]

        class Sync(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("count", c_uint32), ("sets", c_void_p)]

        active = Active(actionSet=self.action_set)
        synced = self.xrSyncActions(self.session, byref(Sync(type=61, count=1, sets=ctypes.addressof(active))))
        if synced != 0:
            # 8 is XR_SESSION_NOT_FOCUSED, e.g. the system menu is open. Stale poses must not look fresh.
            self._pose.clear()
            self.devices.clear()
            self.buttons.clear()
            self.left_pressed = set()
            self.left_held = set()
            self._say_once(f"xrSyncActions returned {synced}")
            return None, empty
        self._read_poses(display_time, local_space)
        edge = self._edge()
        markers = []
        right = self._pose.get(self.right)
        if right is not None:
            tip, rotation, origin = right
            stick = np.array([[*origin, 1.0, 0.9, 0.2, 1.0], [*tip, 1.0, 0.9, 0.2, 1.0]], dtype=np.float32)
            markers.append(stick)
            markers.append(pose_axes(tip, rotation, 0.02))
        return edge, np.concatenate(markers) if markers else empty

    def profiles(self) -> dict[str, str]:
        """Interaction profile the runtime currently reports per hand; empty when none is active."""

        class State(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("interactionProfile", c_uint64)]

        out = {}
        for side, hand in (("left", self.left), ("right", self.right)):
            state = State(type=53)
            if self.xrGetCurrentInteractionProfile(self.session, hand, byref(state)) != 0 or not state.interactionProfile:
                out[side] = ""
                continue
            text = ctypes.create_string_buffer(256)
            size = c_uint32()
            self.xrPathToString(self.instance, state.interactionProfile, 256, byref(size), text)
            out[side] = text.value.decode()
        return out

    def _say_once(self, reason: str) -> None:
        """Log why a controller pose is missing, once per distinct reason."""
        if reason not in self._said:
            self._said.add(reason)
            logger.info("Controller pose missing: {}", reason)

    def _read_poses(self, display_time: int, local_space: int) -> None:
        class Get(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("action", c_uint64), ("subactionPath", c_uint64)]

        class PoseState(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("isActive", c_uint32)]

        class Location(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("flags", c_uint64), ("pose", _Pose)]

        self._pose.clear()
        self.devices.clear()
        for hand, space in self._hand_space.items():
            state = PoseState(type=27)
            query = Get(type=58, action=self.aim, subactionPath=hand)
            side = "left" if hand == self.left else "right"
            got = self.xrGetActionStatePose(self.session, byref(query), byref(state))
            if got != 0:
                self._say_once(f"{side}: xrGetActionStatePose returned {got}")
                continue
            if not state.isActive:
                self._say_once(f"{side}: aim action inactive (no binding for this controller profile?)")
                continue
            located = Location(type=42)
            got = self.xrLocateSpace(space, local_space, display_time, byref(located))
            if got != 0:
                self._say_once(f"{side}: xrLocateSpace returned {got}")
                continue
            flags = int(located.flags)
            if flags & LOCATION_VALID != LOCATION_VALID:
                self._say_once(f"{side}: location flags {flags:#x}")
                continue
            pose = located.pose
            position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
            rotation = rotation_matrix(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = position
            self.devices[side] = (matrix, flags & LOCATION_TRACKED == LOCATION_TRACKED)
            tip = position - rotation[:, 2] * POINTER_M
            self._pose[hand] = (tip, rotation, position)

    def _edge(self) -> tuple[str, np.ndarray] | None:
        class Get(ctypes.Structure):
            _fields_ = [("type", c_uint32), ("next", c_void_p), ("action", c_uint64), ("subactionPath", c_uint64)]

        class FloatState(ctypes.Structure):
            _fields_ = [
                ("type", c_uint32),
                ("next", c_void_p),
                ("current", c_float),
                ("changed", c_uint32),
                ("when", c_int64),
                ("isActive", c_uint32),
            ]

        found: tuple[str, np.ndarray] | None = None
        self.left_pressed = set()
        self.left_held = set()
        for name, action in (("trigger", self.trigger), ("squeeze", self.squeeze)):
            for hand in (self.right, self.left):
                state = FloatState(type=24)
                query = Get(type=58, action=action, subactionPath=hand)
                result = self.xrGetActionStateFloat(self.session, byref(query), byref(state))
                down = result == 0 and bool(state.isActive) and float(state.current) >= self.PRESS
                key = (name, hand)
                rose = down and not self._down[key]
                self._down[key] = down
                if hand == self.left:
                    if rose:
                        self.left_pressed.add(name)
                    if down:
                        self.left_held.add(name)
                    continue
                self.buttons[name] = down
                if not rose or found is not None:
                    continue
                if name == "squeeze":
                    found = ("undo", np.zeros(3))
                elif hand in self._pose:
                    found = ("save", self._pose[hand][0])
        return found


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
