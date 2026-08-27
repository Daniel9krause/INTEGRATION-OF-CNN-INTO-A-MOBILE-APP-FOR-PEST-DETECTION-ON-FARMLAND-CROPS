"""
Local override of p4a's built-in Kivy recipe, identical to upstream 2.3.1
(pythonforandroid/recipes/kivy/__init__.py @ p4a v2026.05.09) except for
one change: 'requests' is removed from python_depends below.

Why this exists: p4a's dependency-resolution assertion
(toolchain.py's `assert set(build_order).intersection(set(python_modules))
== set()`) requires that no package name appear in both:
  - build_order: names that resolved to an actual recipe (built from
    source / a pinned sdist), via `Recipe.get_recipe(name, ctx)`
  - python_modules: names installed generically via pip afterwards

get_recipe_order_and_bootstrap() (pythonforandroid/graph.py) only ever
calls Recipe.get_recipe() on names reachable through recipes' `depends`
graph. A recipe's `python_depends` list is never itself walked back
through Recipe.get_recipe() - it's unconditionally appended to
python_modules as plain strings, regardless of whether one of those
strings also happens to be the name of some OTHER recipe elsewhere in
the build. Upstream Kivy's recipe hardcodes
`python_depends = ['certifi', 'chardet', 'idna', 'requests', 'urllib3',
'filetype']` unconditionally - so as soon as our own buildozer.spec
requirements also list `requests==2.25.1` (giving it a dedicated recipe
via local_recipes/requests/, so build_order contains it, so we can pin
its version and dodge its charset-normalizer dependency - see that
recipe's docstring), 'requests' ends up in both sets simultaneously and
the assertion trips.

We can't edit pythonforandroid's own installed copy of the Kivy recipe,
and there's no per-package way to tell p4a "treat this python_depends
entry as satisfied by a recipe instead of pip" - the only way to change
what Kivy's recipe declares is to shadow the whole recipe by name, which
p4a's local-recipes lookup prefers over its own built-in one. Everything
below is an unmodified copy of upstream except the one line marked; the
three .patch files in this same folder are unmodified copies of
upstream's, since Kivy's own patch-application looks for them next to
whichever __init__.py defined the recipe that's actually running.
"""

from os.path import join
import sys
import packaging.version

import sh
from pythonforandroid.recipe import PyProjectRecipe
from pythonforandroid.toolchain import current_directory, shprint


def get_kivy_version(recipe, arch):
    with current_directory(join(recipe.get_build_dir(arch.arch), "kivy")):
        return shprint(
            sh.Command(sys.executable),
            "-c",
            "import _version; print(_version.__version__)",
        )


def is_kivy_affected_by_deadlock_issue(recipe=None, arch=None):
    return packaging.version.parse(
        str(get_kivy_version(recipe, arch))
    ) < packaging.version.Version("2.2.0.dev0")


def is_kivy_less_than_3(recipe=None, arch=None):
    return packaging.version.parse(
        str(get_kivy_version(recipe, arch))
    ) < packaging.version.Version("3.0.0.dev0")


class KivyRecipe(PyProjectRecipe):
    version = '2.3.1'
    url = 'https://github.com/kivy/kivy/archive/{version}.zip'
    name = 'kivy'

    depends = [('sdl2', 'sdl3'), 'pyjnius', 'setuptools', 'android']
    # 'requests' removed here (see module docstring) - it's provided
    # instead by our own pinned local_recipes/requests/ recipe.
    python_depends = ['certifi', 'chardet', 'idna', 'urllib3', 'filetype']
    hostpython_prerequisites = ["cython>=0.29.1,<=3.0.12"]

    # sdl-gl-swapwindow-nogil.patch is needed to avoid a deadlock.
    # See: https://github.com/kivy/kivy/pull/8025
    # WARNING: Remove this patch when a new Kivy version is released.
    patches = [
        ("sdl-gl-swapwindow-nogil.patch", is_kivy_affected_by_deadlock_issue),
        ("use_cython.patch", is_kivy_less_than_3),
        "no-ast-str.patch"
    ]

    @property
    def need_stl_shared(self):
        if "sdl3" in self.ctx.recipe_build_order:
            return True
        else:
            return False

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)

        # Taken from CythonRecipe
        env['LDFLAGS'] = env['LDFLAGS'] + ' -L{} '.format(
            self.ctx.get_libs_dir(arch.arch) +
            ' -L{} '.format(self.ctx.libs_dir) +
            ' -L{}'.format(join(self.ctx.bootstrap.build_dir, 'obj', 'local',
                                arch.arch)))
        env['LDSHARED'] = env['CC'] + ' -shared'
        env['LIBLINK'] = 'NOTNONE'

        # NDKPLATFORM is our switch for detecting Android platform, so can't be None
        env['NDKPLATFORM'] = "NOTNONE"
        if not is_kivy_less_than_3(self, arch):
            env['KIVY_CROSS_PLATFORM'] = 'android'

        if 'sdl2' in self.ctx.recipe_build_order:
            env['USE_SDL2'] = '1'
            env['KIVY_SPLIT_EXAMPLES'] = '1'
            sdl2_mixer_recipe = self.get_recipe('sdl2_mixer', self.ctx)
            sdl2_image_recipe = self.get_recipe('sdl2_image', self.ctx)
            env['KIVY_SDL2_PATH'] = ':'.join([
                join(self.ctx.bootstrap.build_dir, 'jni', 'SDL', 'include'),
                *sdl2_image_recipe.get_include_dirs(arch),
                *sdl2_mixer_recipe.get_include_dirs(arch),
                join(self.ctx.bootstrap.build_dir, 'jni', 'SDL2_ttf'),
            ])
        if "sdl3" in self.ctx.recipe_build_order:
            sdl3_mixer_recipe = self.get_recipe("sdl3_mixer", self.ctx)
            sdl3_image_recipe = self.get_recipe("sdl3_image", self.ctx)
            sdl3_ttf_recipe = self.get_recipe("sdl3_ttf", self.ctx)
            sdl3_recipe = self.get_recipe("sdl3", self.ctx)
            env["USE_SDL3"] = "1"
            env["KIVY_SPLIT_EXAMPLES"] = "1"
            env["KIVY_SDL3_PATH"] = ":".join(
                [
                    *sdl3_mixer_recipe.get_include_dirs(arch),
                    *sdl3_image_recipe.get_include_dirs(arch),
                    *sdl3_ttf_recipe.get_include_dirs(arch),
                    *sdl3_recipe.get_include_dirs(arch),
                ]
            )

        return env


recipe = KivyRecipe()
