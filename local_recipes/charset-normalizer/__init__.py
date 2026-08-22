"""
Local p4a recipe for charset-normalizer, pinned to 3.3.2.

Why this exists: charset-normalizer has no built-in python-for-android
recipe, so it's normally installed via p4a's generic fallback for
"extra pure Python dependencies" (packages discovered transitively -
here, from Kivy's own "requests"/"filetype" requirements, not anything
this app uses directly). That fallback path (pythonforandroid/build.py's
run_pymodules_install) has two problems for this specific package:

  1. It silently drops user version pins for any package without a
     dedicated recipe (confirmed by testing: "charset-normalizer==3.3.2"
     in buildozer.spec's requirements= had zero effect on what actually
     got resolved/installed).
  2. Its final `pip install --target ...` call omits the
     --platform/--python-version overrides it used moments earlier to
     *resolve* packages, so it correctly rejects the Android-tagged
     wheel it just picked for itself ("... is not a supported wheel on
     this platform").

charset-normalizer 3.4.0+ ships an optional mypyc-compiled wheel per
platform (including Android), which is what triggers problem #2. 3.3.2
(Oct 2023) is the last release with only a universal py3-none-any wheel.

Giving it a real recipe sidesteps both problems: recipe-based version
pins ARE respected (as seen throughout this project's numpy/kivy/python3
pins), and recipes install through their own build_arch, not the buggy
generic fallback.
"""

from pythonforandroid.recipe import PythonRecipe


class CharsetNormalizerRecipe(PythonRecipe):
    version = '3.3.2'
    url = (
        'https://files.pythonhosted.org/packages/63/09/'
        'c1bc53dab74b1816a00d8d030de5bf98f724c52c1635e07681d312f20be8/'
        'charset-normalizer-{version}.tar.gz'
    )
    depends = ['setuptools']
    site_packages_name = 'charset_normalizer'
    call_hostpython_via_targetpython = False


recipe = CharsetNormalizerRecipe()
