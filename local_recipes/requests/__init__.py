"""
Local p4a recipe for requests, pinned to 2.25.1.

Why this exists: requests has no built-in python-for-android recipe, so
it's normally installed via p4a's generic fallback for "extra pure
Python dependencies" - here, pulled in transitively by Kivy's own
requires_dist (Kivy declares a bare, unversioned "requests" dependency;
we don't use requests directly anywhere in this app).

An unpinned "requests" resolves to a recent release, which depends on
charset-normalizer>=3.4 or similar - newer than the 3.3.2 our
local_recipes/charset-normalizer/ recipe pins (3.3.2 is the last
charset-normalizer release with only a universal py3-none-any wheel;
3.4+ adds an optional mypyc-compiled platform-specific wheel that trips
a separate, confirmed p4a bug - see that recipe's docstring). Even with
our charset-normalizer recipe installed, p4a's generic transitive-
dependency resolver still independently tries to satisfy requests' own
newer floor, re-triggering the exact same bug via a second, unrelated
charset-normalizer install attempt.

requests==2.25.1 (Dec 2020) predates requests' switch to
charset-normalizer entirely - it depends only on chardet, idna, urllib3,
and certifi (confirmed via PyPI metadata), all of which are universal
"none-any" wheels with no platform-specific-wheel problem. Kivy's own
requires_dist has no version constraint on requests, so any version
satisfies it. Pinning here removes the charset-normalizer dependency
from the graph entirely, rather than continuing to chase which
charset-normalizer version might satisfy every constraint at once.
"""

from pythonforandroid.recipe import PythonRecipe


class RequestsRecipe(PythonRecipe):
    version = '2.25.1'
    url = (
        'https://files.pythonhosted.org/packages/6b/47/'
        'c14abc08432ab22dc18b9892252efaf005ab44066de871e72a38d6af464b/'
        'requests-{version}.tar.gz'
    )
    depends = ['setuptools']
    site_packages_name = 'requests'
    call_hostpython_via_targetpython = False


recipe = RequestsRecipe()
